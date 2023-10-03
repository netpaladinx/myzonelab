# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# intialized Numpy. must do.
np.import_array()


# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memory management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


# Declare the prototype of the C functions in mask_apis.h
cdef extern from "mask_apis.h":
    ctypedef unsigned int uint
    ctypedef unsigned char uchar
    ctypedef struct RLE:
        uint height
        uint width
        uint n_counts
        uint* counts
    void rleInit(RLE *p_rle, uint height, uint width, uint n_counts, uint *counts)
    void rleFree(RLE *p_rle)
    void rlesInit(RLE **p_rles, uint n_rles)
    void rlesFree(RLE **p_rles, uint n_rles)
    void rleEncode(RLE *rles, const uchar *masks, uint height, uint width, uint n_rles)
    void rleDecode(const RLE *rles, uchar *masks, uint n_rles)
    void rleMerge(const RLE *rles, uint n_rles, RLE *out_rle, uint intersect)
    void rleDiff(const RLE *rles, uint n_rles, RLE *out_rle)
    void rleArea(const RLE *rles, uint n_rles, uint *areas)
    void rleIou(RLE *rles_a, RLE *rles_b, uint n_rles_a, uint n_rles_b, double *ious)
    void rleNms(RLE *rles, uint n_rles, uint *keep, double thr)
    void bboxIou(double *bboxes_a, double *bboxes_b, uint n_bboxes_a, uint n_bboxes_b, double *ious)
    void bboxNms(double *bboxes, uint n_bboxes, uint *keep, double thr)
    void rleToBbox(const RLE *rles, double *bboxes, uint n_rles)
    void rleFrBbox(RLE *rles, const double *bboxes, uint height, uint width, uint n_bboxes)
    void rleFrPoly(RLE *rle, const double *vertices_xy, uint n_vertices, uint height, uint width)
    char *rleToString(const RLE *rle)
    void rleFrString(RLE *rle, char *s, uint height, uint width)


# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLEs:
    cdef RLE *_rles
    cdef uint _n_rles

    def __cinit__(self, uint n_rles=0):
        rlesInit(&self._rles, n_rles)
        self._n_rles = n_rles

    def __dealloc__(self):
        if self._rles is not NULL:
            rlesFree(&self._rles, self._n_rles)
    
    def __getattr__(self, key):
        if key == 'n_rles':
            return self._n_rles
        print(key)
        raise AttributeError(key)


# python class to wrap Mask array in C
# the class handles the memory allocation and deallocation
cdef class Masks:
    cdef uchar *_masks
    cdef uint _height
    cdef uint _width
    cdef uint _n_masks

    def __cinit__(self, height, width, n_masks):
        self._masks = <uchar*>malloc(sizeof(uchar) * height * width * n_masks)
        self._height = height
        self._width = width
        self._n_masks = n_masks

    # def __dealloc__(self):
        # the memory management of _mask has been passed to np.ndarray
        # it doesn't need to be freed here

    # called when passing into np.array() and return an np.ndarray in column-major order
    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self._height * self._width * self._n_masks
        # Create a 1D array, and reshape it to fortran/Matlab column-major array
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._masks).reshape(
            (self._height, self._width, self._n_masks), order='F')
        # The _masks allocated by Masks is now handled by ndarray
        PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
        return ndarray


# internal conversion from Python RLEs object to compressed RLE data format (list(dict)) ('counts' a RLE string)
def _toString(RLEs rles):
    cdef uint n_rles = rles.n_rles
    cdef bytes py_string
    cdef char* c_string
    rles_data = []
    for i in range(n_rles):
        c_string = rleToString(<RLE*> &rles._rles[i])
        py_string = c_string
        rles_data.append({
            'size': [rles._rles[i].height, rles._rles[i].width],
            'counts': py_string
        })
        free(c_string)
    return rles_data


# internal conversion from compressed RLE data format (list(dict)) to Python RLEs object
def _frString(rles_data):
    cdef uint n_rles = len(rles_data)
    rles = RLEs(n_rles)
    cdef bytes py_string
    cdef char* c_string
    for i, rle_data in enumerate(rles_data):
        py_string = str.encode(rle_data['counts']) if type(rle_data['counts']) == str else rle_data['counts']
        c_string = py_string
        rleFrString(<RLE*> &rles._rles[i], <char*> c_string, rle_data['size'][0], rle_data['size'][1])
    return rles


# encode masks to compressed RLE data format (list(dict))
def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] masks):
    height, width, n_masks = masks.shape[0], masks.shape[1], masks.shape[2]
    cdef RLEs rles = RLEs(n_masks)
    rleEncode(rles._rles, <uchar*>masks.data, height, width, n_masks)
    rles_data = _toString(rles)
    return rles_data


# decode masks from compressed RLE data format to mask (list(dict))
def decode(rles_data):
    cdef RLEs rles = _frString(rles_data)
    height, width, n_rles = rles._rles[0].height, rles._rles[0].width, rles._n_rles
    masks = Masks(height, width, n_rles)
    rleDecode(<RLE*>rles._rles, masks._masks, n_rles)
    return np.array(masks)


def merge(rles_data, intersect=0):
    cdef RLEs rles = _frString(rles_data)
    cdef RLEs out_rles = RLEs(1)
    rleMerge(<RLE*>rles._rles, <uint> rles._n_rles, <RLE*> out_rles._rles, intersect)
    out_rle_data = _toString(out_rles)[0]
    return out_rle_data


def diff(rles_data):
    cdef RLEs rles = _frString(rles_data)
    cdef RLEs out_rles = RLEs(1)
    rleDiff(<RLE*>rles._rles, <uint> rles._n_rles, <RLE*> out_rles._rles)
    out_rle_data = _toString(out_rles)[0]
    return out_rle_data


def area(rles_data):
    cdef RLEs rles = _frString(rles_data)
    cdef uint* _areas = <uint*> malloc(rles._n_rles * sizeof(uint))
    rleArea(rles._rles, rles._n_rles, _areas)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> rles._n_rles
    areas = np.array((rles._n_rles,), dtype=np.uint8)
    areas = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _areas)
    PyArray_ENABLEFLAGS(areas, np.NPY_OWNDATA)
    return areas


# iou computation. support function overload (RLEs-RLEs and bbox-bbox).
def iou(rles_data_a, rles_data_b, is_rle=None):
    def _preproc(rles_data):
        if len(rles_data) == 0:
            return rles_data

        if type(rles_data) == np.ndarray:  # only for bboxes
            if len(rles_data.shape) == 1:
                rles_data = rles_data.reshape((1, rles_data.shape[0]))
            # check if it's Nx4 bbox
            if not len(rles_data.shape) == 2 or not rles_data.shape[1] == 4:
                raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
            rles_data = rles_data.astype(np.double)

        elif type(rles_data) == list:
            # check if list is in box format and convert it to np.ndarray
            if is_rle is None:
                isbox = all([len(obj) == 4 and (type(obj) == list or type(obj) == np.ndarray) for obj in rles_data])
                isrle = all([type(obj) == dict for obj in rles_data])
            else:
                isbox = not is_rle
                isrle = is_rle

            if isbox:
                rles_data = np.array(rles_data, dtype=np.double)
                if len(rles_data.shape) == 1:
                    rles_data = rles_data.reshape((1, rles_data.shape[0]))
            elif isrle:
                rles_data = _frString(rles_data)
            else:
                raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
        else:
            raise Exception('unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')

        rles = rles_data
        return rles

    def _rleIou(RLEs rles_a, RLEs rles_b, uint n_a, uint n_b, np.ndarray[np.double_t, ndim=1] _ious):
        rleIou(<RLE*> rles_a._rles, <RLE*> rles_b._rles, n_a, n_b, <double*> _ious.data)
    
    def _bboxIou(np.ndarray[np.double_t, ndim=2] rles_a, np.ndarray[np.double_t, ndim=2] rles_b, uint n_a, uint n_b, np.ndarray[np.double_t, ndim=1] _ious):
        bboxIou(<double*> rles_a.data, <double*> rles_b.data, n_a, n_b, <double*>_ious.data)
    
    def _len(rles):
        cdef uint N = 0
        if type(rles) == RLEs:
            N = rles.n_rles
        elif len(rles)==0:
            pass
        elif type(rles) == np.ndarray:
            N = rles.shape[0]
        return N
    
    # simple type checking
    cdef uint n_a, n_b
    
    rles_a = _preproc(rles_data_a)
    rles_b = _preproc(rles_data_b)
    n_a = _len(rles_a)
    n_b = _len(rles_b)
    if n_a == 0 or n_b == 0:
        return []
    if not type(rles_a) == type(rles_b):
        raise Exception('The rles_a and rles_b should have the same data type, either RLEs, list or np.ndarray')

    # define local variables
    cdef double* _ious = <double*> 0
    cdef np.npy_intp shape[1]
    
    # check type and assign iou function
    if type(rles_a) == RLEs:
        _iouFun = _rleIou
    elif type(rles_a) == np.ndarray:
        _iouFun = _bboxIou
    else:
        raise Exception('input data type not allowed.')
    
    _ious = <double*> malloc(n_a * n_b * sizeof(double))
    ious = np.zeros((n_a * n_b,), dtype=np.double)
    shape[0] = <np.npy_intp> n_a * n_b
    ious = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _ious)
    PyArray_ENABLEFLAGS(ious, np.NPY_OWNDATA)
    _iouFun(rles_a, rles_b, n_a, n_b, ious)
    return ious.reshape((n_a, n_b), order='F')


def toBbox(rles_data):
    cdef RLEs rles = _frString(rles_data)
    cdef uint n_rles = rles.n_rles
    cdef double *_bboxes = <double *> malloc(4 * n_rles * sizeof(double))
    rleToBbox(<const RLE*> rles._rles, _bboxes, n_rles)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 4 * n_rles
    bboxes = np.array((1, 4 * n_rles), dtype=np.double)
    bboxes = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bboxes).reshape((n_rles, 4))
    PyArray_ENABLEFLAGS(bboxes, np.NPY_OWNDATA)
    return bboxes


def frBbox(np.ndarray[np.double_t, ndim=2] bboxes, uint height, uint width):
    cdef uint n_bboxes = bboxes.shape[0]
    rles = RLEs(n_bboxes)
    rleFrBbox(<RLE*> rles._rles, <const double *> bboxes.data, height, width, n_bboxes)
    rles_data = _toString(rles)
    return rles_data


def frPoly(poly, uint height, uint width):
    cdef np.ndarray[np.double_t, ndim=1] np_poly
    n_rles = len(poly)
    rles = RLEs(n_rles)
    for i, p in enumerate(poly):
        np_poly = np.array(p, dtype=np.double, order='F')
        rleFrPoly(<RLE*>&rles._rles[i], <const double*> np_poly.data, int(len(p)/2), height, width)
    rles_data = _toString(rles)
    return rles_data


def frUncompressedRLE(ucRles, uint height, uint width):
    cdef np.ndarray[np.uint32_t, ndim=1] counts
    cdef RLE rle
    cdef uint *data
    n_rles = len(ucRles)
    rles_data = []
    for i in range(n_rles):
        rles = RLEs(1)
        counts = np.array(ucRles[i]['counts'], dtype=np.uint32)
        # time for malloc can be saved here but it's fine
        data = <uint*> malloc(len(counts)* sizeof(uint))
        for j in range(len(counts)):
            data[j] = <uint> counts[j]
        rle = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(counts), <uint*> data)
        rles._rles[0] = rle
        rles_data.append(_toString(rles)[0])
    return rles_data


def toUncompressedRLE(rle):
    cdef RLEs rles = _frString([rle])
    counts = []
    for i in range(rles._rles[0].n_counts):
        counts.append(rles._rles[0].counts[i])
    return {
        'size': [rles._rles[0].height, rles._rles[0].width],
        'counts': counts
    }
    

def frPyObjects(pyobj, height, width):
    # encode rle from a list of python objects (bbox, polygon, uncompressed rle)
    if type(pyobj) == np.ndarray:
        rles_data = frBbox(pyobj, height, width)
    elif type(pyobj) == list and hasattr(pyobj[0], '__iter__') and len(pyobj[0]) == 4:
        rles_data = frBbox(pyobj, height, width)
    elif type(pyobj) == list and hasattr(pyobj[0], '__iter__') and len(pyobj[0]) > 4:
        rles_data = frPoly(pyobj, height, width)
    elif type(pyobj) == list and type(pyobj[0]) == dict and 'counts' in pyobj[0] and 'size' in pyobj[0]:
        rles_data = frUncompressedRLE(pyobj, height, width)
    # encode rle from single python object (bbox, polygon, uncompressed rle)
    elif type(pyobj) == list and len(pyobj) == 4:
        rles_data = frBbox([pyobj], height, width)[0]
    elif type(pyobj) == list and len(pyobj) > 4:
        rles_data = frPoly([pyobj], height, width)[0]
    elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj and isinstance(pyobj['counts'], list):
        rles_data = frUncompressedRLE([pyobj], height, width)[0]
    else:
        raise Exception('input type is not supported.')
    return rles_data
