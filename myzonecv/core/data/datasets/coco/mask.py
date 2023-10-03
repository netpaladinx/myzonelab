from . import _mask

# Interface for manipulating masks stored in RLE (Run-Length Encoding) data format.
#
# RLE is a simple yet efficient format for storing binary masks. RLE first divides a vector (or vectorized image)
# into a series of piecewise constant regions and then for each piece simply stores the length of that piece.
# For example, given M=[0 0 1 1 1 0 1] the RLE counts would be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts
# would be [0 6 1] (note that the odd counts are always the numbers of zeros). Instead of storing the counts directly,
# additional compression is achieved with a variable bitrate representation based on a common scheme called LEB128.
#
# Compression is greatest given large piecewise constant regions. Specifically, the size of the RLE is proportional to
# the number of *boundaries* in M (or for an image the number of boundaries in the y direction). Assuming fairly
# simple shapes, the RLE representation is O(sqrt(n)) where n is number of pixels in the object. Hence space usage
# is substantially lower, especially for large simple objects (large n).
#
# Many common operations on masks can be computed directly using the RLE (without need for decoding). This includes
# computations such as area, union, intersection, etc. All of these operations are linear in the size of the RLE,
# in other words they are O(sqrt(n)) where n is the area of the object. Computing these operations on the original
# mask is O(n). Thus, using the RLE can result in substantial computational savings.


def encode(masks):
    """ masks (numpy array (hxwxn) in column-major order or (hxw)) => RLE data format (list(dict) or dict)
    """
    if len(masks.shape) == 3:
        return _mask.encode(masks)
    elif len(masks.shape) == 2:
        h, w = masks.shape
        return _mask.encode(masks.reshape((h, w, 1), order='F'))[0]


def decode(rles_data):
    """ RLE data format (list(dict) or dict) => masks (numpy array (hxwxn) in column-major order or (hxw)) 
    """
    if type(rles_data) == list:
        return _mask.decode(rles_data)
    else:
        return _mask.decode([rles_data])[:, :, 0]


# compute union or intersection: RLE data format (list(dict)) => merged RLE data format (dict)
# out_rle_data = merge(rles_data, intersect=0)
merge = _mask.merge


# compute: rles_data[0] - rles_data[1] - ... - rles_data[n-1]
# out_rle_data = diff(rles_data)
diff = _mask.diff


def area(rles_data):
    """ compute areas: RLE data format (list(dict) or dict) => areas (numpy array (n) or uint32) 
    """
    if type(rles_data) == list:
        return _mask.area(rles_data)
    else:
        return _mask.area([rles_data])[0]


# compute intersection over union: two RLE data formats (list(dict)) => ious (numpy array (n) or double)
# ious = iou(rles_data_a, rles_data_b)
iou = _mask.iou


def toBbox(rles_data):
    """ get bounding boxes: RLE data format (list(dict) or dict) => bboxes (numpy array (nx4) or (4))
    """
    if type(rles_data) == list:
        return _mask.toBbox(rles_data)
    else:
        return _mask.toBbox([rles_data])[0]


# polygon or bbox or uncompressed RLE data => RLE data format (list(dict))
# rles_data = frPyObjects(pyobj, height, width)
frPyObjects = _mask.frPyObjects


# RLE data format (dict) => uncompressed RLE data (dict)
# uncomp_rle_data = toUncompressedRLE(rle_data)
toUncompressedRLE = _mask.toUncompressedRLE
