#pragma once

typedef unsigned int uint;
typedef unsigned char uchar;

typedef struct
{
    uint height, width, n_counts;
    uint *counts; // column-major order
} RLE;

/* Initialize/destroy RLE. */
void rleInit(RLE *p_rle, uint height, uint width, uint n_counts, uint *counts);
void rleFree(RLE *p_rle);

/* Initialize/destroy RLE array. */
void rlesInit(RLE **p_rles, uint n_rles);
void rlesFree(RLE **p_rles, uint n_rles);

/* Encode binary masks using RLE. */
void rleEncode(RLE *rles, const uchar *masks, uint height, uint width, uint n_rles);

/* Decode binary masks encoded via RLE. */
void rleDecode(const RLE *rles, uchar *masks, uint n_rles);

/* Compute union or intersection of encoded masks. */
void rleMerge(const RLE *rles, uint n_rles, RLE *out_rle, uint intersect);

/* Compute R[0] - R[1] - ... - R[n-1] */
void rleDiff(const RLE *rles, uint n_rles, RLE *out_rle);

/* Compute area of encoded masks. */
void rleArea(const RLE *rles, uint n_rles, uint *areas);

/* Compute intersection over union between masks. */
void rleIou(RLE *rles_a, RLE *rles_b, uint n_rles_a, uint n_rles_b, double *ious);

/* Compute non-maximum suppression between bounding masks */
void rleNms(RLE *rles, uint n_rles, uint *keep, double thr);

/* Compute intersection over union between bounding boxes. */
void bboxIou(double *bboxes_a, double *bboxes_b, uint n_bboxes_a, uint n_bboxes_b, double *ious);

/* Compute non-maximum suppression between bounding boxes */
void bboxNms(double *bboxes, uint n_bboxes, uint *keep, double thr);

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox(const RLE *rles, double *bboxes, uint n_rles);

/* Convert bounding boxes to encoded masks. */
void rleFrBbox(RLE *rles, const double *bboxes, uint height, uint width, uint n_bboxes);

/* Convert polygon to encoded mask. */
void rleFrPoly(RLE *rle, const double *vertices_xy, uint n_vertices, uint height, uint width);

/* Get compressed string representation of encoded mask. */
char *rleToString(const RLE *rle);

/* Convert from compressed string representation of encoded mask. */
void rleFrString(RLE *rle, char *s, uint height, uint width);
