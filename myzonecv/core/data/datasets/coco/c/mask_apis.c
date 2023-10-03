#include <stdlib.h>
#include <math.h>
#include "mask_apis.h"

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

void rleInit(RLE *p_rle, uint height, uint width, uint n_counts, uint *counts)
{
    uint i;
    p_rle->height = height;
    p_rle->width = width;
    p_rle->n_counts = n_counts;
    p_rle->counts = (n_counts == 0) ? 0 : malloc(sizeof(uint) * n_counts);

    if (n_counts > 0)
        for (i = 0; i < n_counts; i++)
            p_rle->counts[i] = counts[i];
}

void rleFree(RLE *p_rle)
{
    free(p_rle->counts);
    p_rle->counts = 0;
}

void rlesInit(RLE **p_rles, uint n_rles)
{
    uint i;
    *p_rles = (RLE *)malloc(sizeof(RLE) * n_rles);

    for (i = 0; i < n_rles; i++)
        rleInit((*p_rles) + i, 0, 0, 0, 0);
}

void rlesFree(RLE **p_rles, uint n_rles)
{
    uint i;

    for (i = 0; i < n_rles; i++)
        rleFree((*p_rles) + i);

    free(*p_rles);
    *p_rles = 0;
}

void rleEncode(RLE *rles, const uchar *masks, uint height, uint width, uint n_rles)
{
    uint pixels = width * height;
    uint i, j, k;
    uint *counts, c;
    uchar v;

    counts = malloc(sizeof(uint) * (pixels + 1));
    for (i = 0; i < n_rles; i++)
    {
        const uchar *mask = masks + pixels * i;
        k = 0;
        v = 0;
        c = 0;
        for (j = 0; j < pixels; j++)
        {
            if (mask[j] != v)
            {
                counts[k++] = c;
                c = 0;
                v = mask[j];
            }
            c++;
        }
        counts[k++] = c;
        rleInit(rles + i, height, width, k, counts);
    }
    free(counts);
}

void rleDecode(const RLE *rles, uchar *masks, uint n_rles)
{
    uint i, j, k;
    uchar *p = masks;

    for (i = 0; i < n_rles; i++)
    {
        uchar v = 0;
        for (j = 0; j < rles[i].n_counts; j++)
        {
            for (k = 0; k < rles[i].counts[j]; k++)
                *(p++) = v;
            v = !v;
        }
    }
}

void rleMerge(const RLE *rles, uint n_rles, RLE *out_rle, uint intersect)
{
    uint height = rles[0].height;
    uint width = rles[0].width;
    uint n_counts = rles[0].n_counts;

    uint *counts;
    uint c, a_c, b_c, out_c, next_c;
    uint i, a_i, b_i;
    uchar v, a_v, b_v, prev_v;
    RLE rle_a, rle_b;

    if (n_rles == 0)
    {
        rleInit(out_rle, 0, 0, 0, 0);
        return;
    }
    if (n_rles == 1)
    {
        rleInit(out_rle, height, width, n_counts, rles[0].counts);
        return;
    }

    counts = malloc(sizeof(uint) * (height * width + 1));
    for (i = 0; i < n_counts; i++)
        counts[i] = rles[0].counts[i];

    for (i = 1; i < n_rles; i++)
    {
        rle_b = rles[i];
        if (rle_b.height != height || rle_b.width != width)
        {
            height = width = n_counts = 0;
            break;
        }
        rleInit(&rle_a, height, width, n_counts, counts);

        a_i = b_i = 0;
        a_c = rle_a.counts[a_i++];
        b_c = rle_b.counts[b_i++];
        v = a_v = b_v = 0;
        n_counts = 0;

        out_c = 0;
        next_c = a_c + b_c;
        while (next_c > 0)
        {
            c = MIN2(a_c, b_c);
            out_c += c;
            next_c = 0;

            a_c -= c;
            if (!a_c && a_i < rle_a.n_counts)
            {
                a_c = rle_a.counts[a_i++];
                a_v = !a_v;
            }
            next_c += a_c;

            b_c -= c;
            if (!b_c && b_i < rle_b.n_counts)
            {
                b_c = rle_b.counts[b_i++];
                b_v = !b_v;
            }
            next_c += b_c;

            prev_v = v;
            if (intersect)
                v = a_v && b_v;
            else
                v = a_v || b_v;

            if (v != prev_v || next_c == 0)
            {
                counts[n_counts++] = out_c;
                out_c = 0;
            }
        }
        rleFree(&rle_a);
    }
    rleInit(out_rle, height, width, n_counts, counts);
    free(counts);
}

void rleDiff(const RLE *rles, uint n_rles, RLE *out_rle)
{
    uint height = rles[0].height;
    uint width = rles[0].width;
    uint n_counts = rles[0].n_counts;

    uint *counts;
    uint c, a_c, b_c, out_c, next_c;
    uint i, a_i, b_i;
    uchar v, a_v, b_v, prev_v;
    RLE rle_a, rle_b;

    if (n_rles == 0)
    {
        rleInit(out_rle, 0, 0, 0, 0);
        return;
    }
    if (n_rles == 1)
    {
        rleInit(out_rle, height, width, n_counts, rles[0].counts);
        return;
    }

    counts = malloc(sizeof(uint) * (height * width + 1));
    for (i = 0; i < n_counts; i++)
        counts[i] = rles[0].counts[i];

    for (i = 1; i < n_rles; i++)
    {
        rle_b = rles[i];
        if (rle_b.height != height || rle_b.width != width)
        {
            height = width = n_counts = 0;
            break;
        }
        rleInit(&rle_a, height, width, n_counts, counts);

        a_i = b_i = 0;
        a_c = rle_a.counts[a_i++];
        b_c = rle_b.counts[b_i++];
        v = a_v = b_v = 0;
        n_counts = 0;

        out_c = 0;
        next_c = a_c + b_c;
        while (next_c > 0)
        {
            c = MIN2(a_c, b_c);
            out_c += c;
            next_c = 0;

            a_c -= c;
            if (!a_c && a_i < rle_a.n_counts)
            {
                a_c = rle_a.counts[a_i++];
                a_v = !a_v;
            }
            next_c += a_c;

            b_c -= c;
            if (!b_c && b_i < rle_b.n_counts)
            {
                b_c = rle_b.counts[b_i++];
                b_v = !b_v;
            }
            next_c += b_c;

            prev_v = v;
            v = a_v && !b_v;

            if (v != prev_v || next_c == 0)
            {
                counts[n_counts++] = out_c;
                out_c = 0;
            }
        }
        rleFree(&rle_a);
    }
    rleInit(out_rle, height, width, n_counts, counts);
    free(counts);
}

void rleArea(const RLE *rles, uint n_rles, uint *areas)
{
    uint i, j;
    for (i = 0; i < n_rles; i++)
    {
        areas[i] = 0;
        for (j = 1; j < rles[i].n_counts; j += 2)
            areas[i] += rles[i].counts[j];
    }
}

void rleIou(RLE *rles_a, RLE *rles_b, uint n_rles_a, uint n_rles_b, double *ious)
{
    uint i, j, k;
    uint a_i, b_i;
    uint c, a_c, b_c, next_c, union_c, inter_c;
    uchar a_v, b_v;

    double *bboxes_a = malloc(sizeof(double) * n_rles_a * 4);
    double *bboxes_b = malloc(sizeof(double) * n_rles_b * 4);
    rleToBbox(rles_a, bboxes_a, n_rles_a);
    rleToBbox(rles_b, bboxes_b, n_rles_b);
    bboxIou(bboxes_a, bboxes_b, n_rles_a, n_rles_b, ious);
    free(bboxes_a);
    free(bboxes_b);

    for (i = 0; i < n_rles_a; i++)
    {
        for (j = 0; j < n_rles_b; j++)
        {
            k = j * n_rles_a + i;
            if (ious[k] > 0)
            {
                if (rles_a[i].height != rles_b[j].height || rles_a[i].width != rles_b[j].width)
                {
                    ious[k] = -1;
                    continue;
                }

                a_i = b_i = 0;
                a_c = rles_a[i].counts[a_i++];
                b_c = rles_b[j].counts[b_i++];
                a_v = b_v = 0;
                union_c = inter_c = 0;

                next_c = a_c + b_c;
                while (next_c > 0)
                {
                    c = MIN2(a_c, b_c);
                    if (a_v || b_v)
                    {
                        union_c += c;
                        if (a_v && b_v)
                            inter_c += c;
                    }
                    next_c = 0;

                    a_c -= c;
                    if (!a_c && a_i < rles_a[i].n_counts)
                    {
                        a_c = rles_a[i].counts[a_i++];
                        a_v = !a_v;
                    }
                    next_c += a_c;

                    b_c -= c;
                    if (!b_c && b_i < rles_b[j].n_counts)
                    {
                        b_c = rles_b[j].counts[b_i++];
                        b_v = !b_v;
                    }
                    next_c += b_c;
                }

                if (inter_c == 0)
                    ious[k] = 0;
                else
                    ious[k] = (double)inter_c / (double)union_c;
            }
        }
    }
}

void rleNms(RLE *rles, uint n_rles, uint *keep, double thr)
{
    uint i, j;
    double iou;

    for (i = 0; i < n_rles; i++)
        keep[i] = 1;

    for (i = 0; i < n_rles; i++)
    {
        if (keep[i])
        {
            for (j = i + 1; j < n_rles; j++)
                if (keep[j])
                {
                    rleIou(rles + i, rles + j, 1, 1, &iou);
                    if (iou > thr)
                        keep[j] = 0;
                }
        }
    }
}

void bboxIou(double *bboxes_a, double *bboxes_b, uint n_bboxes_a, uint n_bboxes_b, double *ious)
{
    uint i, j, k;
    double *bbox_a, *bbox_b;
    double area_a, area_b;
    double a_x0, a_y0, a_w, a_h, b_x0, b_y0, b_w, b_h;
    double w, h;

    for (i = 0; i < n_bboxes_a; i++)
    {
        bbox_a = bboxes_a + i * 4;
        a_x0 = bbox_a[0];
        a_y0 = bbox_a[1];
        a_w = bbox_a[2];
        a_h = bbox_a[3];
        area_a = a_w * a_h;

        for (j = 0; j < n_bboxes_b; j++)
        {
            bbox_b = bboxes_b + j * 4;
            b_x0 = bbox_b[0];
            b_y0 = bbox_b[1];
            b_w = bbox_b[2];
            b_h = bbox_b[3];
            area_b = b_w * b_h;

            k = j * n_bboxes_a + i;
            ious[k] = 0;

            w = MIN2(a_x0 + a_w, b_x0 + b_w) - MAX2(a_x0, b_x0);
            if (w <= 0)
                w = 0;

            h = MIN2(a_y0 + a_h, b_y0 + b_h) - MAX2(a_y0, b_y0);
            if (h <= 0)
                h = 0;

            ious[k] = (w * h) / (area_a + area_b - w * h);
        }
    }
}

void bboxNms(double *bboxes, uint n_bboxes, uint *keep, double thr)
{
    uint i, j;
    double iou;

    for (i = 0; i < n_bboxes; i++)
        keep[i] = 1;

    for (i = 0; i < n_bboxes; i++)
    {
        if (keep[i])
        {
            for (j = i + 1; j < n_bboxes; j++)
                if (keep[j])
                {
                    bboxIou(bboxes + i * 4, bboxes + j * 4, 1, 1, &iou);
                    if (iou > thr)
                        keep[j] = 0;
                }
        }
    }
}

void rleToBbox(const RLE *rles, double *bboxes, uint n_rles)
{
    uint i, j;
    uint height, width, n_counts, n_counts_even;
    uint x_min, y_min, x_max, y_max;
    uint running_idx, border_idx, border_x, border_y, last_enter_x = 0;

    for (i = 0; i < n_rles; i++)
    {
        height = rles[i].height;
        width = rles[i].width;
        n_counts = rles[i].n_counts;
        n_counts_even = ((uint)(n_counts / 2)) * 2;

        x_min = width;
        y_min = height;
        x_max = 0;
        y_max = 0;

        if (n_counts_even == 0)
        {
            bboxes[4 * i + 0] = bboxes[4 * i + 1] = bboxes[4 * i + 2] = bboxes[4 * i + 3] = 0;
            continue;
        }

        running_idx = 0;
        for (j = 0; j < n_counts_even; j++)
        {
            running_idx += rles[i].counts[j];
            border_idx = running_idx - j % 2;
            border_y = border_idx % height;
            border_x = (border_idx - border_y) / height;
            if (j % 2 == 0)
            {
                // enter
                last_enter_x = border_x;
            }
            else
            {
                // exit
                if (last_enter_x < border_x)
                {
                    y_min = 0;
                    y_max = height - 1;
                }
            }
            x_min = MIN2(x_min, border_x);
            x_max = MAX2(x_max, border_x);
            y_min = MIN2(y_min, border_y);
            y_max = MAX2(y_max, border_y);
        }

        // x0, y0, w, h
        bboxes[4 * i + 0] = x_min;
        bboxes[4 * i + 1] = y_min;
        bboxes[4 * i + 2] = x_max - x_min + 1;
        bboxes[4 * i + 3] = y_max - y_min + 1;
    }
}

void rleFrBbox(RLE *rles, const double *bboxes, uint height, uint width, uint n_bboxes)
{
    uint i;
    double x0, y0, x1, y1;
    for (i = 0; i < n_bboxes; i++)
    {
        x0 = bboxes[4 * i + 0];
        y0 = bboxes[4 * i + 1];
        x1 = bboxes[4 * i + 2];
        y1 = bboxes[4 * i + 3];

        double vertices_xy[8] = {x0, y0, x0, y1, x1, y1, x1, y0};
        rleFrPoly(rles + i, vertices_xy, 4, height, width);
    }
}

int uintCompare(const void *a, const void *b)
{
    uint c = *((uint *)a), d = *((uint *)b);
    return c > d ? 1 : (c < d ? -1 : 0);
}

// vertices_xy: (0.0, 0.0) at the corner of the corner pixel of (height x width) due to double
void rleFrPoly(RLE *rle, const double *vertices_xy, uint n_vertices, uint height, uint width)
{
    /* upsample and get discrete points densely along entire boundary */
    uint i, len = 0;
    double up_scale = 5;
    int *x_list, *y_list, *u_list, *v_list;
    int x0, y0, x1, y1, dx, dy, temp, step, offset;
    uint flip; // to make sure: (1) x0 < x1 if dx >= dy (2) y0 < y1 if dx < dy
    double slope, coord_x, coord_y;
    uint *indices_a, *indices_b, idx, prev_idx;

    x_list = malloc(sizeof(int) * (n_vertices + 1));
    y_list = malloc(sizeof(int) * (n_vertices + 1));
    for (i = 0; i < n_vertices; i++)
    {
        // (0, 0) at the center of the corner pixel of ((up_scale*height+1) x (up_scale*width+1)) due to int
        x_list[i] = (int)(up_scale * vertices_xy[i * 2 + 0] + .5);
        y_list[i] = (int)(up_scale * vertices_xy[i * 2 + 1] + .5);
    }
    x_list[n_vertices] = x_list[0];
    y_list[n_vertices] = y_list[0];

    for (i = 0; i < n_vertices; i++)
    {
        len += MAX2(abs(x_list[i] - x_list[i + 1]), abs(y_list[i] - y_list[i + 1])) + 1;
    }

    u_list = malloc(sizeof(int) * len);
    v_list = malloc(sizeof(int) * len);
    len = 0;

    for (i = 0; i < n_vertices; i++)
    {
        x0 = x_list[i];
        y0 = y_list[i];
        x1 = x_list[i + 1];
        y1 = y_list[i + 1];
        dx = abs(x1 - x0);
        dy = abs(y1 - y0);
        flip = (dx >= dy && x0 > x1) || (dx < dy && y0 > y1);
        if (flip)
        {
            temp = x0;
            x0 = x1;
            x1 = temp;
            temp = y0;
            y0 = y1;
            y1 = temp;
        }
        slope = dx >= dy ? (double)(y1 - y0) / dx : (double)(x1 - x0) / dy;
        if (dx >= dy)
        {
            for (step = 0; step <= dx; step++)
            {
                offset = flip ? dx - step : step;
                u_list[len] = x0 + offset;
                v_list[len] = (int)(y0 + offset * slope + 0.5);
                len++;
            }
        }
        else
        {
            for (step = 0; step <= dy; step++)
            {
                offset = flip ? dy - step : step;
                v_list[len] = y0 + offset;
                u_list[len] = (int)(x0 + offset * slope + 0.5);
                len++;
            }
        }
    }

    /* get points along y-boundary and downsample */
    free(x_list);
    free(y_list);
    n_vertices = len;
    len = 0;

    x_list = malloc(sizeof(int) * n_vertices);
    y_list = malloc(sizeof(int) * n_vertices);
    for (i = 1; i < n_vertices; i++)
    {
        if (u_list[i] != u_list[i - 1])
        {
            coord_x = (double)MIN2(u_list[i], u_list[i - 1]);
            // 0/5, 1/5, 2/5 (it counts for this pixel), 3/5, 4/5, 5/5 due to 2/5 + .5/5 - .5 = 0
            coord_x = (coord_x + .5) / up_scale - .5;
            if (floor(coord_x) != coord_x || coord_x < 0 || coord_x > width - 1)
                continue;
            coord_y = (double)MIN2(v_list[i], v_list[i - 1]);
            coord_y = (coord_y + .5) / up_scale - .5;
            if (coord_y < 0)
                coord_y = 0;
            if (coord_y > height - 1)
                coord_y = height - 1;
            x_list[len] = (int)coord_x;
            y_list[len] = (int)coord_y;
            len++;
        }
    }

    /* compute rle encoding given y-boundary points */
    n_vertices = len;
    indices_a = malloc(sizeof(uint) * (n_vertices + 1));
    for (i = 0; i < n_vertices; i++)
        indices_a[i] = (uint)(x_list[i] * (int)(height) + y_list[i]);
    indices_a[n_vertices++] = (uint)(height * width);
    free(u_list);
    free(v_list);
    free(x_list);
    free(y_list);
    qsort(indices_a, n_vertices, sizeof(uint), uintCompare);

    prev_idx = idx = 0;
    for (i = 0; i < n_vertices; i++)
    {
        idx = indices_a[i];
        indices_a[i] -= prev_idx;
        prev_idx = idx;
    }

    indices_b = malloc(sizeof(uint) * n_vertices);
    i = len = 0;
    indices_b[len++] = indices_a[i++];
    while (i < n_vertices)
    {
        if (indices_a[i] > 0)
        {
            indices_b[len++] = indices_a[i++];
        }
        else
        {
            i++;
            if (i < n_vertices)
                indices_b[len - 1] += indices_a[i++];
        }
    }
    rleInit(rle, height, width, len, indices_b);
    free(indices_a);
    free(indices_b);
}

char *rleToString(const RLE *rle)
{
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    uint i, n_counts = rle->n_counts, p = 0;
    long x;
    int more;
    char *s = malloc(sizeof(char) * n_counts * 6);

    for (i = 0; i < n_counts; i++)
    {
        x = (long)rle->counts[i];
        if (i > 2)
            x -= (long)rle->counts[i - 2];
        more = 1;

        while (more)
        {
            char c = x & 0x1f;
            x >>= 5;
            more = (c & 0x10) ? x != -1 : x != 0;
            if (more)
                c |= 0x20;
            c += 48;
            s[p++] = c;
        }
    }
    s[p] = 0;
    return s;
}

void rleFrString(RLE *rle, char *s, uint height, uint width)
{
    uint n_counts = 0, p = 0, k;
    long x;
    int more;
    uint *counts;

    while (s[n_counts])
        n_counts++;

    counts = malloc(sizeof(uint) * n_counts);
    n_counts = 0;

    while (s[p])
    {
        x = 0;
        k = 0;
        more = 1;

        while (more)
        {
            char c = s[p] - 48;
            x |= (c & 0x1f) << 5 * k;
            more = c & 0x20;
            p++;
            k++;
            if (!more && (c & 0x10))
                x |= -1 << 5 * k;
        }

        if (n_counts > 2)
            x += (long)counts[n_counts - 2];
        counts[n_counts++] = (uint)x;
    }

    rleInit(rle, height, width, n_counts, counts);
    free(counts);
}