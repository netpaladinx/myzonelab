import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cv2

from .coco_utils import get_img, seg2mask, npf
from .coco_consts import SKELETON, KEYPOINT_NAMES


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


get_color = Colors()


def draw_anns(anns,
              img=None,
              coco_data=None,
              draw_bbox=True,
              draw_seg=True,
              draw_kpts=True,
              show_kpt_names=False,
              dpi=120,
              line_width=2,
              marker_scale=1.,
              seed=None,
              save_file=None):
    # use matplotlib to draw

    if seed is not None:
        np.random.seed(seed)

    def random_color():
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        return c

    img = get_img(anns[0], img, coco_data)
    img_h, img_w = img.shape[:2]
    fig = Figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi, frameon=False, tight_layout=dict(pad=0))
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    ax.set_autoscale_on(True)
    ax.axis('off')
    bbox_polygons = []
    bbox_color = []
    seg_polygons = []
    seg_color = []

    ax.imshow(img)

    for ann in anns:
        c = random_color()

        bbox = ann.get('bbox')
        if draw_bbox and bbox:
            x0, y0, w, h = bbox[:4]
            x0, y0, w, h = int(x0), int(y0), int(w), int(h)
            poly = [[x0, y0], [x0, y0 + h], [x0 + w, y0 + h], [x0 + w, y0]]
            bbox_polygons.append(Polygon(npf(poly).reshape((4, 2))))
            bbox_color.append(c)

        seg = ann.get('segmentation')
        if draw_seg and seg:
            if isinstance(seg, list):  # polygon
                for s in seg:
                    seg_polygons.append(Polygon(npf(s).reshape((-1, 2))))
                    seg_color.append(c)
            else:  # RLE
                mask = seg2mask(seg, img_w, img_h)
                mask_h, mask_w = mask.shape[:2]
                mask_img = np.ones((mask_h, mask_w, 3))
                for i in range(3):
                    mask_img[:, :, i] = c[i]
                mask_img = np.dstack((mask_img, mask * 0.5))
                ax.imshow(mask_img)

        kpts = ann.get('keypoints')
        if draw_kpts and kpts:
            kpts = npf(kpts)
            x, y, v = kpts[0::3], kpts[1::3], kpts[2::3]
            for sk in SKELETON:
                if np.all(v[sk] > 0):
                    ax.plot(x[sk], y[sk], linewidth=line_width, color=c)
            ax.plot(x[v > 0], y[v > 0], 'o',
                    markersize=4 * marker_scale, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2 * marker_scale)
            ax.plot(x[v > 1], y[v > 1], 'o',
                    markersize=4 * marker_scale, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2 * marker_scale)
            if show_kpt_names:
                for i, (tx, ty) in enumerate(zip(x, y)):
                    if v[i] > 0:
                        ax.text(tx + .03, ty + .03, KEYPOINT_NAMES[i], color='w', fontsize='small')

    if seg_polygons:
        p = PatchCollection(seg_polygons, facecolor=seg_color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(seg_polygons, facecolor='none', edgecolors=seg_color, linewidths=2)
        ax.add_collection(p)

    if bbox_polygons:
        p = PatchCollection(bbox_polygons, facecolor='none', edgecolors=bbox_color, linewidths=2)
        ax.add_collection(p)

    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)

    if save_file:
        cv2.imwrite(save_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img


def draw_results(results,
                 img,
                 kpt_radius=4,
                 link_thickness=2,
                 text_thickness=2,
                 font_scale=0.5,
                 bbox_thickness=2,
                 kpt_score_thr=0.,
                 random_bbox_color=False,
                 blend_bbox_color=False,
                 show_kpt_confs=False,
                 seed=None,
                 save_file=None):
    # use cv2 to draw

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                        [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                        [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]])

    gradient = np.array([[255, 0, 0], [255, 30, 0], [255, 60, 0], [255, 90, 0], [255, 120, 0], [255, 150, 0],
                         [255, 200, 0], [200, 200, 0], [100, 200, 0], [0, 200, 0], [0, 255, 0]])

    default_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    default_link_color = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]

    if seed is not None:
        np.random.seed(seed)

    def to_color(c):
        r, g, b = c
        return (int(r), int(g), int(b))

    def put_text(img, label, x, y, color, font_scale, thickness):
        text_size, text_baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        text_x0 = x
        text_y0 = max(0, y - text_size[1] - text_baseline)
        text_x1 = x + text_size[0]
        text_y1 = text_y0 + text_size[1] + text_baseline
        cv2.rectangle(img, (text_x0, text_y0), (text_x1, text_y1), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, label, (text_x0, text_y1 - text_baseline), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness)
        return text_x1 + text_size[1], text_y0

    img = get_img(img=img)
    assert isinstance(img, np.ndarray) and img.ndim == 3
    img = img.copy()

    n_results = len(results)
    if random_bbox_color:
        bbox_colors = [to_color(c) for c in palette[np.random.choice(len(palette), n_results)]]
    else:
        bbox_colors = [to_color(c) for c in palette[[0] * n_results]]

    if blend_bbox_color:
        pose_kpt_color = [np.tile([c], (len(KEYPOINT_NAMES), 1)) * 0.5 + default_kpt_color * 0.5 for c in bbox_colors]
        pose_link_color = [np.tile([c], (len(SKELETON), 1)) * 0.5 + default_link_color * 0.5 for c in bbox_colors]
    else:
        pose_kpt_color = [default_kpt_color] * n_results
        pose_link_color = [default_link_color] * n_results

    x0, y0, x1, y1 = -1, -1, -1, -1
    for i, res in enumerate(results):
        bbox = res.get('bbox')
        bbox_conf = None
        if bbox is not None:
            x0, y0, w, h = bbox[:4]
            x0, y0, w, h = int(x0), int(y0), int(w), int(h)
            x1, y1 = x0 + w, y0 + h
            cv2.rectangle(img, (x0, y0), (x1, y1), bbox_colors[i], thickness=bbox_thickness)

            if len(bbox) >= 5:
                bbox_conf = bbox[4]

        kpts = res.get('keypoints')
        kpts_conf = None
        if kpts is not None:
            kpts = npf(kpts).reshape(-1, 3)
            kpts_conf = []
            for j, kpt in enumerate(kpts):
                x, y, s = kpt
                x, y = int(x), int(y)
                if s > kpt_score_thr:
                    x0 = min(x if x0 == -1 else x0, x)
                    y0 = min(y if y0 == -1 else y0, y)
                    x1 = max(x if x1 == -1 else x1, x)
                    y1 = max(y if y1 == -1 else y1, y)
                    kpts_conf.append(s)
                    kpt_color = to_color(pose_kpt_color[i][j])
                    cv2.circle(img, (x, y), kpt_radius, kpt_color, -1)
                    if show_kpt_confs:
                        put_text(img, f'{s: .3f}', x, y, kpt_color, font_scale * 0.5, text_thickness * 0.5)

            kpts_conf = npf(kpts_conf).mean() if kpts_conf else None

            for j, sk in enumerate(SKELETON):
                p0_x, p0_y, p0_s = int(kpts[sk[0], 0]), int(kpts[sk[0], 1]), kpts[sk[0], 2]
                p1_x, p1_y, p1_s = int(kpts[sk[1], 0]), int(kpts[sk[1], 1]), kpts[sk[1], 2]
                if p0_s > kpt_score_thr and p1_s > kpt_score_thr:
                    link_color = to_color(pose_link_color[i][j])
                    cv2.line(img, (p0_x, p0_y), (p1_x, p1_y), link_color, thickness=link_thickness)

        tx, ty = x0, y0
        if bbox_conf is not None:
            label = f'bbox_conf: {bbox_conf: .5f}'
            text_color = to_color(gradient[int(bbox_conf * 10)])
            tx, ty = put_text(img, label, tx, ty, text_color, font_scale, text_thickness)
        if kpts_conf is not None:
            label = f'kpts_conf: {kpts_conf: .5f}'
            text_color = to_color(gradient[int(kpts_conf * 10)])
            tx, ty = put_text(img, label, tx, ty, text_color, font_scale, text_thickness)

        osk = res.get('osk')
        if osk and x0 != -1 and y0 != -1 and x1 != -1 and y1 != -1:
            label = f'osk: {osk: .5f}'
            text_color = to_color(gradient[int(osk * 10)])
            put_text(img, label, tx, ty, text_color, font_scale, text_thickness)

    if save_file:
        cv2.imwrite(save_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img
