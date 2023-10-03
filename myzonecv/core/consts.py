# common consts
IMAGENET_MEAN = 0.447
IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
BORDER_COLOR_VALUE = (114, 114, 114)
BBOX_SCALE_UNIT = 200
BBOX_PADDING_RATIO = 1.25

# eval consts
MAX_DETECTIONS_PER_IMG = {
    'kpts': [20],
    'bbox': [300],
    'seg': [20]
}
EVAL_SCORE_THRES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
EVAL_RECALL_THRES = [
    0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2,
    0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
    0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4,
    0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5,
    0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
    0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7,
    0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
    0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9,
    0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.
]
EVAL_AREA_RANGES = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2]]
EVAL_AREA_LABELS = ['all', 'small', 'medium', 'large']

# COCO consts
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]
COCO_KEYPOINT_INDEX2NAME = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
    [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
]
COCO_KEYPOINT_FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
COCO_KEYPOINT_UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
COCO_KEYPOINT_LOWER_BODY = [11, 12, 13, 14, 15, 16]
COCO_KEYPOINT_WEIGHTS = [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5]
COCO_KEYPOINT_SIGMAS = [.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, 0.107, 0.107, .087, .087, .089, .089]
COCO_PERSON_CAT_ID = (1,)

COCO_EVAL_OKS_HARD_FACTORS = [1, 1.2, 1.5, 2, 3]

# color-adjust consts
CADJUST_SIDE_TRUNCATION = 0.0

CADJUST_IMAGE_STATS = ['r_mean', 'r_std', 'g_mean', 'g_std', 'b_mean', 'b_std', 'brightness', 'contrast', 's_mean', 's_std']

CADJUST_CONTRAST_RANGE = (2, 10)
CADJUST_BIAS_RANGE = (0.25, 0.75)
CADJUST_GAMMA_RANGE = (0.9, 1.1)
CADJUST_SATURATION_RANGE = (0.6, 1.2)

CADJUST_RED_MEAN = 0.35
CADJUST_RED_STD = 0.15
CADJUST_GREEN_MEAN = 0.37
CADJUST_GREEN_STD = 0.19
CADJUST_BLUE_MEAN = 0.3
CADJUST_BLUE_STD = 0.15
CADJUST_BRIGHTNESS = CADJUST_RED_MEAN * 0.299 + CADJUST_GREEN_MEAN * 0.587 + CADJUST_BLUE_MEAN * 0.114
CADJUST_CONTRAST = CADJUST_RED_STD * 0.299 + CADJUST_GREEN_STD * 0.587 + CADJUST_BLUE_STD * 0.114
CADJUST_SATURATION_MEAN = 0.24
CADJUST_SATURATION_STD = 0.1

CADJUST_RED_MEAN_MARGIN = 0         # 0.03**2
CADJUST_RED_STD_MARGIN = 0          # 0.01**2
CADJUST_GREEN_MEAN_MARGIN = 0       # 0.04**2
CADJUST_GREEN_STD_MARGIN = 0        # 0.01**2
CADJUST_BLUE_MEAN_MARGIN = 0        # 0.02**2
CADJUST_BLUE_STD_MARGIN = 0         # 0.01**2
CADJUST_BRIGHTNESS_MARGIN = 0       # 0.03**2
CADJUST_CONTRAST_MARGIN = 0         # 0.01**2
CADJUST_SATURATION_MEAN_MARGIN = 0  # 0.01**2
CADJUST_SATURATION_STD_MARGIN = 0   # 0.01**2
