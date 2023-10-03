import argparse

from myzonecv.apis import view_dataset

DATASET = '20220426_FightFlow_RetrainingData_Error_0-350'
# DATASET = 'FighterBBoxes_Conf0.95x2_Interval3_20211103-20211113'

DATASET_PATH = f'./workspace/data_zoo/trainval/{DATASET}'

SHOW_BBOX = True
SHOW_KPTS = True
SHOW_SEG = False

DISPLAY = False
FPS = 100

SAVE_DIR = f'./workspace/data_zoo/visualized/{DATASET}'
SAVE_MODE = 'by_annotation'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default=DATASET_PATH)
    parser.add_argument('--show-bbox', action='store_true', default=SHOW_BBOX)
    parser.add_argument('--show-kpts', action='store_true', default=SHOW_KPTS)
    parser.add_argument('--show-seg', action='store_true', default=SHOW_SEG)
    parser.add_argument('--display', action='store_true', default=DISPLAY)
    parser.add_argument('--fps', type=int, default=FPS)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    parser.add_argument('--save_mode', type=str, default=SAVE_MODE)
    args = parser.parse_args()

    view_dataset(args.dataset_path,
                 show_bbox=args.show_bbox,
                 show_kpts=args.show_kpts,
                 show_seg=args.show_seg,
                 display=args.display,
                 fps=args.fps,
                 save_dir=args.save_dir,
                 save_mode=args.save_mode)


if __name__ == '__main__':
    main()
