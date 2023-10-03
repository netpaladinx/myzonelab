#!/usr/bin/env python
import os.path as osp
import argparse

from myzonecv.apis import test, get_config_file, utils as U

####### DEBUG #########
DATA_TEST = None  # 'test_stable', None
CONFIG_NAME = 'pose/pose_hrnet_w32_coco_256x192'
WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test_kpts"}__' + CONFIG_NAME
EVAL_HARD_LEVEL = 0  # 3
MODEL_TAG = 'pretrained'
DATA_TAG = 'debug2'
SUFFIX = f'{MODEL_TAG}__{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
SUMMARY_NAME = 'summary__' + SUFFIX
PLOT_NAME = 'plot__' + SUFFIX
ANALYSIS_NAME = 'analysis__' + SUFFIX
VISUALIZE_NAME = 'visualize__' + SUFFIX
ANN_FILE = f'./workspace/data_zoo/trainval/person_keypoints_{DATA_TAG}/annotations.json'
IMG_DIR = f'./workspace/data_zoo/trainval/person_keypoints_{DATA_TAG}/images'
CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
CHECKPOINT_TYPE = 'pretrained'
BATCH_SIZE = 1
NUM_WORKERS = 1
SELECT_POLICY = None

###############################
#####    ACCURACY TEST    #####
###############################


def get_ann_file(data_tag):
    if data_tag == 'validation':
        return f'./workspace/data_zoo/trainval/Validation_FightFlow_V1/annotations.json'
    elif data_tag in ('0331',):
        return f'./workspace/data_zoo/trainval/MyZoneUFC-Pose/2022{data_tag}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
    else:
        return f'./workspace/data_zoo/trainval/person_keypoints_{data_tag}/annotations.json'


def get_img_dir(data_tag):
    if data_tag == 'validation':
        return f'./workspace/data_zoo/trainval/Validation_FightFlow_V1/images'
    elif data_tag in ('0331',):
        return f'./workspace/data_zoo/trainval/MyZoneUFC-Pose/2022{data_tag}_FightFlow_RetrainingData_Error_0-350_Revised/images'
    else:
        return f'./workspace/data_zoo/trainval/person_keypoints_{data_tag}/images'


def get_ckpt_path(model_tag, model_name='hrnet_w32_coco_256x192'):
    if model_tag == 'pretrained':
        return './workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
    else:
        return f'./workspace/model_zoo/pose_models/{model_name}-{model_tag}.pth'


def get_ckpt_type(model_tag):
    if model_tag == 'pretrained':
        return 'pretrained'
    else:
        return 'checkpoint'


def get_suffix(data_tag, model_tag, eval_hard_level=0):
    if data_tag == 'validation':
        return f'{model_tag}__{data_tag}__hard_{eval_hard_level}' if eval_hard_level > 0 else f'{model_tag}__{data_tag}'
    elif data_tag in ('0331',):
        return f'{model_tag}__{data_tag}__hard_{eval_hard_level}' if eval_hard_level > 0 else f'{model_tag}__retraindata_{data_tag}'
    else:
        return f'{model_tag}__hard_{eval_hard_level}' if eval_hard_level > 0 else f'{model_tag}'


def get_work_dir(data_test, config_name):
    if not data_test:
        data_test = 'test'
    return f'./workspace/experiments/{data_test}__{config_name}'


def get_test_params(data_test, config_name, data_tag, model_tag, model_name='hrnet_w32_coco_256x192', eval_hard_level=0):
    work_dir = get_work_dir(data_test, config_name)
    suffix = get_suffix(data_tag, model_tag, eval_hard_level)
    summary_name = 'summary__' + suffix
    plot_name = 'plot__' + suffix
    analysis_name = 'analysis__' + suffix
    visualize_name = 'visualize__' + suffix
    ann_file = get_ann_file(data_tag)
    img_dir = get_img_dir(data_tag)
    checkpoint_path = get_ckpt_path(model_tag, model_name)
    checkpoint_type = get_ckpt_type(model_tag)
    return work_dir, summary_name, plot_name, analysis_name, visualize_name, ann_file, img_dir, checkpoint_path, checkpoint_type


##### hrnet_w32_coco_256x192-c78dce93_20200708.pth #####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_w32_coco_256x192'
# MODEL_TAG = 'pretrained'

##### hrnet_w32_coco_256x192-retrained_20220324T135345.pth #####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_w32_coco_256x192'
# MODEL_TAG = 'retrained_20220324T135345'

##### hrnet_w32_coco_256x192-retrained_20220404T143948.pth #####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_w32_coco_256x192'
# MODEL_TAG = 'retrained_20220404T143948'

##### hrnet_w32_coco_256x192-retrained_20220407T132620.pth ####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_w32_coco_256x192'
# MODEL_TAG = 'retrained_20220407T132620'

##### hrnet_w32_coco_256x192-retrained_20220504T084953.pth #####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_w32_coco_256x192'
# MODEL_TAG = 'retrained_20220504T084953'

##### hrnet_w32_coco_256x192_v2-experimental_20220605T060104.pth #####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_w32_coco_256x192_v2'
# MODEL_TAG = 'experimental_20220605T060104'

##### hrnet_stable_w32_coco_256x192-experimental_stable(v2,mul2,bs32,wei0.5,noflip,epoch50)_20220611T124145.pth ####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# DATA_TAG = 'validation'  # '0331'
# MODEL_NAME = 'hrnet_stable_w32_coco_256x192'
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5,noflip,epoch50)_20220611T124145'

##### hrnet_w32_coco_256x192-c78dce93_20200708.pth #####
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# CONFIG_NAME = 'pose/pose_hrnet_w32_coco_256x192'
# DATA_TAG = '0331'  # 'validation'  #
# MODEL_NAME = 'hrnet_w32_coco_256x192'
# MODEL_TAG = 'retrained_20220709T151140B'

# DATA_TEST = None
# EVAL_HARD_LEVEL = 0
# (WORK_DIR, SUMMARY_NAME, PLOT_NAME, ANALYSIS_NAME, VISUALIZE_NAME,
#  ANN_FILE, IMG_DIR, CHECKPOINT_PATH, CHECKPOINT_TYPE) = get_test_params(DATA_TEST, CONFIG_NAME, DATA_TAG, MODEL_TAG, MODEL_NAME, EVAL_HARD_LEVEL)
# BATCH_SIZE, NUM_WORKERS = 8, 2
# SELECT_POLICY = None

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul3,bs16,wei1)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220605T095023.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul2,bs32,wei0.5)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220606T051824.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.5)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}_20220606T181532.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.1)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}_20220607T193336.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.1)_laplace(simple)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}_20220609T083216.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.1)_laplace(shr0.5,wei0.8)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}_20220610T071203.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul2,bs32,wei0.5)_distaware(gamma0.5)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220607T173214.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul2,bs32,wei0.5)_distaware(gamma0.1)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220609T040937.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220610T081417.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5,noflip)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220611T035548.pth'
# CHECKPOINT_TYPE = 'checkpoint'

################################
# DATA_TEST = 'test_stable'  # None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test"}__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5,noflip,epoch50)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220611T124145.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

################################
# DATA_TEST = None  # None or 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test"}__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,epoch50)'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}_20220620T191312.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
#NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.1,noflip,epoch50)_20220623T183208.pth #####
# DATA_TEST = None
# MODEL_TAG = ('hrnet_stable_w32_coco_256x192', 'experimental_stable(lossv2,mul2,bs32,wei0.1,noflip,epoch50)', '20220623T183208')
# DATA_TAG = '20220331'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test"}__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# SUFFIX = f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/{MODEL_TAG[0]}-{MODEL_TAG[1]}_{MODEL_TAG[2]}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 16
# NUM_WORKERS = 4

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,epoch50)_20220622T091125.pth #####
# DATA_TEST = None
# MODEL_TAG = ('hrnet_stable_w32_coco_256x192', 'experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,epoch50)', '20220622T091125')
# DATA_TAG = '20220331'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test"}__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# SUFFIX = f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/{MODEL_TAG[0]}-{MODEL_TAG[1]}_{MODEL_TAG[2]}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 16
# NUM_WORKERS = 4

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.5,epoch50)_20220623T183305.pth #####
# DATA_TEST = None
# MODEL_TAG = ('hrnet_stable_w32_coco_256x192', 'experimental_stable(lossv2,mul2,bs32,wei0.5,epoch50)', '20220623T183305')
# DATA_TAG = '20220331'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test"}__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# SUFFIX = f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/{MODEL_TAG[0]}-{MODEL_TAG[1]}_{MODEL_TAG[2]}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 16
# NUM_WORKERS = 4

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,vthr0.8,epoch50)_20220622T090716.pth #####
# DATA_TEST = None
# MODEL_TAG = ('hrnet_stable_w32_coco_256x192', 'experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,vthr0.8,epoch50)', '20220622T090716')
# DATA_TAG = '20220331'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = f'./workspace/experiments/{DATA_TEST or "test"}__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# SUFFIX = f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG[1]}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Original/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/{MODEL_TAG[0]}-{MODEL_TAG[1]}_{MODEL_TAG[2]}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 16
# NUM_WORKERS = 4

#################################
#####    STABLENESS TEST    #####
#################################

##### hrnet_w32_coco_256x192-c78dce93_20200708.pth #####
# DATA_TEST = 'test_stable'  # 'test_stable', None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'pretrained'
# DATA_TAG = '0331'
# SUFFIX = f'{MODEL_TAG}__retraindata_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__retraindata_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# ANN_FILE = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/2022{DATA_TAG}_FightFlow_RetrainingData_Error_0-350/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
# CHECKPOINT_TYPE = 'pretrained'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_w32_coco_256x192-retrained_20220504T084953.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v1'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'retrained_20220504T084953'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_w32_coco_256x192_v2-experimental_20220605T060104.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_20220605T060104'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(mul3,bs16,wei1)_20220605T095023.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul3,bs16,wei1)_20220605T095023'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(mul2,bs32,wei0.5)_20220606T051824.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul2,bs32,wei0.5)_20220606T051824'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1


##### hrnet_w32_coco_256x192_v2-experimental_distaware(gamma0.5)_20220606T181532.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.5)_20220606T181532'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_w32_coco_256x192_v2-experimental_distaware(gamma0.1)_20220607T193336.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.1)_20220607T193336'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_w32_coco_256x192_v2-experimental_distaware(gamma0.1)_laplace(simple)_20220609T083216.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.1)_laplace(simple)_20220609T083216'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_w32_coco_256x192_v2-experimental_distaware(gamma0.1)_laplace(shr0.5,wei0.8)_20220610T071203.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_distaware(gamma0.1)_laplace(shr0.5,wei0.8)_20220610T071203'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(mul2,bs32,wei0.5)_distaware(gamma0.5)_20220607T173214.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul2,bs32,wei0.5)_distaware(gamma0.5)_20220607T173214'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(mul2,bs32,wei0.5)_distaware(gamma0.1)_20220609T040937.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(mul2,bs32,wei0.5)_distaware(gamma0.1)_20220609T040937'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(v2,mul2,bs32,wei0.5)_20220610T081417.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5)_20220610T081417'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(v2,mul2,bs32,wei0.5,noflip)_20220611T035548.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5,noflip)_20220611T035548'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(v2,mul2,bs32,wei0.5,noflip,epoch50)_20220611T124145.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(v2,mul2,bs32,wei0.5,noflip,epoch50)_20220611T124145'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,epoch50)_20220622T091125.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,epoch50)_20220622T091125'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.5,epoch50)_20220623T183305.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(lossv2,mul2,bs32,wei0.5,epoch50)_20220623T183305'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.1,noflip,epoch50)_20220623T183208.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(lossv2,mul2,bs32,wei0.1,noflip,epoch50)_20220623T183208'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

##### hrnet_stable_w32_coco_256x192-experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,vthr0.8,epoch50)_20220622T090716.pth #####
# DATA_TEST = 'test_stable'
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/experiments/test__' + CONFIG_NAME
# EVAL_HARD_LEVEL = 0  # 3
# MODEL_TAG = 'experimental_stable(lossv2,mul2,bs32,wei0.5,noflip,vthr0.8,epoch50)_20220622T090716'
# DATA_TAG = '20220331'
# SUFFIX = f'{MODEL_TAG}__data_{DATA_TAG}__hard_{EVAL_HARD_LEVEL}' if EVAL_HARD_LEVEL > 0 else f'{MODEL_TAG}__data_{DATA_TAG}'
# SUMMARY_NAME = 'summary__' + SUFFIX
# PLOT_NAME = 'plot__' + SUFFIX
# ANALYSIS_NAME = 'analysis__' + SUFFIX
# VISUALIZE_NAME = None
# ANN_FILE = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/annotations.json'
# IMG_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow_RetrainingData_Error_0-350_Revised/images'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_stable_w32_coco_256x192-{MODEL_TAG}.pth'
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 1
# NUM_WORKERS = 1

####################################
#####    VISUALIZATION TEST    #####
####################################
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/tasks/test-visualize_pose_datasets'
# DATA_DIR = './workspace/data_zoo/trainval/20220507_FightFlow_RetrainingData_Error_0-350_Revised'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_20220317_FightFlow #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_20220317'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_20220331_FightFlow #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_20220331'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_20220426_FightFlow #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_20220426'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_20220507_FightFlow #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_20220507'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_20220514_FightFlow #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_20220514'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_20220521_FightFlow #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_20220521'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_FightFlow'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### Validation_FightFlow_V1 #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized/test_validation'
# DATA_TAG = 'Validation_FightFlow_V1'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}'
# MODEL_TAG = "experimental_20220605T060104"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'visualize_model-{MODEL_TAG}_{DATA_TAG}'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192_v2-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### 20220317_FightFlow_RetrainingData_Error_0-350_Revised #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized'
# DATA_TAG = '20220317_FightFlow'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_RetrainingData_Error_0-350_Revised'
# MODEL_TAG = "retrained_20220504T143019"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'cleandata_model-{MODEL_TAG}_data-{DATA_TAG}'
# SELECT_POLICY = 'worst_select'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### 20220331_FightFlow_RetrainingData_Error_0-350_Revised #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized'
# DATA_TAG = '20220331_FightFlow'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_RetrainingData_Error_0-350_Revised'
# MODEL_TAG = "retrained_20220504T143019"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'cleandata_model-{MODEL_TAG}_data-{DATA_TAG}'
# SELECT_POLICY = 'worst_select'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### 20220426_FightFlow_RetrainingData_Error_0-350_Revised #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized'
# DATA_TAG = '20220426_FightFlow'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_RetrainingData_Error_0-350_Revised'
# MODEL_TAG = "retrained_20220504T143019"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'cleandata_model-{MODEL_TAG}_data-{DATA_TAG}'
# SELECT_POLICY = 'worst_select'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### 20220507_FightFlow_RetrainingData_Error_0-350_Revised #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized'
# DATA_TAG = '20220507_FightFlow'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_RetrainingData_Error_0-350_Revised'
# MODEL_TAG = "retrained_20220504T143019"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'cleandata_model-{MODEL_TAG}_data-{DATA_TAG}'
# SELECT_POLICY = 'worst_select'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2

##### 20220514_FightFlow_RetrainingData_Error_0-350_Revised #####
# DATA_TEST = None
# CONFIG_NAME = 'pose_hrnet_w32_coco_256x192_v2'
# WORK_DIR = './workspace/data_zoo/visualized'
# DATA_TAG = '20220514_FightFlow'
# DATA_DIR = f'./workspace/data_zoo/trainval/{DATA_TAG}_RetrainingData_Error_0-350_Revised'
# MODEL_TAG = "retrained_20220504T143019"
# SUMMARY_NAME = None
# PLOT_NAME = None
# ANALYSIS_NAME = None
# VISUALIZE_NAME = f'cleandata_model-{MODEL_TAG}_data-{DATA_TAG}'
# SELECT_POLICY = 'worst_select'
# IMG_DIR = f'{DATA_DIR}/images'
# ANN_FILE = f'{DATA_DIR}/annotations.json'
# CHECKPOINT_PATH = f'./workspace/model_zoo/pose_models/hrnet_w32_coco_256x192-{MODEL_TAG}.pth'
# EVAL_HARD_LEVEL = 0
# CHECKPOINT_TYPE = 'checkpoint'
# BATCH_SIZE = 8
# NUM_WORKERS = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-test', type=str, default=DATA_TEST)
    parser.add_argument('--config', type=str, default=CONFIG_NAME)
    parser.add_argument('--work-dir', type=str, default=WORK_DIR)
    parser.add_argument('--summary-name', type=str, default=SUMMARY_NAME)
    parser.add_argument('--plot-name', type=str, default=PLOT_NAME)
    parser.add_argument('--analysis-name', type=str, default=ANALYSIS_NAME)
    parser.add_argument('--visualize-name', type=str, default=VISUALIZE_NAME)
    parser.add_argument('--select-policy', type=str, default=SELECT_POLICY)
    parser.add_argument('--eval-hard-level', type=int, default=EVAL_HARD_LEVEL)
    parser.add_argument('--ann-file', type=str, default=ANN_FILE)
    parser.add_argument('--img-dir', type=str, default=IMG_DIR)
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    args = parser.parse_args()

    config = get_config_file(args.config)

    summary_path = None
    plot_dir = None
    analysis_dir = None
    visualize_dir = None
    if args.work_dir:
        if args.summary_name:
            summary_path = osp.join(args.work_dir, f'{args.summary_name}.json')
        if args.plot_name:
            plot_dir = osp.join(args.work_dir, f'{args.plot_name}')
        if args.analysis_name:
            analysis_dir = osp.join(args.work_dir, f'{args.analysis_name}')
        if args.visualize_name:
            visualize_dir = osp.join(args.work_dir, f'{args.visualize_name}')

    ann_file = U.parse_path(args.ann_file, is_file=True)
    img_dir = U.parse_path(args.img_dir, is_dir=True)

    assert osp.isfile(args.checkpoint), f'Invalid checkpoint: {args.checkpoint}'
    kwargs = {
        'data_test': args.data_test,
        'work_dir': args.work_dir,
        'summary_path': summary_path,
        'plot_dir': plot_dir,
        'analysis_dir': analysis_dir,
        'visualize_dir': visualize_dir,
        'select_policy': args.select_policy,
        'eval_hard_level': args.eval_hard_level,
        'data.test.data_source': {'ann_file': ann_file, 'img_dir': img_dir},
        'data.test.data_loader': {'batch_size': BATCH_SIZE, 'num_workers': NUM_WORKERS},
        'model.init_cfg': {'type': CHECKPOINT_TYPE, 'path': args.checkpoint}
    }

    test(config, **kwargs)


if __name__ == '__main__':
    main()
