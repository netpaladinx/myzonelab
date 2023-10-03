import os.path as osp
import json

import albumentations as A
import numpy as np
import cv2

from ...registry import DATA_TRANSFORMS
from ...utils import img_as_float, get_color_stats, apply_sigmoidal, apply_gamma, apply_saturation, img_as_ubyte
from ...consts import CADJUST_SIDE_TRUNCATION, CADJUST_IMAGE_STATS, CADJUST_CONTRAST_RANGE, CADJUST_BIAS_RANGE, CADJUST_GAMMA_RANGE, CADJUST_SATURATION_RANGE


@DATA_TRANSFORMS.register_class('albumentations')
class Albumentations:
    """ Apply coordinates-unvarying augmentation """

    def __init__(self,
                 apply_prob=1,
                 blur_prob=0.03,
                 random_gamma_prob=0.03,
                 hue_saturation_value_prob=0.03,
                 rgb_shift_prob=0.03,
                 motion_blur_prob=0.03,
                 median_blur_prob=0.03,
                 gaussian_blur_prob=0.03,
                 gauss_noise_prob=0.03,
                 glass_blur_prob=0.03,
                 clahe_prob=0.03,
                 invert_img_prob=0.03,
                 to_gray_prob=0.03,
                 to_sepia_prob=0.03,
                 image_compression_prob=0.03,
                 random_brightness_contrast_prob=0.03,
                 random_snow_prob=0.03,
                 random_rain_prob=0.03,
                 random_fog_prob=0.03,
                 random_sun_flare_prob=0.03,
                 random_shadow_prob=0.03,
                 random_tone_curve_prob=0.03,
                 channel_dropout_prob=0.03,
                 iso_noise_prob=0.03,
                 solarize_prob=0.03,
                 equalize_prob=0.03,
                 posterize_prob=0.03,
                 downscale_prob=0.03,
                 multiplicative_noise_prob=0.03,
                 fancy_pca_prob=0.03,
                 color_jitter_prob=0.03,
                 sharpen_prob=0.03,
                 emboss_prob=0.03,
                 superpixels_prob=0.03,
                 channel_shuffle_prob=0.03,
                 disable_all_except=None):

        self.apply_prob = apply_prob
        self.disable_all_except = disable_all_except

        self.transform = A.Compose([
            A.Blur(p=self._get_p('blur', blur_prob)),  # blur image using a random-sized kernel
            A.RandomGamma(p=self._get_p('random_gamma', random_gamma_prob)),  # apply random gamma (default: gamma_limit=(80, 120))
            A.HueSaturationValue(p=self._get_p('hue_saturation_value', hue_saturation_value_prob)),  # Randomly change hue, saturation and value of the input image.
            A.RGBShift(p=self._get_p('rgb_shift', rgb_shift_prob)),  # Randomly shift values for each channel of the input RGB image.
            A.MotionBlur(p=self._get_p('motion_blur', motion_blur_prob)),  # Apply motion blur to the input image using a random-sized kernel.
            A.MedianBlur(p=self._get_p('median_blur', median_blur_prob)),  # blur image using a median filter with a random aperture linear size
            A.GaussianBlur(p=self._get_p('gaussian_blur', gaussian_blur_prob)),  # Blur the input image using a Gaussian filter with a random kernel size.
            A.GlassBlur(p=self._get_p('glass_blur', glass_blur_prob)),  # Apply glass noise to the input image.
            A.CLAHE(p=self._get_p('clahe', clahe_prob)),  # apply Contrast Limited Adaptive Histogram Equalization
            A.InvertImg(p=self._get_p('invert_img', invert_img_prob)),  # Invert the input image by subtracting pixel values from 255.
            A.ToGray(p=self._get_p('to_gray', to_gray_prob)),  # convert RGB to grayscale (invert grayscale if mean pixel value >= 127)
            A.ToSepia(p=self._get_p('to_sepia', to_sepia_prob)),  # Applies sepia filter to the input RGB image
            A.ImageCompression(quality_lower=75, p=self._get_p('image_compression', image_compression_prob)),  # decrease Jpeg, WebP compression of an image
            A.RandomBrightnessContrast(p=self._get_p('random_brightness_contrast', random_brightness_contrast_prob)),  # randomly change brightness and contrast
            A.RandomSnow(p=self._get_p('random_snow', random_snow_prob)),  # Bleach out some pixel values simulating snow.
            A.RandomRain(p=self._get_p('random_rain', random_rain_prob)),  # Adds rain effects.
            A.RandomFog(p=self._get_p('random_fog', random_fog_prob)),  # Simulates fog for the image
            A.RandomSunFlare(p=self._get_p('random_sun_flare', random_sun_flare_prob)),  # Simulates Sun Flare for the image
            A.RandomShadow(p=self._get_p('random_shadow', random_shadow_prob)),  # Simulates shadows for the image
            A.RandomToneCurve(p=self._get_p('random_tone_curve', random_tone_curve_prob)),  # Randomly change relationship between bright and dark areas by manipulating its tone curve.
            A.ChannelDropout(p=self._get_p('channel_dropout', channel_dropout_prob)),  # Randomly Drop Channels in the input Image.
            A.ISONoise(p=self._get_p('iso_noise', iso_noise_prob)),  # Apply camera sensor noise.
            A.Solarize(p=self._get_p('solarize', solarize_prob)),  # Invert all pixel values above a threshold.
            A.Equalize(p=self._get_p('equalize', equalize_prob)),  # Equalize the image histogram.
            A.Posterize(p=self._get_p('posterize', posterize_prob)),  # Reduce the number of bits for each color channel.
            A.Downscale(p=self._get_p('downscale', downscale_prob)),  # Decreases image quality by downscaling and upscaling back.
            A.MultiplicativeNoise(p=self._get_p('multiplicative_noise', multiplicative_noise_prob)),  # Multiply image to random number or array of numbers.
            A.FancyPCA(p=self._get_p('fancy_pca', fancy_pca_prob)),  # Augment RGB image using FancyPCA from Krizhevsky's paper
            A.ColorJitter(p=self._get_p('color_jitter', color_jitter_prob)),  # Randomly changes the brightness, contrast, and saturation of an image
            A.Sharpen(p=self._get_p('sharpen', sharpen_prob)),  # Sharpen the input image and overlays the result with the original image.
            A.Emboss(p=self._get_p('emboss', emboss_prob)),  # Emboss the input image and overlays the result with the original image.
            A.Superpixels(p=self._get_p('superpixels', superpixels_prob)),  # Transform images partially/completely to their superpixel representation
            A.ChannelShuffle(p=self._get_p('channel_shuffle', channel_shuffle_prob)),  # randomly rearrange channels of the input RGB image.
            A.GaussNoise(p=self._get_p('gauss_noise', gauss_noise_prob))  # Apply gaussian noise to the input image.
        ])

    def _get_p(self, name, p):
        if self.disable_all_except and (name == self.disable_all_except or name in self.disable_all_except):
            return p
        return 0.

    def __call__(self, input_dict, dataset, step):
        if isinstance(input_dict, list):
            input_dict = [self._album_agument(in_dict) for in_dict in input_dict]
        else:
            input_dict = self._album_agument(input_dict)

        return input_dict

    def _album_agument(self, input_dict):
        if np.random.rand() < self.apply_prob:
            img = input_dict['img']

            if img.ndim == 4:
                for i in range(img.shape[0]):
                    ret = self.transform(image=img[i])
                    img[i] = np.ascontiguousarray(ret['image'])
            else:
                ret = self.transform(image=img)
                img = np.ascontiguousarray(ret['image'])

            input_dict['img'] = img
        return input_dict


@DATA_TRANSFORMS.register_class('agument_hsv')
class AgumentHSV:
    """ HSV color-space augmentation """

    def __init__(self, apply_prob=0.5, h_gain=0.02, s_gain=0.6, v_gain=0.4):
        self.apply_prob = apply_prob
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, input_dict, dataset, step):
        if isinstance(input_dict, list):
            input_dict = [self._agument_hsv(in_dict) for in_dict in input_dict]
        else:
            input_dict = self._agument_hsv(input_dict)

        return input_dict

    def _agument_hsv(self, input_dict):
        if np.random.rand() < self.apply_prob:
            img = input_dict['img']

            if img.ndim == 4:
                for i in range(img.shape[0]):
                    gains = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
                    img[i] = self._transform(img[i], gains)
            else:
                gains = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
                img = self._transform(img, gains)

            input_dict['img'] = img
        return input_dict

    @staticmethod
    def _transform(img, gains):
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype

        x = np.arange(0, 256, dtype=gains.dtype)
        lut_hue = ((x * gains[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * gains[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * gains[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return img


@DATA_TRANSFORMS.register_class('adjust_color')
class AdjustColor:
    def __init__(self, apply_prob=1, keep_adjusted_as_orig=False):
        self.apply_prob = apply_prob
        self.keep_adjusted_as_orig = keep_adjusted_as_orig

        params_file = osp.join(osp.dirname(osp.realpath(__file__)), '_params_adjustcolor.json')
        self.layers = json.load(open(params_file))

        self.contrast_range = CADJUST_CONTRAST_RANGE
        self.bias_range = CADJUST_BIAS_RANGE
        self.gamma_range = CADJUST_GAMMA_RANGE
        self.saturation_range = CADJUST_SATURATION_RANGE
        self.side_truncation = CADJUST_SIDE_TRUNCATION

    def __call__(self, input_dict, dataset, step):
        if isinstance(input_dict, list):
            input_dict = [self._adjust_color(in_dict) for in_dict in input_dict]
        else:
            input_dict = self._adjust_color(input_dict)

        return input_dict

    def _adjust_color(self, input_dict):
        if self.apply_prob == 1 or np.random.rand() < self.apply_prob:
            img = input_dict['img']

            if img.ndim == 4:
                for i in range(img.shape[0]):
                    contrast, bias, red, green, blue, saturation = self._get_op_params(img[i])
                    img[i] = self._adjust_image(img[i], contrast, bias, red, green, blue, saturation)
            else:
                contrast, bias, red, green, blue, saturation = self._get_op_params(img)
                img = self._adjust_image(img, contrast, bias, red, green, blue, saturation)

            input_dict['img'] = img
            if self.keep_adjusted_as_orig:
                input_dict['orig_img'] = img.copy()
        return input_dict

    def _get_op_params(self, img):
        assert img.dtype == np.uint8
        img = img_as_float(img, min=1)
        stats = get_color_stats(img, side_truncation=self.side_truncation)
        stats = np.array([stats[name] for name in CADJUST_IMAGE_STATS])

        out = stats
        for layer in self.layers:
            if layer['type'] == 'linear':
                weight, bias = np.array(layer['weight']), np.array(layer['bias'])
                out = out @ weight.T + bias
            elif layer['type'] == 'relu':
                out = np.clip(out, 0, None)
        out = 1 / (1 + np.exp(-out))

        contrast = out[0] * (self.contrast_range[1] - self.contrast_range[0]) + self.contrast_range[0]
        bias = out[1] * (self.bias_range[1] - self.bias_range[0]) + self.bias_range[0]
        red = out[2] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        green = out[3] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        blue = out[4] * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        saturation = out[5] * (self.saturation_range[1] - self.saturation_range[0]) + self.saturation_range[0]
        return contrast, bias, red, green, blue, saturation

    @staticmethod
    def _adjust_image(img, contrast, bias, red, green, blue, saturation):
        assert img.dtype == np.uint8
        img = img_as_float(img, min=1)
        img = apply_sigmoidal(img, contrast, bias)
        img = apply_gamma(img, (red, green, blue))
        img = apply_saturation(img, saturation)
        img = img_as_ubyte(img)
        return img
