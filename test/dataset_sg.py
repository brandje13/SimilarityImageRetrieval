#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import core.transforms as transforms
import torch.utils.data
from test.dataset import DataSet

import json

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


class DataSet_SG(DataSet):
    """SG dataset."""

    def __init__(self, data_path, dataset, fn, split, scale_list):
        super().__init__(data_path, dataset, fn, split)
        self._scale_list = scale_list

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        im = im.transpose([2, 0, 1])
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, _MEAN, _SD)
        return im

    def __getitem__(self, index):
        # Load the image
        im = self._load_img(index)
        im_list = []

        for scale in self._scale_list:
            if scale == 1.0:
                im_np = im.astype(np.float32, copy=False)
                im_list.append(im_np)
            elif scale < 1.0:
                im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                im_np = im_resize.astype(np.float32, copy=False)
                im_list.append(im_np)
            elif scale > 1.0:
                im_resize = cv2.resize(im, dsize=(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                im_np = im_resize.astype(np.float32, copy=False)
                im_list.append(im_np)
            else:
                assert()

        for idx in range(len(im_list)):
            im_list[idx] = self._prepare_im(im_list[idx])
        return im_list
