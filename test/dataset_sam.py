#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import cv2
from test.dataset import DataSet


class DataSet_SAM(DataSet):
    """SAM dataset."""

    def __init__(self, data_path, dataset, fn, split):
        super().__init__(data_path, dataset, fn, split)

    def __getitem__(self, index):
        # Load the image
        im = self._load_img(index)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

