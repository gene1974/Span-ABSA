# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings
from code.config import get_default_args
from code.factory import Factory

def test(model_time):
    warnings.filterwarnings("ignore")
    root = "../JDComment_seg"
    args = get_default_args(root=root)
    factory = Factory(args)
    factory.test(model_time)

if __name__ == '__main__':
    model_time = '20240407_1722'
    test(model_time)
