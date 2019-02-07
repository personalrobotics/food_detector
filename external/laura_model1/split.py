#!/usr/bin/env python3

import os
import glob
import random
import shutil


def split_images(flist, train_dir, validate_dir, category):
    for idx, sfname in enumerate(flist):
        if idx < len(flist) * 0.9:
            target_dir = os.path.join(train_dir, category)
        else:
            target_dir = os.path.join(validate_dir, category)
        tfname = os.path.join(target_dir, os.path.basename(sfname))
        shutil.copy(sfname, tfname)
        print('copy: {} -> {}'.format(sfname, tfname))


def run_split():
    print('split_images')

    train_dir = './data/train'
    validate_dir = './data/validate'

    os.makedirs(os.path.join(train_dir, 'success'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'failure'), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, 'success'), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, 'failure'), exist_ok=True)

    src_base_dir = os.path.expanduser('~/Images')
    success_list = glob.glob(os.path.join(src_base_dir, 'success/*.png'))
    failure_list = glob.glob(os.path.join(src_base_dir, 'failure/*.png'))

    random.shuffle(success_list)
    random.shuffle(failure_list)

    split_images(success_list, train_dir, validate_dir, 'success')
    split_images(failure_list, train_dir, validate_dir, 'failure')


if __name__ == '__main__':
    run_split()
