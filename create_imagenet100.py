import os
import numpy as np
import pandas as pd

from distutils.dir_util import copy_tree

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

class_ids = pd.read_csv('classes.csv')
class_ids = class_ids['Classes'].to_list()

train_folder = 'data/imagenet/train'
val_folder = 'data/imagenet/val'

train_destination = 'data/imagenet100/train'
val_destination = 'data/imagenet100/val'

for c_id in class_ids:
    train_folder_id = os.path.join(train_folder, c_id)
    val_folder_id = os.path.join(val_folder, c_id)
    
    train_destination_id = os.path.join(train_destination, c_id)
    val_destination_id = os.path.join(val_destination, c_id)

    copyanything(train_folder_id, train_destination_id)
    copyanything(val_folder_id, val_destination_id)


    
