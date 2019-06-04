import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.patches as patches

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import mammo_baseline_pathology

print('hello world')
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "mammography/checkpoints")

# Dataset directory
DATASET_DIR = os.path.join(ROOT_DIR, "dataset/mammo")

# %load_ext autoreload
# %autoreload 2

# # Inference Configuration
config = mammo_baseline_pathology.MammoInferenceConfig()
# config.display()

DEVICE = 'cpu'
# DEVICE = "/gpu:0"  #
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax


if __name__ =='__main__':
    print('hello world')
    dataset = mammo_baseline_pathology.MammoDataset()
    dataset.load_mammo(DATASET_DIR, "mass_train", augmented=False, json_filename="mammo_all_ddsm_mass_train.json")
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference",
                                  model_dir=LOGS_DIR,
                                  config=config)

    # Run Detection

    image_id = random.choice(dataset.image_ids)
    # image_id = 343
    print("Image ID:", image_id)
    _25aug_1024reso_3x_3class = "mammo20180824T1833"
    n_epochs = "mask_rcnn_mammo_000" + str(5) + ".h5"
    weights_path = os.path.join(LOGS_DIR, _25aug_1024reso_3x_3class, n_epochs)

    # print("\n", i, ": Loading weights ", weights_path)
    # time_now = time.time()
    model.load_weights(weights_path, by_name=True)

    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                           dataset.image_reference(image_id)))
    print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

    # Run object detection
    results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

    # # Display results
    r = results[0]
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # # Compute AP over range 0.5 to 0.95 and print it
    mAP, AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                                     r['rois'], r['class_ids'], r['scores'], r['masks'],
                                     verbose=1)

    # print(AP)
    print(gt_class_id)
    print(r['class_ids'])

    visualize.display_differences(
        image,
        gt_bbox, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'],
        dataset.class_names, ax=get_ax(),
        show_box=False, show_mask=True,
        iou_threshold=0.5, score_threshold=0.9)

    # Path to a specific weights file
    # weights_path = "/path/to/mask_rcnn_nucleus.h5"
    path = "mammo20190517T1130"

    epochs_trained = 10
    limit = 361


    def compute_batch_ap(dataset, image_ids, verbose=1):
        APs = []
        for image_id in image_ids:
            # Load image
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config,
                                       image_id, use_mini_mask=False)
            # Run object detection
            results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
            # Compute AP over range 0.5 to 0.95
            r = results[0]
            ap = utils.compute_ap_range(
                gt_bbox, gt_class_id, gt_mask,
                r['rois'], r['class_ids'], r['scores'], r['masks'],
                verbose=0)
            APs.append(ap)
            if verbose:
                info = dataset.image_info[image_id]
                meta = modellib.parse_image_meta(image_meta[np.newaxis, ...])
                print("{:3} {}   AP: {:.2f}".format(
                    meta["image_id"][0], meta["original_image_shape"][0], ap))
        return APs


    for i in range(2, epochs_trained + 1):
        if i < 10:
            n_epochs = "mask_rcnn_mammo_000" + str(i) + ".h5"
            weights_path = os.path.join(LOGS_DIR, third, n_epochs)
            print("Loading weights ", weights_path)
            model.load_weights(weights_path, by_name=True)
            APs = compute_batch_ap(dataset, dataset.image_ids[:limit], verbose=0)
            print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))
        elif i >= 10:
            n_epochs = "mask_rcnn_mammo_00" + str(i) + ".h5"
            weights_path = os.path.join(LOGS_DIR, path, n_epochs)
            print("Loading weights ", weights_path)
            model.load_weights(weights_path, by_name=True)
            APs = compute_batch_ap(dataset, dataset.image_ids[:limit], verbose=0)
            print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))

    # Run on validation set

    