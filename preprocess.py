import numpy as np
import tensorflow as tf
import os
import glob
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

def get_images_paths(directory_name, image_type='png'):
    """
    get the file name/path of all the files within a folder.
        e.g. glob.glob("/home/adam/*/*.txt").
    Use glob.escape to escape strings that are not meant to be patterns
        glob.glob(glob.escape(directory_name) + "/*.txt")
    :param directory_name: (str) the root directory name that contains all the images we want
    :param image: (str) either "jpg" or "png"
    :return: a list of queried files and directories
    """
    # concatnate strings
    end = "/*." + image_type

    return glob.glob(glob.escape(directory_name) + end)

def extract_classwise_instances(samples, output_dir, label_field, size_lower_limit, ext=".png"):
    print("Extracting object instances...")
    for sample in samples.iter_samples(progress=True):
        img = cv2.imread(sample.filepath)
        img_h,img_w,c = img.shape
        if img_h >= size_lower_limit and img_w >= size_lower_limit:
            for det in sample[label_field].detections:
                mask = det.mask
                [x,y,w,h] = det.bounding_box
                x = int(x * img_w)
                y = int(y * img_h)
                h, w = mask.shape
                mask_img = img[y:y+h, x:x+w, :]
                alpha = mask.astype(np.uint8)*255
                alpha = np.expand_dims(alpha, 2)
                mask_img = np.concatenate((mask_img, alpha), axis=2)
                label = det.label
                label_dir = os.path.join(output_dir, label)

                if not os.path.exists(label_dir):
                    os.mkdir(label_dir)
                output_filepath = os.path.join(label_dir, det.id+ext)
                cv2.imwrite(output_filepath, mask_img)