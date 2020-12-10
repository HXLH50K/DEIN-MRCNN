# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

#%%
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from skimage import exposure
import pydicom
from imgaug import augmenters as iaa
import imgaug as ia
#%%
# Root directory of the project
MRCNN_DIR = os.path.abspath("../")
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(MRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import os.path as osp
from mrcnn.model import log

# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.compat.v1.Session(config=config)
KTF.set_session(sess)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MRCNN_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGENET_MODEL_PATH = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")

# %%
import pandas as pd
df = pd.read_excel(io = osp.join(ROOT_DIR,"dataset","pathology.xls"), header = 0)
# %%
# ## Configurations
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "liver"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    # NUM_CLASSES = 5 + 1
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    PRE_NMS_LIMIT = 6000

    IMAGE_CHANNEL_COUNT = 1
    BACKBONE = "resnet101"
    # BACKBONE = "MIFnet"

    USE_MINI_MASK = False

    LEARNING_RATE = 1e-5

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
    }
    TRAIN_BN = False
config = ShapesConfig()
config.display()

# ## Notebook Preferences
# %%
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# %%
class DrugDataset(utils.Dataset):
    # label.png中，像素值为0的是背景，1为第一类目标，2维第二类……
    # 获得mask中目标的种类数量（不含背景）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_txt_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['txt_path'])as f:
            labels = []
            for line in f:
                labels.append(line.strip('\n'))
            del labels[0]
        return labels

    def convert(self, img, width, center):
        low = center-width/2
        high = center+width/2
        img[img<low] = low
        img[img>high] = high
        newimg = (img-low)/width
        newimg = (newimg*255).astype('uint8')
        return newimg

    def load_image(self, image_id):
        """Load the specified image 
            if png jpg, return a [H,W,3] Numpy array.
            if dcm, return a [H,W,3] and the 3 last channel 
                is same dcm with different window center/width.
        """
        # Load image
        path = self.image_info[image_id]['path']
        if path.endswith(".png" or ".jpg"):
            image = skimage.io.imread(path)
            # If grayscale. Convert to RGB for consistency.
            if image.ndim != 3:
                image = skimage.color.gray2rgb(image)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
        elif path.endswith(".dcm"):
            ds = pydicom.read_file(path)
            img0 = ds.pixel_array.copy().astype(np.int16)
            img0 = img0 * ds.RescaleSlope + ds.RescaleIntercept
            img0[img0 < -1024] = -1024
            # M99 = np.percentile(img0, 99)
            # M55 = np.percentile(img0, 55)
            # image = exposure.equalize_hist(img0, mask=(img0>M55) & (img0<M99))
            # image = np.stack([self.convert(img0.copy(), 1200, 50),
            #                   self.convert(img0.copy(),  600, 50),
            #                   self.convert(img0.copy(),  300, 50)],
            #                   axis=-1)
            image = img0[...,np.newaxis]
        return image
        

    # 重新写draw_mask
    def draw_mask(self, count, mask, image, image_id):
        a = np.reshape(image, (-1,))
        b = np.zeros((a.size, a.max()+1))
        b[np.arange(a.size),a] = 1
        c = np.reshape(b, (mask.shape[0], mask.shape[1], mask.shape[2]+1))
        mask = c[..., 1:]
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别（我的是yellow,white,back三类）
    #这里的class是类别顺序要按着自己网络输出的顺序来添加。
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, count, height, width, data_list):
        self.add_class("liver", 1, "liver_0")
        # self.add_class("liver", 2, "liver_1")
        # self.add_class("liver", 3, "liver_2")
        # self.add_class("liver", 4, "liver_3")
        # self.add_class("liver", 5, "liver_4")
        for i in range(count):
            # img_path = os.path.join(data_list[i], "img.png")
            json_dir_path = data_list[i]
            dcm_parent_path = osp.dirname(json_dir_path).replace("dataset", "dataset_dicom")
            dcm_right_name = osp.basename(json_dir_path).replace("_json", ".dcm").split("-")[2]
            assert osp.isdir(dcm_parent_path)
            dcm_list = os.listdir(dcm_parent_path)
            for x in dcm_list:
                if x.find(dcm_right_name) != -1:
                    dcm_name = x
                    break
            dcm_path = osp.join(dcm_parent_path, dcm_name)
            mask_path = os.path.join(data_list[i], "label.png")
            txt_path = os.path.join(data_list[i], "label_names.txt")
            self.add_image("liver", image_id=i, 
                            path=dcm_path,
                            width=width, height=height, 
                            mask_path=mask_path, txt_path=txt_path)

    # 重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        labels = []
        labels = self.from_txt_get_class(image_id)
        count = len(labels)

        patient_id = osp.basename(osp.dirname(osp.dirname(info["path"]))) # "aXX"
        patient_id_num = int(patient_id[1:]) # XX
        # img = Image.open(info['mask_path'])
        img = cv2.imread(info['mask_path'], cv2.IMREAD_GRAYSCALE)
        num_obj = self.get_obj_index(img)

        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)

        mask = self.draw_mask(count, mask, img, image_id)
        
        labels_from = []
        for i in range(len(labels)):
            label = ""
            if re.findall("liver|[12345]", labels[i]) != []:
                # label+="liver_"
                # label+=str(df.values[patient_id_num-1,1])
                label+="liver_0"
            labels_from.append(label)
        class_ids = np.array([self.class_names.index(s) for s in labels_from])
        # class_ids = np.array([self.class_names.index(s) for s in labels])
        return mask.astype(np.bool), class_ids.astype(np.int32)


# %%
# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# %%
"""The new data division method, each patient's every img will only appear in one set, divide train val test"""
data_folder = os.path.join(ROOT_DIR, "dataset")
dcm_folder = os.path.join(ROOT_DIR, "dataset_dicom")
patient_list = [osp.join(data_folder,name) for name in os.listdir(data_folder)
                    if osp.isdir(osp.join(data_folder, name))]

from sklearn.model_selection import train_test_split
patient_list_intrain, patient_list_test = train_test_split(patient_list, test_size=8, random_state=42)
patient_list_train, patient_list_val = train_test_split(patient_list_intrain, test_size=0.3, random_state=43)

data_list_train = []
for patient_folder in patient_list_train:
    for root, dirs, files in os.walk(patient_folder, topdown=True):
        for name in dirs:
            if name.endswith("_json"):
                data_list_train.append(osp.join(root, name))
data_list_val = []
for patient_folder in patient_list_val:
    for root, dirs, files in os.walk(patient_folder, topdown=True):
        for name in dirs:
            if name.endswith("_json"):
                data_list_val.append(osp.join(root, name))

count = len(patient_list)
count_train = len(data_list_train)
count_val = len(data_list_val)
# %%
# 修改为自己的网络输入大小
width = 512
height = 512
# %%
dataset_train = DrugDataset()
dataset_train.load_shapes(count_train, height, width, data_list_train)
dataset_train.prepare()

# Validation dataset
dataset_val = DrugDataset()
dataset_val.load_shapes(count_val, height, width, data_list_val)
dataset_val.prepare()
# %%
# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
#  %%
# Create model in training mode
print('create model')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "none"  # imagenet, coco, or last

if init_with == "imagenet":
    if not os.path.exists(IMAGENET_MODEL_PATH):
        utils.download_trained_weights(IMAGENET_MODEL_PATH)
    # model.load_weights(model.get_imagenet_weights(), by_name=True)
    model.load_weights(IMAGENET_MODEL_PATH, by_name=True)
elif init_with == "coco":
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)
elif init_with == "none":
    pass

# %%
# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.



# callbacks
import keras.callbacks as KC

# %%:
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
callbacks = [
    KC.TerminateOnNaN(),
    KC.CSVLogger('training.csv', separator=',', append=False)
]
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(iaa.Fliplr(0.5)),
    sometimes(iaa.Crop(percent=(0,0.2)))
])
# seq = iaa.SomeOf((1, None), [
#     iaa.Fliplr(1),
#     sometimes(iaa.Crop(percent=(0,0.2))),
#     iaa.Flipud(1),
#     iaa.Affine(rotate=(-45, 45)), 
#     iaa.Affine(rotate=(-90, 90)), 
#     iaa.Affine(scale=(0.5, 1.5)),
#     iaa.MedianBlur(),
#     iaa.Dropout(),
#     iaa.AdditiveGaussianNoise(),
#     iaa.CoarseSaltAndPepper(p=0.2),
#     iaa.GaussianBlur(),
#     iaa.GammaContrast(),
#     iaa.PerspectiveTransform(scale=0.1),
#     iaa.Pad()
# ], random_order=True)
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers="all",
            custom_callbacks=callbacks,
            augmentation=None)

# %%
# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_liver.h5")
model.keras_model.save_weights(model_path)

# %%
import requests
requests.get("https://sc.ftqq.com/SCU105852T7a5cbc7a42c7e841cc46a6c629911cde5f0eff52ded6f.send?text=train_completed" )
# ## Detection