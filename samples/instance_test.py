
# %%
import os
import os.path as osp
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
MRCNN_DIR = os.path.abspath("../")
ROOT_DIR = osp.abspath("../../")
sys.path.append(MRCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import cv2
from mrcnn.model import log
import pydicom
from keras.utils import to_categorical
matplotlib.rcParams['figure.figsize'] = [10, 10]
# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.compat.v1.ConfigProto()  
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
KTF.set_session(sess)
# %%
# 修改为自己的识别类别
# class_names = ['BG', 'liver_0','liver_1','liver_2','liver_3','liver_4']
class_names = ['BG', 'liver_0']
width=512
height=512

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
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    PRE_NMS_LIMIT = 6000

    IMAGE_CHANNEL_COUNT = 1
    BACKBONE = "resnet101"

    USE_MINI_MASK = False

    LEARNING_RATE = 1e-5

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
    }

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# %%
config = InferenceConfig()
config.display()

# %%
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MRCNN_DIR, "logs")

# Directory of images to run detection on
data_folder = os.path.join(ROOT_DIR, "dataset")
dcm_folder = os.path.join(ROOT_DIR, "dataset_dicom")
patient_list = [osp.join(data_folder,name) for name in os.listdir(data_folder)
                    if osp.isdir(osp.join(data_folder, name))]

# from sklearn.model_selection import train_test_split
# patient_list_intrain, patient_list_test = train_test_split(patient_list, test_size=8, random_state=42)
# patient_list_train, patient_list_val = train_test_split(patient_list_intrain, test_size=0.3, random_state=42)

patient_id_test = ['a43', 'a86', 'a28', 'a48',
                   'a23', 'a73', 'a46', 'a92']
patient_list_test = []
for x in patient_id_test:
    patient_list_test.append(osp.join(data_folder, x))

data_list_test = []
# for patient_folder in patient_list_test:
for patient_folder in patient_list_test:
    for root, dirs, files in os.walk(patient_folder, topdown=True):
        for name in dirs:
            if name.endswith("_json"):
                data_list_test.append(osp.join(root, name))
# %%
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", config=config , model_dir=MODEL_DIR)

# %%
# Local path to trained weights file
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_liver.h5")
model.load_weights(MODEL_PATH, by_name=True)
# %%
# def load_image(path):
#     if path.endswith(".png" or ".jpg"):
#         image = skimage.io.imread(path)
#     elif path.endswith(".dcm"):
#         ds = pydicom.read_file(path)
#         image = ds.pixel_array
#     # If grayscale. Convert to RGB for consistency.
#     if image.ndim != 3:
#         image = skimage.color.gray2rgb(image)
#     # If has an alpha channel, remove it for consistency
#     if image.shape[-1] == 4:
#         image = image[..., :3]
#     if ds:
#         return ds, image
#     else:
#         return image
def convert(img, width, center):
        low = center-width/2
        high = center+width/2
        img[img<low] = low
        img[img>high] = high
        newimg = (img-low)/width
        newimg = (newimg*255).astype('uint8')
        return newimg

def load_image(path):
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
        img0[img0 == -2000] = 0
        img0[img0 == -2048] = 0
        img0 = img0 * ds.RescaleSlope + ds.RescaleIntercept
        # image = np.stack([convert(img0.copy(), 1200, 50),
        #                   convert(img0.copy(),  600, 50),
        #                   convert(img0.copy(),  300, 50)],
        #                     axis=-1)
        image = img0[...,np.newaxis]
    if ds:
        return ds, image
    else:
        return image

def find_dcm(data_path):
    dcm_parent_path = osp.dirname(data_path).replace("dataset", "dataset_dicom")
    dcm_right_name = osp.basename(data_path).replace("_json", ".dcm").split("-")[2]
    assert osp.isdir(dcm_parent_path)
    dcm_list_temp = os.listdir(dcm_parent_path)
    for x in dcm_list_temp:
        if x.find(dcm_right_name) != -1:
            dcm_name = x
            break
    dcm_path = osp.join(dcm_parent_path, dcm_name)
    return dcm_path

def mask_ascend_dim(masks, class_ids):
    dim = len(class_names)
    shape = (width, height, dim)
    new_mask = np.zeros(shape, dtype = bool)
    if masks is not None:
        for i in range(len(class_ids)):
            new_mask[:,:,class_ids[i]] += masks[:,:,i]
    return new_mask

def load_mask(data_path):
    mask_path = os.path.join(data_path, "label.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    def mask_ascend_dim(mask):
        dim = np.max(mask)
        shape = (width, height, dim)
        new_mask = np.zeros(shape, dtype = int)
        for k in range(1, dim+1):
            for i in range(height):
                for j in range(width):
                    if mask[i][j]==k:
                        new_mask[i][j][k-1]=1
        return new_mask.astype(np.bool)
    mask = mask_ascend_dim(mask)
    return mask

def caculate_dice(y_true, y_pred, threshold = 0.5, smooth = 0.000001):
    y_true_flatten = y_true.flatten().astype(np.bool)
    y_pred_flatten = y_pred.flatten().astype(np.bool)
    return (2. * np.sum(y_true_flatten * y_pred_flatten)) / (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + smooth)
    
# def caculate_F1(GS, PS):
#     n = len(class_names)
#     LS = list(range(n))
#     TP = FP = FN = 0
#     for i in LS:
#         if i in PS and i in GS:
#             TP+=1
#         elif i in PS and i not in GS:
#             FP+=1
#         elif i not in PS and i in GS:
#             FN+=1
#     P = TP / (TP+FP)
#     R = TP / (TP+FN)
#     return P, R

def DICE(masks_true, masks_pred):
    """
    Get the DICE/IOU between each predicted mask and each true mask.

    Inputs:
        masks_true : array-like
            A 3D array of shape (n_true_masks, image_height, image_width)
        masks_pred : array-like
            A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns:
        array-like
            A 2D array of shape (n_true_masks, n_predicted_masks), where
            the element at position (i, j) denotes the dice between the `i`th true
            mask and the `j`th predicted mask.
    """
    if masks_true.shape[1:] != masks_pred.shape[1:]:
        raise ValueError('Predicted masks have wrong shape!')
    n_true_masks, height, width = masks_true.shape
    n_pred_masks = masks_pred.shape[0]
    m_true = masks_true.copy().reshape(n_true_masks, height * width).T
    m_pred = masks_pred.copy().reshape(n_pred_masks, height * width)
    numerator = np.dot(m_pred, m_true)
    denominator = m_pred.sum(1).reshape(-1, 1) + m_true.sum(0).reshape(1, -1)
    # return numerator / (denominator - numerator) #IOU
    return 2*numerator / denominator #dice

def evaluate_image(masks_true, masks_pred, thresholds = 0.5):
    """
    Get the average precision for the true and predicted masks of a single image,
    averaged over a set of thresholds

    Inputs:
        masks_true : array-like
            A 3D array of shape (n_true_masks, image_height, image_width)
        masks_pred : array-like
            A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns:
        float
            The mean average precision of intersection over union between
            all pairs of true and predicted region masks.

    """
    int_o_un = DICE(masks_true, masks_pred)
    benched = int_o_un > thresholds
    tp = benched.sum(-1).sum(-1)  # noqa
    fp = (benched.sum(1) == 0).sum(0)
    fn = (benched.sum(0) == 0).sum(0)
    return np.mean(tp / (tp + fp + fn))

def pixel2area(ds):
    return ds.PixelSpacing[0] * ds.PixelSpacing[1]

savedir = "./img"
def saveimg(mask_pred, mask_true, ids):
    g = np.logical_and(mask_pred, mask_true)
    b = np.logical_and(np.logical_not(mask_pred), mask_true)
    r = np.logical_and(mask_pred, np.logical_not(mask_true))
    mask = np.zeros((height, width, 3))
    mask[..., 0] = r[..., 0]
    mask[..., 1] = g[..., 0]
    mask[..., 2] = b[..., 0]
    apl = np.logical_or(mask_pred, mask_true)
    plt.axis('off')
    img1 = plt.imshow(img, cmap='gray', alpha=1)
    img1 = plt.imshow(mask*apl, cmap='gray',alpha=0.3)
    plt.savefig(osp.join(savedir, ids+"_mask.png"), figsize=(10, 10))
    img2 = plt.imshow(img,cmap='gray')
    plt.savefig(osp.join(savedir, ids+"_ori.png"), figsize=(10, 10))
# %%
# data_list_test = np.random.choice(data_list_test, 1)
n = len(data_list_test)
test_res = [] #[(id,dice,P,R),...]
err_list = []
gt_label = []
pred_label = []
# %%
import pandas as pd
df = pd.read_excel(io = osp.join(ROOT_DIR,"dataset","pathology.xls"), header = 0)
# %%
# a = []
for i, data_path in enumerate(data_list_test):
    dcm_path = find_dcm(data_path)
    patient_id = osp.basename(osp.dirname(osp.dirname(data_path)))
    patient_id_num = int(patient_id[1:])
    stage_id = osp.basename(osp.dirname(data_path))
    img_id = osp.basename(data_path).split("-")[-1].split('_')[0]
    main_id = patient_id + '_' + stage_id +'_' + img_id
    ds, img = load_image(dcm_path)

    # Run detection
    results = model.detect([img], verbose=0)

    r = results[0]
    try:
        # class_name_true = "liver_"+str(df.values[patient_id_num-1,1])
        class_name_true = "liver_0"
        class_ids_true = np.array([class_names.index(class_name_true)])
        class_ids_pred = r['class_ids']
        gt_label.append(to_categorical(class_ids_true, config.NUM_CLASSES).sum(0).astype(np.bool))
        pred_label.append(to_categorical(class_ids_pred, config.NUM_CLASSES).sum(0).astype(np.bool))

        mask_true = load_mask(data_path)
        mask_pred = r['masks']

        #caculate the max dice in all pred instance
        mask_true_ = mask_true.transpose(2, 0, 1).astype(np.int32)
        mask_pred_ = mask_pred.transpose(2, 0, 1).astype(np.int32)
        dice_ = DICE(mask_true_, mask_pred_)
        dice_instance = np.max(dice_) if dice_.size > 0 else 0
        
        #caculate dice of all pred instance union
        mask_true_ = mask_ascend_dim(mask_true, class_ids_true).sum(-1).astype(np.bool)
        mask_pred_ = mask_ascend_dim(mask_pred, class_ids_pred).sum(-1).astype(np.bool)
        dice_union = caculate_dice(mask_true_, mask_pred_)

        print("{}: {}/{}, num_ins: {}, dice_ins: {}, dice_union: {}"\
            .format(main_id, i+1, n, mask_pred.shape[-1], dice_instance, dice_union))
        test_res.append((main_id, dice_instance, dice_union))
    except Exception as e:
        print("error: {}, {}".format(main_id, e))
        err_list.append((main_id, e))
    saveimg(mask_pred, mask_true, main_id)
    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
    #                             figsize=(8,8), show_bbox=True)
# %%
# gt_label = np.array(gt_label)[:,1:]
# pred_label = np.array(pred_label)[:,1:]
# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
# class_f1 = f1_score(gt_label, pred_label, average='micro')
# print("Classification report: \n", (classification_report(gt_label, pred_label)))
# print("F1 micro averaging:",(class_f1))
# %%
test_res.sort(key = lambda x:x[1])
ids, dices, dices1 = zip(*test_res)
dice_max = np.max(dices)
dice_min = np.min(dices)
dice_mean = np.mean(dices)
dice_var = np.var(dices)
# class_f1_mean = np.sum(class_f1)/n
# class_f1_mean = class_f1
print("dice_max: {}, dice_min: {}, dice_mean: {}, dice_var: {}".format(dice_max, dice_min, dice_mean, dice_var))