
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
# %%
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
KTF.set_session(sess)
# %%
# 修改为自己的识别类别
class_names = ['BG', 'liver_0','liver_1','liver_2','liver_3','liver_4']
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
    NUM_CLASSES = 5 + 1  # background + 3 shapes

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

    BACKBONE = "resnet101"

    USE_MINI_MASK = False

    LEARNING_RATE = 3e-6

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.,
    }


class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
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

from sklearn.model_selection import train_test_split
patient_list_intrain, patient_list_test = train_test_split(patient_list, test_size=8, random_state=42)
patient_list_train, patient_list_val = train_test_split(patient_list_intrain, test_size=0.3, random_state=42)

data_list_test = []
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
        image = np.stack([convert(img0.copy(), 1200, 50),
                          convert(img0.copy(),  600, 50),
                          convert(img0.copy(),  300, 50)],
                            axis=-1)
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

def caculate_F1(GS, PS):
    n = len(class_names)
    LS = list(range(n))
    TP = FP = FN = 0
    for i in LS:
        if i in PS and i in GS:
            TP+=1
        elif i in PS and i not in GS:
            FP+=1
        elif i not in PS and i in GS:
            FN+=1
    P = TP / (TP+FP)
    R = TP / (TP+FN)
    return P, R

def pixel2area(ds):
    return ds.PixelSpacing[0] * ds.PixelSpacing[1]
# %%
# data_list_test = data_list_test[:5]
n = len(data_list_test)
test_res = [] #[(id,dice,P,R),...]
err_list = []
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
    # if main_id != "a92_p_00023":
    #     continue
    # if main_id not in ["a97_p_00021", "a17_d_00018", "a02_p_00016"]:
    #     continue
    # if main_id != "a92_p_00022":
    #     continue
    ds, img = load_image(dcm_path)
    
    img=cv2.resize(img,(width,height))

    # Run detection
    results = model.detect([img], verbose=0)

    # Visualize results
    r = results[0]
    # try:
    class_name_true = "liver_"+str(df.values[patient_id_num-1,1])
    class_ids_true = [class_names.index(class_name_true)]
    class_ids_pred = r['class_ids']

    mask_true = mask_ascend_dim(load_mask(data_path), class_ids_true).sum(-1).astype(np.bool)
    mask_pred = mask_ascend_dim(r['masks'], class_ids_pred).sum(-1).astype(np.bool)
    
    dice = caculate_dice(mask_true, mask_pred)
    # precision, recall = caculate_F1(class_ids_true, class_ids_pred)
    precision, recall = 0, 0

    print("{}: {}/{}, dice: {}, class_eval: {}/{}".format(main_id, i+1, n, dice, precision, recall))
    test_res.append((main_id, dice, (precision, recall)))
    # except Exception as e:
    #     print("error: {}, {}".format(main_id, e))
    #     err_list.append((main_id, e))

    # visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'],
    #                             figsize=(8,8), show_bbox=True)
# %%
test_res.sort(key = lambda x:x[1])
ids = [x[0] for x in test_res]
dices = [x[1] for x in test_res]
class_precisions = [x[2][0] for x in test_res]
class_recalls = [x[2][1] for x in test_res]
# for x in test_res:
#     print("{}: {:.4f}, {:.4f}".format(x[0], x[1], x[2]))
# if err_list!=[]:
#     print("error count: {}".format(len(err_list)))
#     print("error samples: {}".format(err_list))
dice_max = max(test_res, key = lambda x:x[1])
dice_min = min(test_res, key = lambda x:x[1])
dice_mean = np.sum(dices)/n
class_precision_mean = np.sum(class_precisions)/n
class_recall_mean = np.sum(class_recalls)/n
class_F1_mean = 2 * class_precision_mean * class_recall_mean / (class_precision_mean + class_recall_mean)
print("dice_max: {}, dice_min: {}, dice_mean: {}, class_F1_mean: {}".format(dice_max, dice_min, dice_mean, class_F1_mean))
# %%
writer = pd.ExcelWriter("test_res.xlsx")

res = {
    "id": ids,
    "dice": dices,
    "class_P": class_precisions,
    "class_R": class_recalls
}
res = pd.DataFrame(res)
res.to_excel(writer, sheet_name="result")

err = {
    "id": [x[0] for x in err_list],
    "type": [x[1] for x in err_list]
}
err = pd.DataFrame(err)
err.to_excel(writer, sheet_name="error")

abst = {
    "dice_mean": dice_mean,
    "class_P_mean": class_precision_mean,
    "class_R_mean": class_recall_mean,
    "class_F1_mean": class_F1_mean,
}
abst = pd.DataFrame(abst,index=[0]).T
abst.to_excel(writer, sheet_name="abstarct",na_rep='NULL',header=False)

writer.close()
# # %%
# with open("./test_res.log","w+") as f:
#     for x in err_list:
#         f.write("err:"+str(x)+"\n")
#     f.write("dice_mean:"+str(dice_mean)+"\n")
#     f.write("class_score_mean:"+str(class_score_mean)+"\n")
#     f.write("mask_score_mean:"+str(mask_score_mean)+"\n")
#     f.write("score_mean:"+str(score_mean)+"\n")
#     for x in test_res:
#         f.write(str(x[0])+":"+str(x[1])+","+str(x[2])+","+str(x[3])+","+str(x[4])+"\n")
# %%
plt.figure()
plt.plot(range(len(dices)), dices)
plt.grid(True)
plt.savefig(fname = "result.png", format = "png" ,dpi=500, bbox_inches = 'tight')
# %%