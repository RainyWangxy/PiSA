import argparse
from pycocotools.coco import COCO
import numpy as np
import os
from PiSA import set_random_seed
pp = argparse.ArgumentParser(description='')
pp.add_argument('--dataset', type=str, default='pascal', choices=['pascal', 'coco'])
args = pp.parse_args()
set_random_seed(0)
def get_max_label_from_coco_segmentation(coco_annotation_file, image_dir):
    coco = COCO(coco_annotation_file)
    cat_to_idx = {}
    cat_cnt = 0
    for cat_id in coco.cats.keys():
        cat_to_idx[cat_id] = cat_cnt
        cat_cnt += 1
    image_ids = coco.getImgIds()
    max_label_dict = {}
    
    for image_id in image_ids:
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        max_area = 0
        annotations = coco.loadAnns(annotation_ids)        
        max_label = None
        
        for ann in annotations:
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                area = ann['area']
                if area > max_area:
                    max_area = area
                    max_label = cat_to_idx[ann['category_id']]
        if max_label != None:
            max_label_dict[image_id] = max_label
    Image_Ids_seg = []
    single_salient_seg = [] 
    for img_idx, label in max_label_dict.items():
        label_init = [0] * 80
        Image_Ids_seg.append(f"train2014/COCO_train2014_{str(int(img_idx)).zfill(12)}.jpg")
        label_init[label] = 1
        single_salient_seg.append(label_init)
    single_salient_seg = np.array(single_salient_seg)
    Image_Ids_seg = np.array(Image_Ids_seg)
    
    
    train_labels_seg = np.zeros((len(Image_Ids_seg), 80))
    train_labels_all = np.load("format_label/coco/formatted_train_labels.npy")
    train_img_all = np.load("format_label/coco/formatted_train_images.npy")
    all_dict = {}
    for idx in range(train_img_all.shape[0]):
        all_dict[train_img_all[idx]] = train_labels_all[idx]
    for idx in range(Image_Ids_seg.shape[0]):
        train_labels_seg[idx] = all_dict[Image_Ids_seg[idx]]
    np.save("format_label/coco/formatted_train_labels_seg.npy",train_labels_seg)
    np.save("format_label/coco/single_salient_seg.npy",single_salient_seg)
    np.save("format_label/coco/formatted_train_images_seg.npy",Image_Ids_seg)
    return 


import os
import numpy as np
from PIL import Image
from collections import Counter

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def get_largest_label_from_mask(mask_path, label_true):
    mask = np.array(Image.open(mask_path))

    counts = Counter(mask.flatten())
    if 0 in counts:
        del counts[0]  # 删除背景
    if 255 in counts:
        del counts[255]
    counts_cp = counts.copy()
    for idx in counts_cp.keys():
        if label_true[int(idx) - 1] == 0:
            del counts[idx]
    if not counts:
        return None
    
    largest_label = max(counts.items(), key=lambda x: x[1])[0]
    
    return int(largest_label) - 1 

def process_voc_dataset(segmentation_dir):
    ImgIds_seg = []
    single_label_seg = []
    train_labels_seg = []
    train_labels_all = np.load("format_label/pascal/formatted_train_labels.npy")
    train_img_all = np.load("format_label/pascal/formatted_train_images.npy")
    all_dict = {}
    for idx in range(train_img_all.shape[0]):
        all_dict[train_img_all[idx]] = list(train_labels_all[idx])    
    for filename in os.listdir(segmentation_dir):
        if not filename.endswith(".png"):
            continue
        
        mask_path = os.path.join(segmentation_dir, filename)
        # 获取图像 ID
        img_id = filename.split(".png")[0]
        img_id_jpg = f"{img_id}.jpg"
        if img_id_jpg not in all_dict:
            continue
        label = get_largest_label_from_mask(mask_path,all_dict[img_id_jpg])
        if label!= None:
            ImgIds_seg.append(img_id_jpg)
            train_labels_seg.append(all_dict[img_id_jpg])
            label_init = [0] * 20
            label_init[label] = 1
            single_label_seg.append(label_init)
            
        else:
            print(f"No valid label found for {filename}")
    ImgIds_seg = np.array(ImgIds_seg)
    single_label_seg = np.array(single_label_seg)
    train_labels_seg = np.array(train_labels_seg)
    np.save("format_label/pascal/formatted_train_labels_seg.npy",train_labels_seg)
    np.save("format_label/pascal/formatted_train_images_seg.npy",ImgIds_seg)
    np.save("format_label/pascal/single_salient_seg.npy",single_label_seg)
    return 

if __name__ == "__main__":
    dataName = args.dataset # "coco", "nuswide", "pascal"
    if dataName == "coco":
        coco_annotation_file = "data/coco/annotations/instances_train2014.json"
        image_dir = "data/coco/train2014"
        get_max_label_from_coco_segmentation(coco_annotation_file, image_dir)
    elif dataName == "nuswide":
        pass
    elif dataName == "pascal":
        segmentation_dir = "data/pascal/VOCdevkit/VOC2012/SegmentationClass"
        process_voc_dataset(segmentation_dir)
    else:
        print("no dataset found")