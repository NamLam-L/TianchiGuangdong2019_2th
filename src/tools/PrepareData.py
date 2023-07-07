# -*- coding: utf-8 -*-
import mmcv
import os
from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
import shutil
import numpy as np
# from icecream import ic
from collections import Counter
import mmengine
from mmengine.fileio import dump
import json
num_classes = 11
eng_dict = {0: 'non-conductive',
  1: 'Jet',
  2: 'wiping flowers',
  3: 'variegated',
  4: 'orange peel',
  5: 'Paint bubble',
  6: 'Leak the bottom',
  7: 'Dirty point',
  8: 'Corner leak bottom',
  9: 'Pitting'}
eng_dict_reverse = {'non-conductive':0,
  'Jet':1,
  'wiping flowers': 2,
  'variegated': 3,
  'orange peel':4,
  'Paint bubble': 5,
  'Leak the bottom': 6,
  'Dirty point': 7,
  'Corner leak bottom': 8,
  'Pitting': 9}
china_dict = {'不导电': 'non-conductive',
 '喷流' : 'Jet',
 '擦花': 'wiping flowers',
 '杂色': 'variegated',
 '桔皮': 'orange peel',
 '漆泡': 'Paint bubble',
 '漏底': 'Leak the bottom',
 '脏点': 'Dirty point',
 '角位漏底': 'Corner leak bottom',
 '起坑': 'Pitting'}
def construct_imginfo(filename, h, w, ID): # (file_name, h, ,w, id)
    image = {"license": 1,
             "file_name": filename,
             "coco_url": "xxx",
             "height": h,
             "width": w,
             "date_captured": "2019-06-25",
             "flickr_url": "xxx",
             "id": ID
             }
    return image


def construct_ann(obj_id, ID, category_id, seg, area, bbox):
    ann = {"id": obj_id,
           "image_id": ID,
           "category_id": category_id,
           "segmentation": seg,
           "area": area,
           "bbox": bbox,
           "iscrowd": 0,
           "ignore": 0
           }
    return ann


def add_normal(normal_dir, out_file):
    coco = COCO(out_file)
    ID = max(coco.getImgIds()) + 1
    annotations = mmengine.load(out_file)
    normal_list = os.listdir(normal_dir)
    for normal in tqdm(normal_list):
        source = "{}/{}".format(normal_dir, normal)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] + ".jpg"
        img_info = construct_imginfo("normal", filename, template, h, w, ID)
        ID += 1
        annotations["images"].append(img_info)
    print(len(annotations["images"]))
    a = open(out_file, 'w')
    a.close()
    mmcv.dump(annotations, out_file)


def generate_normal(normal_dir, out_file):
    cls2ind = mmengine.load("./source/cls2ind.pkl")
    ind2cls = mmengine.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    ID = 0
    normal_list = os.listdir(normal_dir)
    for normal in tqdm(normal_list):
        source = "{}/{}".format(normal_dir, normal)
        img = cv2.imread(source + "/{}.jpg".format(normal))
        h, w, _ = img.shape
        filename = normal + ".jpg"
        template = normal.split("_")[0] + ".jpg"
        img_info = construct_imginfo("normal", filename, template, h, w, ID)
        ID += 1
        annotations["images"].append(img_info)
    print(len(annotations["images"]))
    # a = open(out_file, 'w')
    # a.close()
    dump(annotations, out_file)

def generate_coco(annos, out_file):
    # cls2ind = mmengine.load("/content/drive/MyDrive/Aluko/Fabric/source/cls2ind.pkl")
    # ind2cls = mmengine.load("/content/drive/MyDrive/Aluko/Fabric/source/ind2cls.pkl")
    ind2cls = eng_dict
    

    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind, clss in ind2cls.items():
        category = {"id": ind, "name": clss, "supercategory": "object", }
        categories.append(category)
    annotations = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    img_names = {}
    IMG_ID = 0
    OBJ_ID = 0
    for img_name in tqdm(os.listdir(annos)):
        if("json" in img_name):
            continue
        else:
            file_name = os.path.join(annos, img_name)
            json_file_path = os.path.join(annos, img_name.split(".")[0] + ".json")
            if(file_name not in img_names):
                img_names[file_name] = IMG_ID
                img = cv2.imread(file_name)
                h, w, _ = img.shape
                img_info = construct_imginfo(file_name, h, w, IMG_ID)
                annotations["images"].append(img_info)
                IMG_ID = IMG_ID + 1
            if(os.path.exists(json_file_path)): # The image with bb or the flaw images
                img_id = img_names[file_name]
                root = mmengine.load(json_file_path)
                for obj in root["shapes"]:
                    class_name = obj["label"]
                    corners = [obj["points"][0][1] , obj["points"][0][0] , obj["points"][0][1] , obj["points"][0][0] ] #y_min, x_min, y_max, x_max
                    for c_point in obj["points"][1:]:
                        if(c_point[0]  < corners[1]):
                            corners[1] = c_point[0] 
                        elif(c_point[0]  > corners[3]):
                            corners[3] = c_point[0] 
                        if(c_point[1]  < corners[0]):
                            corners[0] = c_point[1] 
                        elif(c_point[1] > corners[2]):
                            corners[2] = c_point[1] 
                    cat_ID = eng_dict_reverse[china_dict[class_name]]
                    ymin, xmin, ymax, xmax = corners
                    area = (ymax- ymin) * (xmax - xmin)
                    seg = [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]]
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    ann = construct_ann(OBJ_ID, img_id, cat_ID, seg, area, bbox)
                    annotations["annotations"].append(ann)
                    OBJ_ID += 1
            else: # The flawness images TODO: need to implement
                pass
    print(len(annotations["images"]))
    print(annotations)
    with open(out_file, "w") as outfile:
      json.dump(annotations, outfile)
    # a = open(out_file, 'w')
    # a.close()
    # dump(annotations, out_file)


def generate_train(coco, val):
    cls2ind = mmengine.load("./source/cls2ind.pkl")
    ind2cls = mmengine.load("./source/ind2cls.pkl")
    info = {
        "description": "cloth",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    }
    license = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"}]
    categories = []
    for ind in range(len(ind2cls)):
        category = {"id": ind, "name": ind2cls[ind], "supercategory": "object", }
        categories.append(category)
    anno_train = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    anno_val = {"info": info, "images": [], "annotations": [], "categories": categories, "license": license}
    ids = coco.getImgIds()
    for imgId in ids:
        img_info = coco.loadImgs(imgId)[0]
        cls = img_info["cls"]
        ann_ids = coco.getAnnIds(img_info['id'])
        ann_info = coco.loadAnns(ann_ids)
        if cls in val:
            anno_val["images"].append(img_info)
            anno_val["annotations"] += ann_info
        else:
            anno_train["images"].append(img_info)
            anno_train["annotations"] += ann_info
    mmcv.dump(anno_train, "../data/round2_data/Annotations/anno_train.json")
    mmcv.dump(anno_val, "../data/round2_data/Annotations/anno_val.json")


def split(out_file):
    all_class = mmengine.load("./source/temp_cls.pkl")
    np.random.seed(1)
    val = np.random.choice(all_class, 28, replace=False)
    train = list(set(all_class) - set(val))
    coco = COCO(out_file)
    generate_train(coco, val)

if __name__ == "__main__":
    # data_root = "../data/"
    data_dir = "/content"
    annos = "/content/drive/MyDrive/Aluko/train_data"
    out_file = "{}/train_anno.json".format(data_dir)
    print("convert to coco format...")
    generate_coco(annos, out_file)
    # add_normal("../data/round2_data/normal", out_file)