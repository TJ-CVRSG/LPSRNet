import os
import pathlib
import numpy as np

WIDTH = 720  # original image width for a ccpd image
HEIGHT = 1160   # original image height for a ccpd image

CCPD_PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽",
                  "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘",
                  "青", "宁", "新", "警", "学"]

CCPD_ALPHABETS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0"]

CCPD_ADS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
            "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "警", "学"]

# 删除“警”和“学”
PROVINCES_DICT = {'皖': '<Anhui>',
                  '沪': '<Shanghai>',
                  '津': '<Tianjin>',
                  '渝': '<Chongqing>',
                  '冀': '<Hebei>',
                  '晋': '<Shanxi>',
                  '蒙': '<InnerMongolia>',
                  '辽': '<Liaoning>',
                  '吉': '<Jilin>',
                  '黑': '<Heilongjiang>',
                  '苏': '<Jiangsu>',
                  '浙': '<Zhejiang>',
                  '京': '<Beijing>',
                  '闽': '<Fujian>',
                  '赣': '<Jiangxi>',
                  '鲁': '<Shandong>',
                  '豫': '<Henan>',
                  '鄂': '<Hubei>',
                  '湘': '<Hunan>',
                  '粤': '<Guangdong>',
                  '桂': '<Guangxi>',
                  '琼': '<Hainan>',
                  '川': '<Sichuan>',
                  '贵': '<Guizhou>',
                  '云': '<Yunnan>',
                  '藏': '<Tibet>',
                  '陕': '<Shaanxi>',
                  '甘': '<Gansu>',
                  '青': '<Qinghai>',
                  '宁': '<Ningxia>',
                  '新': '<Xinjiang>'}


def ccpd_bounding_box(image_path):
    """Returns the bounding box of a ccpd image."""
    img_id = image_path.split("/")[-1]
    if len(img_id.split("-")) != 7:
        keypoints_serial = '-'.join(img_id.split("-")[3:-3])
        keypoint_serial = keypoints_serial.split("_")
    else:
        keypoint_serial = img_id.split("-")[3].split("_")
    assert len(keypoint_serial) == 4

    keypoints = []
    for i in range(4):
        tmp = keypoint_serial[i].split("&")
        keypoints.append(int(tmp[0]))
        keypoints.append(int(tmp[1]))

    # get bounding box and transform to the cropped coordinate system
    xmin = min(keypoints[2], keypoints[4])
    ymin = min(keypoints[5], keypoints[7])
    xmax = max(keypoints[0], keypoints[6])
    ymax = max(keypoints[1], keypoints[3])

    return [xmin, ymin, xmax, ymax]


def ccpd_keypoint(image_path):
    """Returns the keypoints of a ccpd image."""
    img_id = image_path.split("/")[-1]
    if len(img_id.split("-")) != 7:
        keypoints_serial = '-'.join(img_id.split("-")[3:-3])
        keypoint_serial = keypoints_serial.split("_")
    else:
        keypoint_serial = img_id.split("-")[3].split("_")
    assert len(keypoint_serial) == 4

    keypoints = np.zeros((4, 3), dtype=np.float32)
    for i in range(4):
        tmp = keypoint_serial[i].split("&")
        keypoints[i, 0] = int(tmp[0])
        keypoints[i, 1] = int(tmp[1])

    return keypoints


def get_all_image_data_in_folder(folder_path):
    """Returns a list of all image data in a folder."""
    image_data = []

    folder = pathlib.Path(folder_path)
    image_paths = sorted(list(map(str, list(folder.glob("*.jpg")))))

    for image_path in image_paths:
        bbox = ccpd_bounding_box(image_path)
        keypoints = ccpd_keypoint(image_path)
        image_data.append({"image_path": image_path, 
                           "bbox": bbox, 
                           "keypoint": keypoints})

    return image_data

def get_all_image_data_from_txt(base_dir, txt_path):
    """Returns a list of all image data in a txt file."""
    image_data = []
    
    with open(txt_path, 'r') as f:
        image_list = f.readlines()
        
        for image_path in image_list:
            image_path = os.path.join(base_dir, image_path.strip())
            bbox = ccpd_bounding_box(image_path)
            keypoints = ccpd_keypoint(image_path)
            image_data.append({"image_path": image_path, 
                            "bbox": bbox, 
                            "keypoint": keypoints})
    
    return image_data


def filter_images(image_data,
                  min_ratio=2.0,
                  max_ratio=4.0,
                  min_scale=0.1,
                  max_scale=0.7):
    """Returns a list of images that satisfy our requirements."""

    filtered_image_data = []
    for data in image_data:
        xmin, ymin, xmax, ymax = data["bbox"]

        if xmin < 0 or ymin < 0 or xmax > WIDTH or ymax > HEIGHT:
            continue

        bbox_witdh = float(xmax - xmin)
        bbox_height = float(ymax - ymin)

        bbox_ratio = bbox_witdh / bbox_height
        # filter out images with overly skewed license plates
        if bbox_ratio < min_ratio or bbox_ratio > max_ratio:
            continue

        bbox_scale = bbox_witdh / WIDTH
        # filter out images with license plates that are too small or too large
        if bbox_scale < min_scale or bbox_scale > max_scale:
            continue

        filtered_image_data.append(data)

    return filtered_image_data


def get_lptext(image_path):
    lptext = ""
    img_id = image_path.split("/")[-1]
    text_idx = img_id.split("-")[-3].split("_")
    lptext += CCPD_PROVINCES[int(text_idx[0])]
    lptext += CCPD_ALPHABETS[int(text_idx[1])]
    for i in range(2, len(text_idx)):
        lptext += CCPD_ADS[int(text_idx[i])]
    return lptext
