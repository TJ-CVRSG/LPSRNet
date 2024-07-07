import cv2
import torch
import numpy as np
from generate_multi_plate_align import MultiPlateGeneratorAlign


CHAR_LIST = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "皖",
    "京",
    "渝",
    "闽",
    "甘",
    "粤",
    "桂",
    "贵",
    "琼",
    "冀",
    "黑",
    "豫",
    "鄂",
    "湘",
    "蒙",
    "苏",
    "赣",
    "吉",
    "辽",
    "宁",
    "青",
    "陕",
    "鲁",
    "沪",
    "晋",
    "川",
    "津",
    "藏",
    "新",
    "云",
    "浙",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    " ",
]

generator = MultiPlateGeneratorAlign(
    "./tools/generator/plate_model", "./tools/generator/font_model"
)

plate_array = np.ones((len(CHAR_LIST), 32, 128))

for char in CHAR_LIST:
    if char == " ":
        break
    plate = char * 8
    plate_image = generator.generate_plate_special(plate, "green_car", False)
    plate_image = cv2.resize(plate_image, (128, 32)) / 255.0
    plate_array[CHAR_LIST.index(char)] = plate_image


torch.save(
    torch.from_numpy(plate_array.astype(np.float32)), "./data/plate_template.pt"
)
