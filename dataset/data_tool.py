import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import hydra
from omegaconf import DictConfig

# import sys
# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

from dataset.rec_aug import RecAug

CHAR_DICT = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "皖": 10,
    "京": 11,
    "渝": 12,
    "闽": 13,
    "甘": 14,
    "粤": 15,
    "桂": 16,
    "贵": 17,
    "琼": 18,
    "冀": 19,
    "黑": 20,
    "豫": 21,
    "鄂": 22,
    "湘": 23,
    "蒙": 24,
    "苏": 25,
    "赣": 26,
    "吉": 27,
    "辽": 28,
    "宁": 29,
    "青": 30,
    "陕": 31,
    "鲁": 32,
    "沪": 33,
    "晋": 34,
    "川": 35,
    "津": 36,
    "藏": 37,
    "新": 38,
    "云": 39,
    "浙": 40,
    "A": 41,
    "B": 42,
    "C": 43,
    "D": 44,
    "E": 45,
    "F": 46,
    "G": 47,
    "H": 48,
    "J": 49,
    "K": 50,
    "L": 51,
    "M": 52,
    "N": 53,
    "P": 54,
    "Q": 55,
    "R": 56,
    "S": 57,
    "T": 58,
    "U": 59,
    "V": 60,
    "W": 61,
    "X": 62,
    "Y": 63,
    "Z": 64,
    "": 65,
}


class PairedDataset(Dataset):
    def __init__(self, base_dir, synthetic_dir, enable_augment=False) -> None:
        super(PairedDataset, self).__init__()

        self.aug = RecAug(aug_prob=0.5)
        self.pairs = self.build_for_opencv_synthetic(
            hydra.utils.to_absolute_path(base_dir),
            hydra.utils.to_absolute_path(synthetic_dir),
        )
        self.enable_aug = enable_augment

    def build_for_opencv_synthetic(self, base_dir, synthetic_dir):
        pairs = []

        for root, _, files in os.walk(base_dir):
            if len(files) > 1:
                for f in files:
                    plate_a = f.split("_")[0]

                    # if plate_a[:2] == '皖A' and len(plate_a) == 7:
                    #     weight = 1
                    # else:
                    #     weight = 30

                    if len(plate_a) == 7:
                        weight = 1
                    else:
                        weight = 50

                    label = [CHAR_DICT[c] for c in plate_a]
                    label += [65] * (8 - len(label))
                    label = np.array(label, dtype=np.int64)

                    pairs.append(
                        {
                            "Ac_As": os.path.join(root, f),
                            "Bc_Bs_algin": os.path.join(
                                synthetic_dir, plate_a + ".jpg"
                            ),
                            "weight": weight,
                            "ocr_label": label,
                            "ocr_label_length": len(plate_a),
                        }
                    )

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # preprocess a_content_a_style_img
        if self.enable_aug:
            a_content_a_style_img = self.aug(self.pairs[index]["Ac_As"])
        else:
            a_content_a_style_img = cv2.imread(self.pairs[index]["Ac_As"])
        a_content_a_style_img = cv2.cvtColor(a_content_a_style_img, cv2.COLOR_BGR2RGB)
        a_content_a_style_img = (a_content_a_style_img.astype(np.float32) - 128) / 128
        a_content_a_style_img = a_content_a_style_img.transpose(2, 0, 1)

        # preprocess b_content_b_style_img_align
        b_content_b_style_img_align = cv2.imread(
            self.pairs[index]["Bc_Bs_algin"], cv2.IMREAD_GRAYSCALE
        )

        _, align_mask = cv2.threshold(
            255 - b_content_b_style_img_align,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        align_mask = align_mask.astype(np.float32)
        align_mask /= 255
        align_mask *= 5
        align_mask += 1
        align_mask *= 10

        b_content_b_style_img_align = (
            b_content_b_style_img_align.astype(np.float32) - 0
        ) / 255
        b_content_b_style_img_align = b_content_b_style_img_align[:, :, np.newaxis]
        b_content_b_style_img_align = b_content_b_style_img_align.transpose(2, 0, 1)

        return (
            a_content_a_style_img,
            b_content_b_style_img_align,
            align_mask,
            self.pairs[index]["weight"],
            self.pairs[index]["ocr_label"],
            self.pairs[index]["ocr_label_length"],
        )


class UnpairedDataset(Dataset):
    def __init__(self, base_dir) -> None:
        super(UnpairedDataset, self).__init__()
        self.pairs = self.build_for_opencv_synthetic(
            hydra.utils.to_absolute_path(base_dir)
        )

    def build_for_opencv_synthetic(self, base_dir):
        pairs = []
        for root, _, files in os.walk(base_dir):

            if len(files) > 1:
                for f in files:
                    plate_a = f.split("_")[0]
                    # plate_a = f.split('.')[0]

                    pairs.append(
                        {
                            "Ac_As": os.path.join(root, f),
                            "plate": plate_a,
                        }
                    )

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_content_a_style_img = cv2.imread(self.pairs[index]["Ac_As"])
        a_content_a_style_img = cv2.cvtColor(a_content_a_style_img, cv2.COLOR_BGR2RGB)
        a_content_a_style_img = (a_content_a_style_img.astype(np.float32) - 128) / 128
        a_content_a_style_img = a_content_a_style_img.transpose(2, 0, 1)

        return a_content_a_style_img, self.pairs[index]["plate"]


def get_dataloader(cfg_dataset: DictConfig):
    datasets = []
    for cfg in cfg_dataset.data:
        if cfg.enabled:
            datasets.append(
                PairedDataset(cfg.base_dir, cfg.synthetic_dir, cfg.enable_augment)
            )

    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg_dataset.batch_size,
        shuffle=True,
        num_workers=cfg_dataset.num_workers,
        pin_memory=True,
    )

    return dataloader


def get_test_dataloader(base_dir, batch_size=64, num_workers=16):
    dataset = UnpairedDataset(base_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    # dataloader = DataLoader(dataset, batch_size=batch_size,
    #                         shuffle=True, num_workers=num_workers, pin_memory=False)

    return dataloader


# debug
if __name__ == "__main__":
    base_dir = "./data/ccpd/train"
    synthetic_dir = "./data/opencv_synthetic"
    dataset = PairedDataset(base_dir, synthetic_dir)

    print(len(dataset))
    for i in range(100):
        (
            a_content_a_style_img,
            b_content_b_style_img_align,
            align_mask,
            weight,
            ocr_label,
            ocr_label_length,
        ) = dataset[i]

        print(weight)

        cv2.imshow("align_mask", align_mask.astype(np.uint8))
        cv2.imshow("a_content_a_style_img", a_content_a_style_img)
        cv2.imshow("b_content_b_style_img_align", b_content_b_style_img_align)
        cv2.waitKey(0)

    print("Done")
