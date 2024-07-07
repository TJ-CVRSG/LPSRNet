import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from ccpd_utils import (
    get_all_image_data_in_folder,
    get_lptext,
)

data_dirs = [
    "./data/ccpd_lite/train",
    "./data/ccpd_lite/val",
    "./data/ccpd_lite/test",
]
output_image_dir_base = "./data/ccpd_lite_lpr/"

PERSPECTIVE_CALIB = True  # True: perspective calibration, False: bbox crop


if __name__ == "__main__":

    for data_dir in data_dirs:

        # read all image data
        image_data = get_all_image_data_in_folder(data_dir)
        print("Tolal number of images before filtering:", len(image_data))

        output_image_dir = os.path.join(output_image_dir_base, data_dir.split("/")[-1])

        # create output directory
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        # crop images
        for data in tqdm(image_data):
            image = cv2.imread(data["image_path"])

            if not PERSPECTIVE_CALIB:
                bbox = data["bbox"]
                # crop image
                region = image[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
            else:
                keypoint = data["keypoint"]
                pts_target = np.array(
                    [[128, 32], [0, 32], [0, 0], [128, 0]], dtype=np.float32
                )
                h_mat, _ = cv2.findHomography(keypoint, pts_target)
                if h_mat is None:
                    print(data["image_path"])
                    continue
                region = cv2.warpPerspective(
                    image, h_mat, dsize=(128, 32), borderValue=(255, 255, 255)
                )

            # get lptext
            lptext = get_lptext(data["image_path"])

            if len(lptext) == 7:
                class_idx = 0
            elif len(lptext) == 8:
                class_idx = 1
            else:
                raise ValueError("Plate number length is not 7 or 8.")

            cv2.imwrite(
                os.path.join(
                    output_image_dir,
                    f"{lptext}_{class_idx}_{str(int(time.time() * 10000))}.jpg",
                ),
                region,
            )

    print("Done.")
