import os
import cv2
from multiprocessing.pool import ThreadPool
from pathlib import Path

from generate_multi_plate_align import MultiPlateGeneratorAlign

img_dirs = [
    "./data/ccpd_lite_lpr/train",
    "./data/ccpd_lite_lpr/val",
]

output_dir = "./data/opencv_synthetic"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

generator_align = MultiPlateGeneratorAlign(
    "./tools/generator/plate_model", "./tools/generator/font_model"
)


def generate_and_save_plate_image(image_path):
    plate = image_path.stem.split("_")[0]

    plate_image_align = generator_align.generate_plate_special(
        plate + " ", "green_car", False
    )

    plate_image_align = cv2.resize(plate_image_align, (128, 32))

    cv2.imwrite(os.path.join(output_dir, plate + ".jpg"), plate_image_align)


image_path_list = []
for img_dir in img_dirs:
    image_path_list += list(Path(img_dir).glob("*.jpg"))

print(len(image_path_list))
pool = ThreadPool()
pool.map(generate_and_save_plate_image, image_path_list)
pool.close()
pool.join()
print(f"done!")
