import math
import cv2
import numpy as np
import random
from PIL import Image
from dataset.augment import tia_perspective, tia_stretch, tia_distort


class RecAug(object):
    def __init__(self, use_tia=True, aug_prob=0.4, **kwargs):
        self.use_tia = use_tia
        self.aug_prob = aug_prob

    def __call__(self, img_path):
        img = cv2.imread(str(img_path))
        img = warp(img, 10, self.use_tia, self.aug_prob)
        
        return img

def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (3, 3), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


class Config:
    """
    Config
    """

    def __init__(self, use_tia):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 5 * flag()
        self.angley = random.random() * 5 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.w = w
        self.h = h

        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia

        self.crop = True
        self.affine = False
        self.reverse = True
        self.noise = True
        self.jitter = True
        self.blur = True
        self.color = True


def rad(x):
    """
    rad
    """
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w**2 + h**2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0], [
                       0,
                       -np.sin(rad(anglex)),
                       np.cos(rad(anglex)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0], [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # generate 4 points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # Project onto the image plane
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio, dst


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz

def En(image):
    x, y,_ = image.shape  # 获取图片大小
    radius = np.random.randint(10, int(min(x, y)), 1)  #
    pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
    pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    radius = int(radius[0])
    strength = 100
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                # print(result)
                image[i, j, 0] = min((image[i, j, 0] + result),255)
                image[i, j, 1] = min((image[i, j, 1] + result),255)
                image[i, j, 2] = min((image[i, j, 2] + result),255)
    image = image.astype(np.uint8)
    return image

def De(image):
    x, y,_ = image.shape  # 获取图片大小
    radius = np.random.randint(10, int(min(x, y)), 1)  #
    pos_x = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心点坐标
    pos_y = np.random.randint(0, (min(x, y) - radius), 1)  # 获取人脸光照区域的中心坐标
    pos_x = int(pos_x[0])
    pos_y = int(pos_y[0])
    radius = int(radius[0])
    strength = 100
    for j in range(pos_y - radius, pos_y + radius):
        for i in range(pos_x-radius, pos_x+radius):

            distance = math.pow((pos_x - i), 2) + math.pow((pos_y - j), 2)
            distance = np.sqrt(distance)
            if distance < radius:
                result = 1 - distance / radius
                result = result*strength
                # print(result)
                image[i, j, 0] = max((image[i, j, 0] - result),0)
                image[i, j, 1] = max((image[i, j, 1] - result),0)
                image[i, j, 2] = max((image[i, j, 2] - result),0)
    image = image.astype(np.uint8)
    return image

def warp(img, ang, use_tia=True, prob=0.4):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, ang)
    new_img = img

    if config.distort:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            # new_img = tia_distort(new_img, random.randint(3, 6))
            new_img = tia_distort(new_img, random.randint(6, 9))

    if config.stretch:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 20 and img_width >= 20:
            # new_img = tia_stretch(new_img, random.randint(3, 6))
            new_img = tia_stretch(new_img, random.randint(6, 9))

    # if config.perspective:
    #     if random.random() <= prob:
    #         new_img = tia_perspective(new_img)

    # if config.crop:
    #     img_height, img_width = img.shape[0:2]
    #     if random.random() <= prob and img_height >= 20 and img_width >= 20:
    #         new_img = get_crop(new_img)

    # if config.blur:
    #     if random.random() <= prob:
    #         new_img = blur(new_img)
    if config.color:
        if random.random() <= prob:
            new_img = cvtColor(new_img)
    if config.jitter:
        new_img = jitter(new_img)
    if config.noise:
        if random.random() <= prob:
            new_img = add_gasuss_noise(new_img)
    # if config.reverse:
    #     if random.random() <= prob:
    #         new_img = 255 - new_img
    
    if random.random() <= prob:
        new_img = En(new_img)
    if random.random() <= prob:
        new_img = De(new_img)
    
    return new_img

if __name__ == '__main__':
    aug = RecAug(aug_prob=1)
    
    import os
    import cv2
    from pathlib import Path
    from multiprocessing.pool import ThreadPool
    
    image_list = list(Path('/home/cvrsg/Huiye_Datasets/OCR_Task/OCR_Dataset/images/train').glob('*.jpg'))
    
    # for image in image_list:
    #     img = cv2.imread(str(image))
    #     img = aug(img)
    #     # cv2.imwrite(os.path.join('results', image.name), img)
    #     cv2.imwrite('aug.jpg', img)
    #     cv2.imwrite('origin.jpg', cv2.imread(str(image)))
    
    pool = ThreadPool()
    pool.map(aug, image_list)
    pool.close()
    pool.join()