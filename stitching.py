import cv2
import numpy as np


def resize_to_same_height(images):
    """
    将所有图片调整为相同的高度（以最高的图片为准）
    :param images: 图片列表
    :return: 调整后的图片列表
    """
    if len(images) == 0:
        return []
    
    max_height = max(img.shape[0] for img in images)
    
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        if h != max_height:
            scale = max_height / h
            new_width = int(w * scale)
            resized = cv2.resize(img, (new_width, max_height), interpolation=cv2.INTER_AREA)
            resized_images.append(resized)
        else:
            resized_images.append(img)
    
    return resized_images


def stitch_horizontal_simple(images):
    """
    简单的横向拼接：将图片从左到右依次排列
    :param images: 图片列表（按从左到右顺序）
    :return: 拼接后的全景图
    """
    if len(images) == 0:
        raise ValueError("图片列表为空")
    if len(images) == 1:
        return images[0]
    
    resized_images = resize_to_same_height(images)
    
    panorama = resized_images[0]
    for i in range(1, len(resized_images)):
        panorama = cv2.hconcat([panorama, resized_images[i]])
    
    return panorama


def create_panorama(images, method='simple'):
    """
    创建全景图
    :param images: 图片列表（从左到右顺序）
    :param method: 拼接方法，'simple' 为简单横向拼接
    :return: 拼接后的全景图
    """
    print(f"使用简单横向拼接方法，共 {len(images)} 张图片")
    
    panorama = stitch_horizontal_simple(images)
    
    return panorama
