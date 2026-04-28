import cv2
import numpy as np
from utils import convert_to_gray


class ImageStitcher:
    def __init__(self, detector_type='SIFT'):
        """
        初始化图像拼接器
        :param detector_type: 特征检测器类型，'SIFT' 或 'ORB'
        """
        self.detector_type = detector_type.upper()
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
    
    def _create_detector(self):
        """创建特征检测器"""
        if self.detector_type == 'SIFT':
            try:
                return cv2.SIFT_create()
            except AttributeError:
                print("警告: OpenCV未安装SIFT，使用ORB代替")
                return cv2.ORB_create()
        elif self.detector_type == 'ORB':
            return cv2.ORB_create()
        else:
            raise ValueError(f"不支持的检测器类型: {self.detector_type}")
    
    def _create_matcher(self):
        """创建特征匹配器"""
        if self.detector_type == 'SIFT':
            return cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def detect_and_compute(self, image):
        """检测特征点并计算描述符"""
        gray = convert_to_gray(image)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, descriptors1, descriptors2, ratio_thresh=0.75):
        """
        特征匹配
        :param ratio_thresh: Lowe's ratio test阈值
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_homography(self, keypoints1, keypoints2, matches, ransac_thresh=5.0):
        """
        估计单应性矩阵
        """
        if len(matches) < 4:
            print("警告: 匹配点不足，无法估计单应性矩阵")
            return None, None
        
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        
        return H, mask
    
    def warp_images(self, image1, image2, H):
        """
        将image1变换到image2的坐标系并拼接
        """
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        corners1_warped = cv2.perspectiveTransform(corners1, H)
        all_corners = np.concatenate((corners2, corners1_warped), axis=0)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]],
                                   [0, 1, translation_dist[1]],
                                   [0, 0, 1]])
        
        H_warp = H_translation.dot(H)
        
        output_size = (x_max - x_min, y_max - y_min)
        result1 = cv2.warpPerspective(image1, H_warp, output_size)
        
        mask = np.ones_like(image2, dtype=np.uint8) * 255
        mask_warped = cv2.warpPerspective(mask, H_warp, output_size)
        
        result2 = np.zeros_like(result1)
        result2[translation_dist[1]:translation_dist[1] + h2,
               translation_dist[0]:translation_dist[0] + w2] = image2
        
        mask2 = np.zeros_like(result1)
        mask2[translation_dist[1]:translation_dist[1] + h2,
              translation_dist[0]:translation_dist[0] + w2] = 255
        
        blended = self._blend_images(result1, result2, mask_warped, mask2)
        
        return blended
    
    def _blend_images(self, img1, img2, mask1, mask2):
        """
        图像融合（简单的加权平均或直接覆盖）
        """
        blend = np.zeros_like(img1)
        
        mask1_gray = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY) if len(mask1.shape) == 3 else mask1
        mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY) if len(mask2.shape) == 3 else mask2
        
        overlap_mask = cv2.bitwise_and(mask1_gray, mask2_gray)
        non_overlap1 = cv2.bitwise_and(mask1_gray, cv2.bitwise_not(overlap_mask))
        non_overlap2 = cv2.bitwise_and(mask2_gray, cv2.bitwise_not(overlap_mask))
        
        for c in range(img1.shape[2]):
            blend[:, :, c] = np.where(non_overlap1 > 0, img1[:, :, c], 0)
            blend[:, :, c] = np.where(non_overlap2 > 0, img2[:, :, c], blend[:, :, c])
            blend[:, :, c] = np.where(overlap_mask > 0, 
                                       (img1[:, :, c].astype(np.float32) + img2[:, :, c].astype(np.float32)) / 2,
                                       blend[:, :, c])
        
        return blend.astype(np.uint8)
    
    def stitch_two_images(self, image1, image2):
        """
        拼接两张图片（image1在左，image2在右）
        """
        kp1, desc1 = self.detect_and_compute(image1)
        kp2, desc2 = self.detect_and_compute(image2)
        
        if desc1 is None or desc2 is None:
            print("警告: 无法检测到特征点")
            return None
        
        matches = self.match_features(desc1, desc2)
        print(f"找到 {len(matches)} 个良好匹配点")
        
        if len(matches) < 4:
            print("警告: 匹配点不足，尝试反向匹配")
            matches = self.match_features(desc2, desc1)
            if len(matches) < 4:
                print("错误: 无法找到足够的匹配点")
                return None
            
            H, mask = self.estimate_homography(kp2, kp1, matches)
            if H is None:
                print("错误: 无法估计单应性矩阵")
                return None
            
            result = self.warp_images(image2, image1, H)
        else:
            H, mask = self.estimate_homography(kp1, kp2, matches)
            if H is None:
                print("错误: 无法估计单应性矩阵")
                return None
            
            result = self.warp_images(image1, image2, H)
        
        return result
    
    def stitch_multiple_images(self, images, order='left_to_right'):
        """
        拼接多张图片
        :param images: 图片列表，按从左到右顺序
        :param order: 拼接顺序，'left_to_right' 或 'right_to_left'
        """
        if len(images) == 0:
            raise ValueError("图片列表为空")
        if len(images) == 1:
            return images[0]
        
        if order == 'left_to_right':
            result = images[0]
            for i in range(1, len(images)):
                print(f"正在拼接第 {i+1} 张图片...")
                stitched = self.stitch_two_images(result, images[i])
                if stitched is None:
                    print(f"警告: 无法拼接第 {i+1} 张图片，跳过")
                    continue
                result = stitched
        else:
            result = images[-1]
            for i in range(len(images)-2, -1, -1):
                print(f"正在拼接第 {i+1} 张图片...")
                stitched = self.stitch_two_images(images[i], result)
                if stitched is None:
                    print(f"警告: 无法拼接第 {i+1} 张图片，跳过")
                    continue
                result = stitched
        
        return result
    
    def remove_black_borders(self, image):
        """
        去除拼接后的黑边
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return image[y:y+h, x:x+w]
        
        return image


def create_panorama(images, detector_type='SIFT', remove_borders=True):
    """
    便捷函数：创建全景图
    :param images: 图片列表（从左到右）
    :param detector_type: 特征检测器类型
    :param remove_borders: 是否去除黑边
    """
    stitcher = ImageStitcher(detector_type=detector_type)
    panorama = stitcher.stitch_multiple_images(images)
    
    if remove_borders and panorama is not None:
        panorama = stitcher.remove_black_borders(panorama)
    
    return panorama
