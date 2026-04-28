import cv2
import numpy as np
from utils import convert_to_gray, draw_bounding_box, draw_text


class GeometricDetector:
    def __init__(self, min_area=500, max_area=50000):
        """
        初始化几何物体检测器
        :param min_area: 最小检测面积
        :param max_area: 最大检测面积
        """
        self.min_area = min_area
        self.max_area = max_area
        self.reference_objects = []
        self.reference_descriptors = []
    
    def preprocess_image(self, image):
        """
        图像预处理：灰度化、高斯模糊、边缘检测
        """
        gray = convert_to_gray(image)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        return eroded
    
    def find_contours(self, binary_image):
        """
        查找轮廓
        """
        contours, hierarchy = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def filter_contours(self, contours):
        """
        根据面积过滤轮廓
        """
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                filtered.append(cnt)
        return filtered
    
    def get_shape_features(self, contour):
        """
        获取形状特征（Hu矩）
        """
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    def get_color_features(self, image, contour):
        """
        获取颜色特征
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        if len(image.shape) == 3:
            mean_color = cv2.mean(image, mask=mask)[:3]
        else:
            mean_color = (cv2.mean(image, mask=mask)[0],) * 3
        
        return np.array(mean_color)
    
    def get_bounding_box(self, contour):
        """
        获取边界框
        """
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, w, h)
    
    def compute_shape_similarity(self, features1, features2):
        """
        计算形状相似度（基于Hu矩的欧氏距离）
        """
        if features1 is None or features2 is None:
            return 0.0
        
        distance = np.linalg.norm(features1 - features2)
        
        similarity = np.exp(-distance / 5.0)
        
        return max(0.0, min(1.0, similarity))
    
    def compute_color_similarity(self, color1, color2):
        """
        计算颜色相似度
        """
        if color1 is None or color2 is None:
            return 0.5
        
        distance = np.linalg.norm(color1 - color2)
        max_distance = np.sqrt(3 * (255 ** 2))
        
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def compute_overall_similarity(self, shape_sim, color_sim, shape_weight=0.7, color_weight=0.3):
        """
        计算综合相似度
        """
        return shape_sim * shape_weight + color_sim * color_weight
    
    def set_reference_objects(self, images):
        """
        设置参考物体（从多张图片中提取所有几何物体作为参考）
        """
        self.reference_objects = []
        self.reference_descriptors = []
        
        for img_idx, image in enumerate(images):
            binary = self.preprocess_image(image)
            contours = self.find_contours(binary)
            filtered = self.filter_contours(contours)
            
            for cnt in filtered:
                shape_features = self.get_shape_features(cnt)
                color_features = self.get_color_features(image, cnt)
                bbox = self.get_bounding_box(cnt)
                
                self.reference_objects.append({
                    'shape_features': shape_features,
                    'color_features': color_features,
                    'bbox': bbox,
                    'source_img': img_idx
                })
        
        print(f"从 {len(images)} 张图片中提取了 {len(self.reference_objects)} 个参考物体")
    
    def cluster_objects(self, similarity_threshold=0.6):
        """
        对参考物体进行聚类，找出同类几何物体
        """
        if len(self.reference_objects) == 0:
            return []
        
        clusters = []
        
        for i, obj in enumerate(self.reference_objects):
            matched = False
            
            for cluster in clusters:
                ref_obj = cluster[0]
                
                shape_sim = self.compute_shape_similarity(
                    obj['shape_features'], 
                    ref_obj['shape_features']
                )
                color_sim = self.compute_color_similarity(
                    obj['color_features'], 
                    ref_obj['color_features']
                )
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim)
                
                if overall_sim >= similarity_threshold:
                    cluster.append(obj)
                    matched = True
                    break
            
            if not matched:
                clusters.append([obj])
        
        largest_cluster = []
        for cluster in clusters:
            if len(cluster) > len(largest_cluster):
                largest_cluster = cluster
        
        print(f"找到 {len(clusters)} 个聚类，最大聚类包含 {len(largest_cluster)} 个物体")
        
        return largest_cluster
    
    def detect_objects_in_image(self, image, source_img_idx, reference_objects, similarity_threshold=0.5):
        """
        在单张图片中检测与参考物体相似的几何物体
        """
        detections = []
        
        binary = self.preprocess_image(image)
        contours = self.find_contours(binary)
        filtered = self.filter_contours(contours)
        
        for cnt in filtered:
            shape_features = self.get_shape_features(cnt)
            color_features = self.get_color_features(image, cnt)
            bbox = self.get_bounding_box(cnt)
            
            max_similarity = 0.0
            best_match_ref = None
            
            for ref_obj in reference_objects:
                shape_sim = self.compute_shape_similarity(shape_features, ref_obj['shape_features'])
                color_sim = self.compute_color_similarity(color_features, ref_obj['color_features'])
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim)
                
                if overall_sim > max_similarity:
                    max_similarity = overall_sim
                    best_match_ref = ref_obj
            
            if max_similarity >= similarity_threshold:
                detection = {
                    'box': bbox,
                    'similarity': float(max_similarity),
                    'source_img': int(source_img_idx),
                    'contour_area': float(cv2.contourArea(cnt))
                }
                detections.append(detection)
        
        return detections
    
    def detect_objects_in_panorama(self, panorama, reference_objects, similarity_threshold=0.5):
        """
        在全景图中检测相似几何物体
        """
        detections = []
        
        binary = self.preprocess_image(panorama)
        contours = self.find_contours(binary)
        filtered = self.filter_contours(contours)
        
        for cnt in filtered:
            shape_features = self.get_shape_features(cnt)
            color_features = self.get_color_features(panorama, cnt)
            bbox = self.get_bounding_box(cnt)
            
            max_similarity = 0.0
            
            for ref_obj in reference_objects:
                shape_sim = self.compute_shape_similarity(shape_features, ref_obj['shape_features'])
                color_sim = self.compute_color_similarity(color_features, ref_obj['color_features'])
                overall_sim = self.compute_overall_similarity(shape_sim, color_sim)
                
                if overall_sim > max_similarity:
                    max_similarity = overall_sim
            
            if max_similarity >= similarity_threshold:
                detection = {
                    'box': bbox,
                    'similarity': float(max_similarity),
                    'source_img': -1,
                    'contour_area': float(cv2.contourArea(cnt))
                }
                detections.append(detection)
        
        return detections


def detect_and_annotate(images, panorama=None, similarity_threshold=0.5, min_area=500, max_area=50000):
    """
    便捷函数：检测所有图片中的同类几何物体并标注
    :param images: 原始图片列表
    :param panorama: 拼接后的全景图（可选）
    :param similarity_threshold: 相似度阈值
    :param min_area: 最小检测面积
    :param max_area: 最大检测面积
    :return: 所有检测结果，标注后的全景图
    """
    detector = GeometricDetector(min_area=min_area, max_area=max_area)
    
    detector.set_reference_objects(images)
    
    reference_cluster = detector.cluster_objects(similarity_threshold=0.6)
    
    if len(reference_cluster) == 0:
        print("警告: 未找到足够的同类几何物体")
        return [], panorama
    
    all_detections = []
    
    for img_idx, image in enumerate(images):
        detections = detector.detect_objects_in_image(
            image, 
            img_idx, 
            reference_cluster, 
            similarity_threshold
        )
        all_detections.extend(detections)
        print(f"在图片 {img_idx+1} 中检测到 {len(detections)} 个目标")
    
    annotated_panorama = None
    if panorama is not None:
        panorama_detections = detector.detect_objects_in_panorama(
            panorama, 
            reference_cluster, 
            similarity_threshold
        )
        
        annotated_panorama = panorama.copy()
        for det in panorama_detections:
            box = det['box']
            similarity = det['similarity']
            
            x, y, w, h = box
            cv2.rectangle(annotated_panorama, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            
            text = f"{similarity:.2f}"
            cv2.putText(annotated_panorama, text, (int(x), int(y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        print(f"在全景图中检测到 {len(panorama_detections)} 个目标")
    
    return all_detections, annotated_panorama
