import cv2
import numpy as np
from utils import convert_to_gray


class GeometricDetector:
    def __init__(self, min_area=100, max_area=50000):
        """
        初始化几何物体检测器
        :param min_area: 最小检测面积
        :param max_area: 最大检测面积
        """
        self.min_area = min_area
        self.max_area = max_area
        self.reference_objects = []
    
    def preprocess_image(self, image):
        """
        图像预处理：灰度化、高斯模糊、边缘检测
        """
        gray = convert_to_gray(image)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 30, 150)
        
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
        获取边界框 (x, y, w, h)
        """
        x, y, w, h = cv2.boundingRect(contour)
        return (int(x), int(y), int(w), int(h))
    
    def compute_shape_similarity(self, features1, features2):
        """
        计算形状相似度（基于Hu矩的欧氏距离）
        返回值范围：0~1，1表示完全相同
        """
        if features1 is None or features2 is None:
            return 0.0
        
        distance = np.linalg.norm(features1 - features2)
        
        similarity = np.exp(-distance / 3.0)
        
        return max(0.0, min(1.0, similarity))
    
    def compute_color_similarity(self, color1, color2):
        """
        计算颜色相似度
        返回值范围：0~1
        """
        if color1 is None or color2 is None:
            return 0.5
        
        distance = np.linalg.norm(color1 - color2)
        max_distance = np.sqrt(3 * (255 ** 2))
        
        similarity = 1.0 - (distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def compute_overall_similarity(self, shape_sim, color_sim, shape_weight=0.8, color_weight=0.2):
        """
        计算综合相似度
        """
        return shape_sim * shape_weight + color_sim * color_weight
    
    def extract_all_objects(self, images):
        """
        从所有图片中提取所有几何物体
        :param images: 图片列表
        :return: 物体列表，每个物体包含特征和来源信息
        """
        all_objects = []
        
        for img_idx, image in enumerate(images):
            binary = self.preprocess_image(image)
            contours = self.find_contours(binary)
            filtered = self.filter_contours(contours)
            
            for cnt_idx, cnt in enumerate(filtered):
                shape_features = self.get_shape_features(cnt)
                color_features = self.get_color_features(image, cnt)
                bbox = self.get_bounding_box(cnt)
                area = cv2.contourArea(cnt)
                
                obj = {
                    'shape_features': shape_features,
                    'color_features': color_features,
                    'bbox': bbox,
                    'area': area,
                    'source_img': img_idx,
                    'contour': cnt
                }
                all_objects.append(obj)
        
        print(f"从 {len(images)} 张图片中提取了 {len(all_objects)} 个几何物体")
        return all_objects
    
    def cluster_similar_objects(self, objects, similarity_threshold=0.6):
        """
        对物体进行聚类，找出同类几何物体
        :param objects: 物体列表
        :param similarity_threshold: 相似度阈值
        :return: 最大的聚类（同类物体）
        """
        if len(objects) == 0:
            return []
        
        clusters = []
        
        for i, obj in enumerate(objects):
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
        
        clusters.sort(key=lambda x: len(x), reverse=True)
        
        print(f"找到 {len(clusters)} 个聚类")
        for i, cluster in enumerate(clusters[:3]):
            print(f"  聚类 {i+1}: {len(cluster)} 个物体")
        
        if clusters:
            return clusters[0]
        return []
    
    def detect_objects_in_single_image(self, image, source_img_idx, reference_objects, similarity_threshold=0.4):
        """
        在单张图片中检测与参考物体相似的几何物体
        :param image: 输入图像
        :param source_img_idx: 来源图片编号（0开始）
        :param reference_objects: 参考物体列表
        :param similarity_threshold: 相似度阈值
        :return: 检测结果列表，每个结果包含: box, similarity, source_img
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
                    'source_img': int(source_img_idx),
                    'contour_area': float(cv2.contourArea(cnt))
                }
                detections.append(detection)
        
        return detections
    
    def annotate_image_with_detections(self, image, detections):
        """
        在图像上用红色矩形框标注检测结果，并标注相似度
        :param image: 输入图像
        :param detections: 检测结果列表
        :return: 标注后的图像
        """
        annotated = image.copy()
        
        for det in detections:
            box = det['box']
            similarity = det['similarity']
            
            x, y, w, h = box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            text = f"{similarity:.2f}"
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(annotated, text, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated


def detect_all_images(images, similarity_threshold=0.4, min_area=100, max_area=50000):
    """
    检测所有图片中的同类几何物体
    :param images: 原始图片列表（按顺序）
    :param similarity_threshold: 相似度阈值
    :param min_area: 最小检测面积
    :param max_area: 最大检测面积
    :return: (所有检测结果列表, 标注后的图片列表)
        检测结果格式: {'box': (x,y,w,h), 'similarity': 0.85, 'source_img': 0}
    """
    detector = GeometricDetector(min_area=min_area, max_area=max_area)
    
    print("\n[步骤1] 提取所有图片中的几何物体...")
    all_objects = detector.extract_all_objects(images)
    
    if len(all_objects) == 0:
        print("警告: 未检测到任何几何物体")
        return [], []
    
    print("\n[步骤2] 聚类找出同类几何物体...")
    reference_cluster = detector.cluster_similar_objects(all_objects, similarity_threshold=0.6)
    
    if len(reference_cluster) == 0:
        print("警告: 未找到同类几何物体")
        return [], []
    
    print(f"\n[步骤3] 以最大聚类（{len(reference_cluster)}个物体）为参考，检测所有图片中的同类物体...")
    
    all_detections = []
    annotated_images = []
    
    for img_idx, image in enumerate(images):
        detections = detector.detect_objects_in_single_image(
            image, 
            img_idx, 
            reference_cluster, 
            similarity_threshold
        )
        
        annotated = detector.annotate_image_with_detections(image, detections)
        annotated_images.append(annotated)
        
        all_detections.extend(detections)
        print(f"  图片 {img_idx+1}: 检测到 {len(detections)} 个同类物体")
    
    return all_detections, annotated_images
