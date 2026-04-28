import os
import sys

from utils import (
    load_images_from_paths,
    save_image,
    save_detections_to_json
)
from stitching import create_panorama
from detection import detect_and_annotate


def main():
    """
    主函数：实现图片拼接、物体检测和结果输出
    """
    print("=" * 60)
    print("图片全景拼接与几何物体检测系统")
    print("=" * 60)
    
    image_paths = [
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "5.png"
    ]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = [os.path.join(base_dir, path) for path in image_paths]
    
    print("\n[步骤1] 加载图片...")
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"错误: 图片不存在: {path}")
            sys.exit(1)
        print(f"  图片 {i+1}: {os.path.basename(path)}")
    
    images = load_images_from_paths(image_paths)
    print(f"成功加载 {len(images)} 张图片")
    
    print("\n[步骤2] 进行图片拼接...")
    print("  使用SIFT特征检测器进行横向拼接...")
    
    try:
        panorama = create_panorama(images, detector_type='SIFT', remove_borders=True)
        
        if panorama is None:
            print("  警告: SIFT拼接失败，尝试使用ORB...")
            panorama = create_panorama(images, detector_type='ORB', remove_borders=True)
        
        if panorama is None:
            print("错误: 图片拼接失败")
            sys.exit(1)
        
        print("  图片拼接成功!")
        
        panorama_path = os.path.join(base_dir, "panorama.jpg")
        save_image(panorama_path, panorama)
        print(f"  全景图已保存: {panorama_path}")
        
    except Exception as e:
        print(f"错误: 拼接过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n[步骤3] 检测同类几何物体...")
    print("  分析所有图片中的几何物体并聚类...")
    
    try:
        all_detections, annotated_panorama = detect_and_annotate(
            images=images,
            panorama=panorama,
            similarity_threshold=0.5,
            min_area=200,
            max_area=50000
        )
        
        if annotated_panorama is not None:
            annotated_path = os.path.join(base_dir, "panorama_annotated.jpg")
            save_image(annotated_path, annotated_panorama)
            print(f"  标注后的全景图已保存: {annotated_path}")
        
    except Exception as e:
        print(f"错误: 检测过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        all_detections = []
    
    print("\n[步骤4] 保存检测结果...")
    
    formatted_detections = []
    for i, det in enumerate(all_detections):
        formatted_det = {
            "id": i,
            "box": det["box"],
            "similarity": det["similarity"],
            "source_img": det["source_img"] + 1,
            "contour_area": det.get("contour_area", 0)
        }
        formatted_detections.append(formatted_det)
    
    json_path = os.path.join(base_dir, "1.json")
    save_detections_to_json(formatted_detections, json_path)
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  1. panorama.jpg - 拼接后的全景图")
    if os.path.exists(os.path.join(base_dir, "panorama_annotated.jpg")):
        print(f"  2. panorama_annotated.jpg - 标注后的全景图")
    print(f"  3. 1.json - 检测结果数据")
    
    print(f"\n检测统计:")
    print(f"  - 总检测数: {len(formatted_detections)}")
    
    source_counts = {}
    for det in formatted_detections:
        src = det["source_img"]
        source_counts[src] = source_counts.get(src, 0) + 1
    
    for src, count in sorted(source_counts.items()):
        print(f"  - 图片 {src}: {count} 个目标")
    
    if formatted_detections:
        similarities = [d["similarity"] for d in formatted_detections]
        print(f"\n相似度统计:")
        print(f"  - 最高相似度: {max(similarities):.4f}")
        print(f"  - 最低相似度: {min(similarities):.4f}")
        print(f"  - 平均相似度: {sum(similarities)/len(similarities):.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
