import os
import sys

from utils import (
    load_images_from_paths,
    save_image,
    save_detections_to_json
)
from stitching import create_panorama
from detection import detect_all_images


def main():
    """
    主函数：
    1. 加载5张原图（1.png ~ 5.png，从左到右顺序）
    2. 检测每张原图中的同类几何物体（可能不完整）
    3. 用红色矩形框选并标注0~1相似度
    4. 收集目标坐标、相似度、来源图片编号等信息存入1.json
    5. 横向拼接生成panorama.jpg
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
    
    print("\n[步骤1] 加载原图...")
    for i, path in enumerate(image_paths):
        if not os.path.exists(path):
            print(f"错误: 图片不存在: {path}")
            sys.exit(1)
        print(f"  图片 {i+1}: {os.path.basename(path)}")
    
    images = load_images_from_paths(image_paths)
    print(f"成功加载 {len(images)} 张图片")
    
    print("\n[步骤2] 检测原图中的同类几何物体...")
    print("  说明：从所有原图中提取几何物体，聚类找出同类物体，")
    print("        然后在每张原图中检测同类物体（包括边缘处可能不完整的物体）")
    
    try:
        all_detections, annotated_images = detect_all_images(
            images=images,
            similarity_threshold=0.4,
            min_area=100,
            max_area=50000
        )
        
    except Exception as e:
        print(f"错误: 检测过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        all_detections = []
        annotated_images = []
    
    print("\n[步骤3] 保存检测结果到1.json...")
    
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
    
    print("\n[步骤4] 保存标注后的图片（每张原图带红色矩形框和相似度）...")
    for i, annotated in enumerate(annotated_images):
        annotated_path = os.path.join(base_dir, f"annotated_{i+1}.jpg")
        save_image(annotated_path, annotated)
        print(f"  已保存: annotated_{i+1}.jpg")
    
    print("\n[步骤5] 横向拼接生成全景图panorama.jpg...")
    
    try:
        panorama = create_panorama(images, method='simple')
        
        if panorama is None:
            print("错误: 图片拼接失败")
        else:
            panorama_path = os.path.join(base_dir, "panorama.jpg")
            save_image(panorama_path, panorama)
            print(f"  全景图已保存: panorama.jpg")
        
    except Exception as e:
        print(f"错误: 拼接过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  1. panorama.jpg - 横向拼接后的全景图")
    print(f"  2. 1.json - 检测结果数据（坐标、相似度、来源图片编号等）")
    print(f"  3. annotated_1.jpg ~ annotated_5.jpg - 标注后的原图（红色框+相似度）")
    
    print(f"\n检测统计:")
    print(f"  - 总检测数: {len(formatted_detections)} 个同类几何物体")
    
    source_counts = {}
    for det in formatted_detections:
        src = det["source_img"]
        source_counts[src] = source_counts.get(src, 0) + 1
    
    for src in range(1, 6):
        count = source_counts.get(src, 0)
        print(f"  - 图片 {src}: {count} 个目标")
    
    if formatted_detections:
        similarities = [d["similarity"] for d in formatted_detections]
        print(f"\n相似度统计 (0~1，1表示最相似):")
        print(f"  - 最高相似度: {max(similarities):.4f}")
        print(f"  - 最低相似度: {min(similarities):.4f}")
        print(f"  - 平均相似度: {sum(similarities)/len(similarities):.4f}")
    
    print(f"\n1.json 数据格式说明:")
    print(f"  {{")
    print(f"    'id': 检测编号,")
    print(f"    'box': [x, y, width, height],  // 矩形框左上角坐标和宽高")
    print(f"    'similarity': 0.85,  // 相似度 0~1")
    print(f"    'source_img': 1,  // 来源图片编号 (1~5)")
    print(f"    'contour_area': 1500.0  // 轮廓面积")
    print(f"  }}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
