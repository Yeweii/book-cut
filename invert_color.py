import cv2
import numpy as np
import os
import shutil
from pathlib import Path


def detect_inverted(img_path, threshold_edge=50, threshold_center=200) -> bool:
    """
    检测图片是否为黑底白字（反色状态）

    判断依据：边缘区域（背景）偏黑 + 中心区域（文字）偏白 → 需要反色

    Args:
        img_path: 图片路径（支持中文）
        threshold_edge: 边缘平均灰度阈值，低于此值认为背景是黑的
        threshold_center: 中心平均灰度阈值，高于此值认为前景是白的

    Returns:
        True=需要反色，False=正常白底黑字
    """
    try:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False

        h, w = img.shape

        # 边缘区域（外圈10%）
        edge = np.concatenate([
            img[:int(h * 0.1), :].flatten(),
            img[-int(h * 0.1):, :].flatten(),
            img[:, :int(w * 0.1)].flatten(),
            img[:, -int(w * 0.1):].flatten(),
        ])
        edge_mean = edge.mean()

        # 中心区域（中间60%）
        cy1, cy2 = int(h * 0.2), int(h * 0.8)
        cx1, cx2 = int(w * 0.2), int(w * 0.8)
        center_mean = img[cy1:cy2, cx1:cx2].mean()

        return edge_mean < threshold_edge and center_mean > threshold_center

    except Exception:
        return False


def invert_image_color(input_path, output_path):
    """
    对单张图片进行黑白反色处理（兼容中文路径）
    """
    try:
        # 1. 读取图片（支持中文路径）
        img = cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"❌ 读取失败，请检查路径: {input_path}")
            return False

        # 2. 核心反色逻辑：位取反 (0变255, 255变0)
        # 这行代码等同于 255 - img，但 cv2.bitwise_not 经过 C++ 底层优化，速度极快
        inverted_img = cv2.bitwise_not(img)

        # 3. 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 4. 保存图片（支持中文路径）
        ext = Path(output_path).suffix or '.png'
        cv2.imencode(ext, inverted_img)[1].tofile(str(output_path))

        print(f"✅ 成功反色: {Path(input_path).name} -> {Path(output_path).name}")
        return True

    except Exception as e:
        print(f"❌ 处理异常 {input_path}: {e}")
        return False


def batch_invert_images(input_dir, output_dir):
    """
    批量反色处理文件夹内的所有图片
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    if not input_dir.exists():
        print(f"找不到输入目录: {input_dir}")
        return

    # 找出所有图片
    img_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in exts]

    print(f"====== 开始批量黑白反色，共找到 {len(img_paths)} 张图片 ======")
    for img_path in img_paths:
        # 保持原有的目录结构
        relative_path = img_path.relative_to(input_dir)
        target_path = output_dir / relative_path

        invert_image_color(img_path, target_path)

    print("====== 处理完成！ ======")


def batch_invert_if_needed(input_dir, output_dir):
    """
    批量处理：仅对黑底白字的图片进行反色，正常图片直接复制

    Args:
        input_dir: 输入图片目录
        output_dir: 输出图片目录
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    if not input_dir.exists():
        print(f"找不到输入目录: {input_dir}")
        return

    img_paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in exts]

    print(f"====== 开始检测并处理反色，共找到 {len(img_paths)} 张图片 ======")
    inverted_count = 0
    copied_count = 0

    for img_path in img_paths:
        relative_path = img_path.relative_to(input_dir)
        target_path = output_dir / relative_path

        if detect_inverted(img_path):
            invert_image_color(img_path, target_path)
            inverted_count += 1
        else:
            # 正常图片直接复制
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, target_path)
            copied_count += 1

    print(f"====== 处理完成！反色: {inverted_count} | 复制: {copied_count} ======")


if __name__ == "__main__":
    # ==========================================
    # 用法一：处理整个文件夹（批量）
    # ==========================================
    INPUT_FOLDER = "./output_final/小檀栅室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本"
    OUTPUT_FOLDER = "./output_final_invert/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版"  # 反色后的输出目录

    batch_invert_images(INPUT_FOLDER, OUTPUT_FOLDER)

    # ==========================================
    # 用法二：只处理单张图片（取消下方注释即可使用）
    # ==========================================
    # single_input = "测试图片.png"
    # single_output = "测试图片_反色.png"
    # invert_image_color(single_input, single_output)