# -*- coding: utf-8 -*-
"""
古籍扫描件跨页拆分工具

支持两种输入：
  - 跨页图像（左右两个页面拼在一起）
  - 单页图像（直接输出，不做处理）

输出：左图序号大于右图序号（符合书本阅读顺序）
"""

import cv2
import numpy as np
import os
from pathlib import Path


def find_gutter_center(img_gray, margin=0.3):
    """
    通过水平投影法找书缝中线

    算法：
    1. Sobel 竖向梯度，强调竖线（文字、鱼尾）
    2. 沿 X 轴投影，得到每列梯度密度
    3. 在中间区域（margin~1-margin）找密度最低点
    4. 如果结果偏离中心超过20%，使用中心位置

    Args:
        img_gray: 灰度图
        margin: 左右检测区域边距比例

    Returns:
        gutter 中心 x 坐标，检测失败返回 None
    """
    h, w = img_gray.shape

    # Sobel 竖向梯度
    grad = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = np.abs(grad)

    # 沿 X 轴投影
    proj = grad.sum(axis=0)

    # 取中间区域
    x_start = int(w * margin)
    x_end = int(w * (1 - margin))
    mid_proj = proj[x_start:x_end]

    if mid_proj.size == 0:
        return None

    # 密度最低点即为书缝
    gutter_local = np.argmin(mid_proj)
    gutter_x = x_start + gutter_local

    # 如果书缝位置偏离中心超过20%，使用图像中心
    center_x = w // 2
    if abs(gutter_x - center_x) >= w * 0.2:
        gutter_x = center_x

    return gutter_x


def find_gutter_by_darkstrip(img_gray, margin=0.25):
    """
    通过检测中间暗色条带找书缝

    适用于书缝处有明显黑线的古籍扫描件

    Args:
        img_gray: 灰度图
        margin: 检测区域边距

    Returns:
        书缝 x 坐标，检测失败返回 None
    """
    h, w = img_gray.shape

    # 灰度反转：黑→白，背景变深
    inv = 255 - img_gray

    # 中间区域行均值
    x_start = int(w * margin)
    x_end = int(w * (1 - margin))
    strip = img_gray[:, x_start:x_end]
    row_mean = strip.mean(axis=1)

    # 上下边缘去边框影响
    v_margin = int(h * 0.1)
    mid = row_mean[v_margin:-v_margin]

    # 在垂直方向找暗区中心（书缝是竖直的，所以看列方向的暗带）
    col_sum = img_gray[:, x_start:x_end].sum(axis=0)
    col_inv = (255 - col_sum).astype(np.float32)

    # 高斯平滑
    col_smooth = cv2.GaussianBlur(col_inv.reshape(1, -1), (1, 51), 0).flatten()
    gutter_local = np.argmin(col_smooth)
    gutter_x = x_start + gutter_local

    return gutter_x


def find_frame_edges(img_gray):
    """
    通过检测左右两侧边框线位置，找双页之间的书缝

    适用于两边都有框线的古籍扫描件

    算法：
    1. 在左半部分找最右侧的垂直边框线（内容右边界）
    2. 在右半部分找最左侧的垂直边框线（内容左边界）
    3. 书缝位置 = (左边界_x + 右边界_x) / 2

    Args:
        img_gray: 灰度图

    Returns:
        书缝 x 坐标，检测失败返回 None
    """
    h, w = img_gray.shape

    # 统计每列的暗色像素比例
    dark_ratio_per_col = (img_gray < 100).sum(axis=0) / h

    # 左半部分：从右向左扫描，找内容结束的位置（暗色比例突然升高的点）
    left_region = dark_ratio_per_col[:w // 2]
    right_edge = w // 2 - 1

    # 从中间向左扫描，找暗色比例突然升高的位置
    for x in range(w // 2 - 1, max(0, w // 4), -1):
        if dark_ratio_per_col[x] > 0.05:  # 有内容
            # 继续向左找边框（暗色比例接近0的位置）
            for bx in range(x, max(0, x - w // 10), -1):
                if dark_ratio_per_col[bx] < 0.02:
                    right_edge = bx
                    break
            break

    # 右半部分：从左向右扫描，找内容开始的位置（暗色比例突然升高的点）
    left_edge = w // 2

    for x in range(w // 2, min(w - 1, w * 3 // 4)):
        if dark_ratio_per_col[x] > 0.05:  # 有内容
            # 继续向右找边框（暗色比例接近0的位置）
            for bx in range(x, min(w - 1, x + w // 10)):
                if dark_ratio_per_col[bx] < 0.02:
                    left_edge = bx
                    break
            break

    # 计算中缝位置
    gutter_x = (right_edge + left_edge) // 2

    # 如果检测结果不合理，使用中心位置
    center_x = w // 2
    if gutter_x < w * 0.3 or gutter_x > w * 0.7:
        gutter_x = center_x

    return gutter_x


def split_image(image_path, output_dir, base_num=None, split_mode="gradient"):
    """
    将跨页图像拆分为两张独立图片

    自动判断是否为跨页（通过检测中间书缝），是单页则直接输出

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        base_num: 左图起始序号（int），右图序号 = base_num - 1
                  为 None 时自动使用图像文件名中的数字
        split_mode: 拆分模式
            - "gradient": 梯度法（默认），适用于普通跨页
            - "frame": 双框检测法，适用于两边都有边框的古籍

    Returns:
        (left_path, right_path) 输出的两张图片路径；
        单页时返回 (single_path, None)
    """
    path = Path(image_path)
    if not path.exists():
        print(f"错误：找不到文件 '{image_path}'")
        return None, None

    file_data = np.fromfile(str(path), dtype=np.uint8)
    if file_data.size == 0:
        print(f"错误：文件为空 '{image_path}'")
        return None, None

    img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 '{image_path}'")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 根据 split_mode 选择书缝检测方法
    if split_mode == "frame":
        gutter = find_frame_edges(gray)
    else:
        # 尝试多种方法找书缝
        gutter = find_gutter_center(gray)
        if gutter is None:
            gutter = find_gutter_by_darkstrip(gray)

    # 校验：书缝应该在中间区域（20%~80%）
    if gutter is not None and (gutter < w * 0.2 or gutter > w * 0.8):
        gutter = None

    # 单页：未检测到有效书缝
    if gutter is None:
        # 输出原图
        out_path = Path(output_dir) / path.name
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cv2.imencode(path.suffix or ".png", img)[1].tofile(str(out_path))
        print(f"单页图像，直接输出：{out_path.name}")
        return str(out_path), None

    # 书缝处过渡点：向左/右找到灰度跳变（文字开始的位置）
    # 如果有明显的书缝线，才做灰度跳变扫描
    # 否则直接用图像中心切割
    x_left = gutter
    x_right = gutter

    # 尝试向左扫描找灰度跳变
    for x in range(gutter - 1, max(0, gutter - w // 6), -1):
        if abs(int(gray[h // 2, x]) - int(gray[h // 2, x + 1])) > 30:
            x_left = x
            break

    # 尝试向右扫描找灰度跳变
    for x in range(gutter + 1, min(w - 1, gutter + w // 6)):
        if abs(int(gray[h // 2, x]) - int(gray[h // 2, x - 1])) > 30:
            x_right = x
            break

    # 如果跳变点距离中心太近，说明没有明显书缝线，直接用中心切割
    if abs(x_left - gutter) < w * 0.05 or abs(x_right - gutter) < w * 0.05:
        x_left = gutter
        x_right = gutter

    # 切割
    left_img = img[:, :x_left]
    right_img = img[:, x_right:]

    # 注意：拆分后不再调用 trim_border，避免伤及文字
    # 如果需要裁剪白边，应在后续的 ancient_book_engine 处理中完成

    # 检查切割结果是否有效
    if left_img is None or left_img.size == 0 or right_img is None or right_img.size == 0:
        print(f"警告：切割结果为空，跳过：{path.name}")
        return None, None

    # 序号：左 > 右（符合书本阅读顺序）
    # 左图 = verso（右页），右图 = recto（左页）
    if base_num is None:
        base_num = 1
    num_left = base_num
    num_right = base_num - 1

    left_name = f"{num_left:04d}.png"
    right_name = f"{num_right:04d}.png"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    left_path = out_dir / left_name
    right_path = out_dir / right_name

    # 编码并写入
    left_encoded = cv2.imencode(".png", left_img)
    right_encoded = cv2.imencode(".png", right_img)

    if left_encoded is None or right_encoded is None:
        print(f"警告：编码失败，跳过：{path.name}")
        return None, None

    left_encoded[1].tofile(str(left_path))
    right_encoded[1].tofile(str(right_path))

    print(f"拆分：{left_name} | {right_name}（书缝 x={gutter}）")

    return str(left_path), str(right_path)


def trim_border(img):
    """
    裁剪图像四周的多余空白或纯色边框

    Args:
        img: 输入图像（彩色或灰度）

    Returns:
        裁剪后的图像
    """
    if img is None or img.size == 0:
        return img

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 上下扫描：跳过接近背景色的行
    threshold = 10
    bg_color = np.median(gray.ravel())

    # 上边
    y_top = 0
    for y in range(0, h // 3):
        if abs(int(gray[y, w // 2]) - int(bg_color)) > threshold:
            y_top = y
            break

    # 下边
    y_bottom = h
    for y in range(h - 1, h * 2 // 3, -1):
        if abs(int(gray[y, w // 2]) - int(bg_color)) > threshold:
            y_bottom = y + 1
            break

    # 左右扫描
    x_left = 0
    for x in range(0, w // 3):
        if abs(int(gray[h // 2, x]) - int(bg_color)) > threshold:
            x_left = x
            break

    x_right = w
    for x in range(w - 1, w * 2 // 3, -1):
        if abs(int(gray[h // 2, x]) - int(bg_color)) > threshold:
            x_right = x + 1
            break

    # 确保不切到内容
    y_top = min(y_top, h // 10)
    x_left = min(x_left, w // 10)
    y_bottom = max(y_bottom, h - h // 10)
    x_right = max(x_right, w - w // 10)

    return img[y_top:y_bottom, x_left:x_right]


def batch_split(input_dir, output_dir, base_num=None, split_mode="gradient"):
    """
    批量拆分目录中的跨页图像

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        base_num: 左图起始序号，为 None 则按文件名数字递增
        split_mode: 拆分模式，"gradient" 或 "frame"
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    img_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in exts
    ])

    if not img_paths:
        print(f"未找到图片：{input_dir}")
        return

    current_num = base_num if base_num is not None else 1

    for img_path in img_paths:
        left, right = split_image(str(img_path), str(output_dir), current_num, split_mode)
        if left and right:
            # 双页拆分：左右各占一个编号
            current_num += 2
        elif left:
            # 单页：占一个编号
            current_num += 1


if __name__ == "__main__":
    # 示例：处理单个文件
    split_image(
        "./input/跨页图像.png",
        "./output",
        base_num=1,
    )

    # 示例：批量处理
    # batch_split("./input/", "./output", base_num=1)
