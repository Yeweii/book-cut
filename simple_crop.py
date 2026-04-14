# -*- coding: utf-8 -*-
"""
简版白边裁剪器

仅检测并裁剪四边空白区域，不做版心提取。
适用于普通书籍扫描件、无框线情形的简单裁剪需求。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class SimpleCropper:
    """
    简版白边裁剪器

    仅检测并裁剪四边空白区域，不做版心提取。
    复用 AncientBookCropTester 的边缘检测逻辑。
    """

    def __init__(
        self,
        blank_threshold: float = 240,
        content_density_min: float = 0.005,
        edge_ignore_ratio: float = 0.02,
        max_scan_ratio: float = 0.30,
        padding_ratio: float = 0.02,
        padding_min_pixel: int = 25,
    ):
        """
        初始化简版白边裁剪器

        Args:
            blank_threshold: 边缘判定为空白的灰度阈值（均值 > 此值为空白）
            content_density_min: 内容密度阈值（低于此值视为空白）
            edge_ignore_ratio: 边缘安全忽略比例（跳过扫描仪黑边）
            max_scan_ratio: 最大扫描范围比例
            padding_ratio: 裁剪后保留的空白比例
            padding_min_pixel: 空白保底像素值
        """
        self.blank_threshold = blank_threshold
        self.content_density_min = content_density_min
        self.edge_ignore_ratio = edge_ignore_ratio
        self.max_scan_ratio = max_scan_ratio
        self.padding_ratio = padding_ratio
        self.padding_min_pixel = padding_min_pixel

    def detect_frame_border(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测图像是否有外框线

        算法：
        1. 将图像四边各取 5% 宽度/高度作为边缘检测区
        2. 统计边缘区的暗色像素（< 100）比例
        3. 如果上/下/左/右四边都有 > 5% 暗色像素 → 判定为有框线

        Args:
            gray: 灰度图

        Returns:
            (x_left, y_top, x_right, y_bottom) 框线边界坐标，检测失败返回 None
        """
        h, w = gray.shape[:2]

        # 边缘检测区宽度（图像尺寸的 5%）
        edge_width = max(3, int(min(w, h) * 0.05))

        # 统计各边的暗色像素比例
        dark_threshold = 100

        # 上边
        top_strip = gray[:edge_width, :]
        top_dark_ratio = np.sum(top_strip < dark_threshold) / top_strip.size

        # 下边
        bottom_strip = gray[-edge_width:, :]
        bottom_dark_ratio = np.sum(bottom_strip < dark_threshold) / bottom_strip.size

        # 左边
        left_strip = gray[:, :edge_width]
        left_dark_ratio = np.sum(left_strip < dark_threshold) / left_strip.size

        # 右边
        right_strip = gray[:, -edge_width:]
        right_dark_ratio = np.sum(right_strip < dark_threshold) / right_strip.size

        # 判断是否有框线（四边都有 > 5% 暗色像素）
        frame_threshold = 0.05
        has_frame = (
            top_dark_ratio > frame_threshold and
            bottom_dark_ratio > frame_threshold and
            left_dark_ratio > frame_threshold and
            right_dark_ratio > frame_threshold
        )

        if not has_frame:
            return None

        # 找各边的框线精确位置
        # 上边：从上向下扫描，找暗色比例突然升高的位置
        y_top = 0
        for y in range(0, min(h // 3, edge_width * 3)):
            strip = gray[y:y + edge_width, :]
            if np.sum(strip < dark_threshold) / strip.size > frame_threshold:
                y_top = y
                break

        # 下边：从下向上扫描
        y_bottom = h
        for y in range(h - 1, max(h * 2 // 3, h - edge_width * 3), -1):
            strip = gray[y - edge_width:y, :]
            if np.sum(strip < dark_threshold) / strip.size > frame_threshold:
                y_bottom = y
                break

        # 左边：从左向右扫描
        x_left = 0
        for x in range(0, min(w // 3, edge_width * 3)):
            strip = gray[:, x:x + edge_width]
            if np.sum(strip < dark_threshold) / strip.size > frame_threshold:
                x_left = x
                break

        # 右边：从右向左扫描
        x_right = w
        for x in range(w - 1, max(w * 2 // 3, w - edge_width * 3), -1):
            strip = gray[:, x - edge_width:x]
            if np.sum(strip < dark_threshold) / strip.size > frame_threshold:
                x_right = x
                break

        return (x_left, y_top, x_right, y_bottom)

    def crop_by_frame(self, img: np.ndarray, frame_bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """
        根据框线边界裁剪图像

        在框线外侧保留少量空白后裁切

        Args:
            img: 输入图像
            frame_bounds: (x_left, y_top, x_right, y_bottom) 框线边界

        Returns:
            裁剪后的图像
        """
        if img is None or img.size == 0:
            return img

        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        x_left, y_top, x_right, y_bottom = frame_bounds

        # 计算 padding（保留少量空白）
        pad_x = max(self.padding_min_pixel, int((x_right - x_left) * self.padding_ratio))
        pad_y = max(self.padding_min_pixel, int((y_bottom - y_top) * self.padding_ratio))

        # 框线外侧留白（向边缘方向扩展）
        x1 = max(0, min(x_left - pad_x, w - 1))
        y1 = max(0, min(y_top - pad_y, h - 1))
        x2 = max(x1 + 1, min(x_right + pad_x, w))
        y2 = max(y1 + 1, min(y_bottom + pad_y, h))

        return img[y1:y2, x1:x2]

    def crop(self, img: np.ndarray) -> np.ndarray:
        """
        检测四边空白区域并裁剪

        自动识别两种模式：
        - 有框线时：在框线外侧裁切
        - 无框线时：在内容边界裁切 + padding

        Args:
            img: 输入图像（彩色或灰度）

        Returns:
            裁剪后的图像
        """
        if img is None or img.size == 0:
            return img

        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 自动检测是否有框线
        frame_bounds = self.detect_frame_border(gray)

        if frame_bounds is not None:
            # 有框线：在框线外侧裁切
            return self.crop_by_frame(img, frame_bounds)

        # 无框线：使用原有的内容检测法
        x1, y1, x2, y2 = self._get_crop_bounds(gray)

        # 计算 padding（保留一小部分空白防止页面过满）
        pad_x = max(self.padding_min_pixel, int((x2 - x1) * self.padding_ratio))
        pad_y = max(self.padding_min_pixel, int((y2 - y1) * self.padding_ratio))

        # 确保边界有效（加上 padding）
        x1 = max(0, min(x1 - pad_x, w - 1))
        y1 = max(0, min(y1 - pad_y, h - 1))
        x2 = max(x1 + 1, min(x2 + pad_x, w))
        y2 = max(y1 + 1, min(y2 + pad_y, h))

        return img[y1:y2, x1:x2]

    def _get_crop_bounds(self, gray: np.ndarray) -> Tuple[int, int, int, int]:
        """
        返回 (x1, y1, x2, y2) 裁剪边界

        算法：
        1. 灰度化图像
        2. 从四边向内扫描，使用区块密度判断
        3. 跳过 edge_ignore_ratio 安全区
        4. 找到内容起始位置后停止
        """
        h, w = gray.shape[:2]

        # 计算需要强行跳过的外围物理像素数
        ignore_x = max(1, int(w * self.edge_ignore_ratio))
        ignore_y = max(1, int(h * self.edge_ignore_ratio))

        # 最大扫描像素数
        max_strip_px = max(1, int(h * self.max_scan_ratio))
        step_px = max(3, int(min(h, w) * 0.01))  # 步长为 1% 或至少 3 像素

        # 上边界
        y_top = ignore_y
        for y in range(ignore_y, min(max_strip_px, h), step_px):
            strip = gray[y:min(y + step_px, h), :]
            if self._has_content(strip):
                y_top = y
                break

        # 下边界
        y_bottom = h - ignore_y
        for y in range(h - ignore_y - step_px, max(h - max_strip_px, -1), -step_px):
            strip = gray[max(0, y):min(y + step_px, h), :]
            if self._has_content(strip):
                y_bottom = y + step_px
                break

        # 左边界
        x_left = ignore_x
        for x in range(ignore_x, min(max_strip_px, w), step_px):
            strip = gray[:, x:min(x + step_px, w)]
            if self._has_content(strip):
                x_left = x
                break

        # 右边界
        x_right = w - ignore_x
        for x in range(w - ignore_x - step_px, max(w - max_strip_px, -1), -step_px):
            strip = gray[:, max(0, x):min(x + step_px, w)]
            if self._has_content(strip):
                x_right = x + step_px
                break

        return (x_left, y_top, x_right, y_bottom)

    def _has_content(self, strip: np.ndarray) -> bool:
        """
        判断一个区块是否包含内容（过滤噪点）

        Args:
            strip: 图像区块

        Returns:
            True=有内容，False=空白
        """
        # 白底黑字：统计暗色像素（<200 认为是墨迹）
        ink_pixels = np.sum(strip < 200)
        density = ink_pixels / strip.size if strip.size > 0 else 0
        return density >= self.content_density_min


class SimpleCropEngine:
    """
    简版白边裁剪引擎（批量处理）
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_width: int = 0,
        target_height: int = 0,
        compress_format: str = "png",
        jpeg_quality: int = 85,
        max_workers: int = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        初始化简版裁剪引擎

        Args:
            input_dir: 输入图片目录
            output_dir: 输出图片目录
            target_width: 缩放目标宽度（0=不缩放）
            target_height: 缩放目标高度（0=按比例）
            compress_format: 输出格式 "png" | "jpg"
            jpeg_quality: JPG 质量
            max_workers: 最大并发线程数
            verbose: 是否打印详细日志
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_width = max(0, int(target_width))
        self.target_height = max(0, int(target_height))
        self.compress_format = compress_format.lower()
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.max_workers = max(1, int(max_workers or 4))
        self.verbose = verbose

        self.cropper = SimpleCropper(**kwargs)

        self.exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    def process_directory(self, file_pattern: str = "*.png") -> dict:
        """
        批量处理输入目录下的所有匹配图像

        Args:
            file_pattern: 文件匹配模式

        Returns:
            处理统计信息
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        imgs = []
        for pattern in file_pattern.replace(' ', ';').split(';'):
            imgs.extend(sorted(self.input_dir.glob(pattern)))
            imgs.extend(sorted(self.input_dir.glob(pattern.upper())))
            imgs.extend(sorted(self.input_dir.glob(pattern.lower())))
        imgs = sorted(set(imgs))

        total = len(imgs)
        if total == 0:
            self._print(f"[警告] 输入目录中没有找到匹配的图像文件")
            return {'total': 0, 'success': 0, 'failed': 0}

        self._print(f"{'='*70}")
        self._print(f">>> 简版白边裁剪引擎 启动")
        self._print(f">>> 输入目录: {self.input_dir}")
        self._print(f">>> 输出目录: {self.output_dir}")
        self._print(f">>> 图像数量: {total}")
        self._print(f">>> 输出格式: {self.compress_format.upper()}")
        self._print(f"{'='*70}")

        start_time = time.time()
        success = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_single_image, img_path): img_path
                for img_path in imgs
            }

            for future in as_completed(future_to_path):
                result = future.result()
                self._print(result)
                if "[成功]" in result:
                    success += 1
                elif "[失败]" in result:
                    failed += 1

        elapsed = time.time() - start_time
        self._print(f"{'='*70}")
        self._print(f">>> 处理完成!")
        self._print(f">>> 总计: {total} | 成功: {success} | 失败: {failed}")
        self._print(f">>> 总耗时: {elapsed:.2f} 秒")
        self._print(f"{'='*70}")

        return {'total': total, 'success': success, 'failed': failed}

    def _process_single_image(self, img_path: Path) -> str:
        """处理单张图像"""
        try:
            # 读取图像
            data = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                return f"[失败] 无法读取图像: {img_path.name}"

            h_orig, w_orig = img.shape[:2]

            # 裁剪白边
            cropped = self.cropper.crop(img)

            # 缩放
            if self.target_width > 0:
                ratio = self.target_width / cropped.shape[1]
                target_h = int(cropped.shape[0] * ratio)
                target_size = (self.target_width, target_h)
            elif self.target_height > 0:
                ratio = self.target_height / cropped.shape[0]
                target_w = int(cropped.shape[1] * ratio)
                target_size = (target_w, self.target_height)
            else:
                target_size = None

            if target_size:
                resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
            else:
                resized = cropped.copy()

            # 输出
            output_suffix = f".{self.compress_format}"
            output_path = self.output_dir / img_path.with_suffix(output_suffix).name

            if self.compress_format == "png":
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

            cv2.imencode(output_suffix, resized, params)[1].tofile(str(output_path))

            return f"[成功] {img_path.name} -> {output_path.name}"

        except Exception as e:
            return f"[失败] {img_path.name} - {str(e)}"

    def _print(self, msg: str):
        if self.verbose:
            print(msg)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir + "_cropped"
    else:
        input_dir = "./input"
        output_dir = "./output_simple"

    engine = SimpleCropEngine(
        input_dir=input_dir,
        output_dir=output_dir,
        target_width=1200,
    )
    engine.process_directory("*.png")