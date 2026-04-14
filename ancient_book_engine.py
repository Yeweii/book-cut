# -*- coding: utf-8 -*-
"""
===============================================================
古籍图像智能裁剪与压缩引擎 (Ancient Book Intelligent Crop & Compress Engine)
===============================================================

版本: V2.0
功能: 高清古籍扫描件自动化裁剪、去除冗余白边/黑框，并进行极限压缩，
      特别适配墨水屏阅读器（Kindle KPW6）

设计原则:
    1. 文本安全优先 (Text Safety First): 宁可保留少量扫描黑影，绝不切断任何文字笔画
    2. 极限压缩比 (Extreme Compression): 榨干图像存储水分，保持文字边缘锐利
    3. 高鲁棒版式兼容 (Layout Robustness): 兼容纯竖排古籍、横排重印本、贴边文字

作者: AI Algorithm Engineer
日期: 2026-04-10
===============================================================
"""

import cv2
import numpy as np
import os
import gc
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, Any


from simple_crop import SimpleCropper


class AncientBookEngine:
    """
    古籍图像智能裁剪与压缩引擎

    该引擎实现以下核心功能:
    - 多线程并发处理 (ThreadPoolExecutor)
    - 智能版心提取 (smart/union/largest 三种模式)
    - 动态形态学焊接 (防止竖排文字被误判)
    - 文本保护与伪影过滤 (防误切机制)
    - 极限压缩 (PNG 1-bit / JPG 灰阶高压)

    Attributes:
        input_dir: 输入图像目录路径（支持中文路径）
        output_dir: 输出图像目录路径（支持中文路径）
        target_width: 统一缩放目标宽度（0表示不缩放）
        target_height: 统一缩放目标高度（0表示按比例自动计算）
        crop_mode: 版心提取模式 ("smart"/"union"/"largest")
        smart_gap_ratio: smart模式闭运算容差系数
        strict_artifact_filter: 伪影过滤安全阀（True=严格，False=宽松保文字）
        edge_tolerance_ratio: 边缘吸附防切容忍度（相对于图像宽度的比例）
        padding_ratio_w: 水平方向留白比例
        padding_ratio_h: 垂直方向留白比例
        padding_min_pixel: 留白保底像素值
        sharpness: USM锐化强度系数
        otsu_scale: 大津法阈值缩放系数
        compress_format: 输出格式 ("png"/"jpg")
        png_compression: PNG压缩级别 (1-9)
        jpeg_quality: JPG压缩质量 (1-100)
        max_workers: 最大并发线程数
        verbose: 是否输出详细日志
        book_type: 书籍类型 ("ancient"=古籍复杂版心提取，"simple"=普通书籍仅裁白边)
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_width: int = 1200,
        target_height: int = 0,
        # 版式与裁剪参数
        crop_mode: str = "smart",
        smart_gap_ratio: float = 0.015,
        # 文本安全参数
        strict_artifact_filter: bool = False,
        edge_tolerance_ratio: float = 0.05,
        # 留白与边缘保护
        padding_ratio_w: float = 0.02,
        padding_ratio_h: float = 0.02,
        padding_min_pixel: int = 25,
        # 图像增强
        sharpness: float = 1.2,
        otsu_scale: float = 1.0,
        # 投影法阈值参数
        projection_threshold: float = 0.15,
        blank_ratio_threshold: float = 0.05,
        # 压缩参数
        compress_format: str = "png",
        png_compression: int = 9,
        jpeg_quality: int = 85,
        # 性能参数
        max_workers: int = None,
        # 日志
        verbose: bool = True,
        # 书籍类型
        book_type: str = "ancient",
    ):
        """
        初始化古籍图像处理引擎

        Args:
            input_dir: 待处理图像所在目录路径
            output_dir: 处理后图像的输出目录路径
            target_width: 统一缩放目标宽度（像素），0表示不缩放
            target_height: 统一缩放目标高度，0表示按比例自动计算
            crop_mode: 版心提取模式
                - "smart": 智能分离模式，通过Y轴投影提取最大正文块
                - "union": 全局并集模式，提取所有文字的最小包围盒
                - "largest": 最大连通模式，仅提取最大连通域
            smart_gap_ratio: smart模式闭运算容差（相对于图像高度）
            strict_artifact_filter: True=严格过滤边缘伪影，False=宽松保文字
            edge_tolerance_ratio: 边缘防切容忍度（宽度比例）
            padding_ratio_w: 水平留白比例
            padding_ratio_h: 垂直留白比例
            padding_min_pixel: 留白保底像素
            sharpness: USM锐化强度（1.0=不锐化）
            otsu_scale: 大津法阈值缩放系数
            projection_threshold: 归一化投影阈值（判断是否为内容），取值 0.05～0.5
            blank_ratio_threshold: 原始黑色比例下限（排除纯空白），取值 0.01～0.2
            compress_format: 输出格式 ("png"/"jpg")
            png_compression: PNG压缩级别（1-9，9为最高）
            jpeg_quality: JPG质量（1-100）
            max_workers: 最大线程数，None则自动检测
            verbose: 是否打印详细日志
            book_type: 书籍类型 ("ancient"=古籍复杂版心提取，"simple"=普通书籍仅裁白边)
        """
        # ==================== 路径与目录 ====================
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # 确保输出目录存在
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # ==================== 缩放参数 ====================
        self.target_width = max(0, int(target_width))
        self.target_height = max(0, int(target_height))

        # ==================== 版式与裁剪参数 ====================
        self.crop_mode = crop_mode.lower()
        self.smart_gap_ratio = float(smart_gap_ratio)

        # 验证裁剪模式
        if self.crop_mode not in ("smart", "union", "largest"):
            raise ValueError(f"crop_mode 必须为 'smart'/'union'/'largest'，当前值: {crop_mode}")

        # ==================== 文本安全参数 ====================
        self.strict_artifact_filter = bool(strict_artifact_filter)
        self.edge_tolerance_ratio = float(edge_tolerance_ratio)

        # ==================== 留白参数 ====================
        self.padding_ratio_w = float(padding_ratio_w)
        self.padding_ratio_h = float(padding_ratio_h)
        self.padding_min_pixel = int(padding_min_pixel)

        # ==================== 图像增强参数 ====================
        self.sharpness = float(sharpness)
        self.otsu_scale = float(otsu_scale)

        # ==================== 投影法阈值参数 ====================
        self.projection_threshold = float(projection_threshold)
        self.blank_ratio_threshold = float(blank_ratio_threshold)

        # ==================== 压缩参数 ====================
        self.compress_format = compress_format.lower()
        if self.compress_format not in ("png", "jpg"):
            raise ValueError(f"compress_format 必须为 'png' 或 'jpg'，当前值: {compress_format}")

        self.png_compression = max(1, min(9, int(png_compression)))
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))

        # ==================== 书籍类型 ====================
        self.book_type = book_type.lower()
        if self.book_type not in ("ancient", "simple"):
            raise ValueError(f"book_type 必须为 'ancient' 或 'simple'，当前值: {book_type}")

        # simple 模式：初始化 SimpleCropper
        if self.book_type == "simple":
            self.simple_cropper = SimpleCropper(
                edge_ignore_ratio=self.edge_tolerance_ratio,
                padding_ratio=self.padding_ratio_w,
                padding_min_pixel=self.padding_min_pixel,
            )

        # ==================== 性能参数 ====================
        self.max_workers = max(1, int(max_workers or (os.cpu_count() or 1) + 2))
        self.verbose = bool(verbose)

        # ==================== 内部状态 ====================
        self.exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        self._print_lock = threading.Lock()

        # 统计信息
        self._stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
        }

    # ======================================================
    # 公共方法
    # ======================================================

    @classmethod
    def from_config(cls, config_path: str, **overrides) -> 'AncientBookEngine':
        """
        从 YAML 配置文件创建引擎实例

        Args:
            config_path: 配置文件路径（支持中文路径）
            **overrides: 可选的参数覆盖值

        Returns:
            AncientBookEngine 实例
        """
        import yaml

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 展平嵌套配置
        params = {}
        for key, value in config.items():
            if isinstance(value, dict):
                params.update(value)
            else:
                params[key] = value

        # 应用覆盖参数
        params.update(overrides)

        # 转换布尔值字符串
        bool_fields = ['strict_artifact_filter', 'verbose']
        for field in bool_fields:
            if field in params and isinstance(params[field], str):
                params[field] = params[field].lower() in ('true', '1', 'yes', 'on')

        return cls(**params)

    def process_directory(self, file_pattern: str = "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp") -> Dict[str, Any]:
        """
        批量处理输入目录下的所有匹配图像

        Args:
            file_pattern: 文件匹配模式（支持多模式，用分号分隔，如 "*.png;*.jpg"）

        Returns:
            包含处理统计信息的字典
        """
        # 收集待处理图像
        imgs = []
        for pattern in file_pattern.replace(' ', ';').split(';'):
            imgs.extend(sorted(self.input_dir.glob(pattern)))
            imgs.extend(sorted(self.input_dir.glob(pattern.upper())))
            imgs.extend(sorted(self.input_dir.glob(pattern.lower())))

        # 去重
        imgs = sorted(set(imgs))
        total = len(imgs)

        self._stats['total'] = total

        if total == 0:
            self._print("[警告] 输入目录中没有找到匹配的图像文件")
            return self._stats

        self._print(f"{'='*70}")
        self._print(f">>> 古籍图像智能裁剪与压缩引擎 启动")
        self._print(f">>> 输入目录: {self.input_dir}")
        self._print(f">>> 输出目录: {self.output_dir}")
        self._print(f">>> 匹配模式: {file_pattern}")
        self._print(f">>> 图像数量: {total}")
        self._print(f">>> 裁剪模式: {self.crop_mode.upper()}")
        self._print(f">>> 输出格式: {self.compress_format.upper()}")
        self._print(f">>> 最大线程: {self.max_workers}")
        self._print(f"{'='*70}")

        start_time = time.time()

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_single_image, img_path): img_path
                for img_path in imgs
            }

            for future in as_completed(future_to_path):
                result = future.result()
                self._print(result)

                if "[成功]" in result:
                    self._stats['success'] += 1
                elif "[失败]" in result:
                    self._stats['failed'] += 1
                elif "[跳过]" in result:
                    self._stats['skipped'] += 1

        # 强制垃圾回收释放内存
        gc.collect()

        elapsed = time.time() - start_time
        avg_time = elapsed / total if total > 0 else 0

        self._print(f"{'='*70}")
        self._print(f">>> 处理完成!")
        self._print(f">>> 总计: {total} | 成功: {self._stats['success']} | 失败: {self._stats['failed']} | 跳过: {self._stats['skipped']}")
        self._print(f">>> 总耗时: {elapsed:.2f} 秒 | 平均: {avg_time:.3f} 秒/张")
        self._print(f"{'='*70}")

        return self._stats

    def process_single(self, input_path: str, output_path: str = None) -> bool:
        """
        处理单张图像

        Args:
            input_path: 输入图像路径（支持中文路径）
            output_path: 输出路径，None则自动生成

        Returns:
            处理是否成功
        """
        input_p = Path(input_path)

        if not input_p.exists():
            self._print(f"[失败] 文件不存在: {input_path}")
            return False

        if output_path is None:
            output_p = self.output_dir / input_p.with_suffix(f".{self.compress_format}").name
        else:
            output_p = Path(output_path)

        output_p.parent.mkdir(parents=True, exist_ok=True)

        result = self._process_single_image(input_p)
        return "[成功]" in result

    # ======================================================
    # 核心处理流程（单张图像）
    # ======================================================

    def _process_single_image(self, img_path: Path) -> str:
        """
        处理单张图像的完整流程

        处理流程:
        1. 读取图像（支持中文路径）
        2. 大津法二值化（如需要）
        3. 版心提取
        4. 动态留白与边缘保护
        5. 尺寸缩放
        6. USM锐化增强
        7. 极限压缩输出
        8. 内存释放

        Args:
            img_path: 图像路径

        Returns:
            处理结果描述字符串
        """
        try:
            # ---------- 步骤1: 读取图像 ----------
            gray_img = self._read_image_safe(img_path)
            if gray_img is None:
                return f"[跳过] 无法读取图像: {img_path.name}"

            h_orig, w_orig = gray_img.shape[:2]

            # ---------- 步骤2: simple 模式：仅裁白边 ----------
            if self.book_type == "simple":
                cropped = self.simple_cropper.crop(gray_img)
                # simple 模式跳转到缩放步骤（不做版心提取和二阶段精裁）
            else:
                # ---------- 步骤2: 版心提取 ----------
                core_rect = self._extract_core_rect(gray_img)

                # ---------- 步骤3: 计算裁剪坐标 ----------
                if core_rect:
                    x, y, cw, ch = core_rect

                    # 计算动态留白
                    pad_x = max(self.padding_min_pixel, int(cw * self.padding_ratio_w))
                    pad_y = max(self.padding_min_pixel, int(ch * self.padding_ratio_h))

                    # 【修复V2.9】只往内容方向添加padding，不往外延伸
                    # 如果版心距边缘小于padding需求，只往内（内容方向）扩展
                    x1 = x - min(pad_x, x)  # 不往左超出边界
                    y1 = y - min(pad_y, y)  # 不往上超出边界
                    x2 = min(w_orig, x + cw + pad_x)
                    y2 = min(h_orig, y + ch + pad_y)

                    # ---------- 步骤4: 边缘吸附防切机制（二次判断） ----------
                    # 渐进扫描边缘区域（最多 20% 范围），找内容实际起始位置
                    # - 内容紧贴 tolerance 区内侧（<= tol）：保护，x1/y1 保持 padding 计算值
                    # - 内容在 tolerance 区外（> tol）：trim 到 content_start，不保留空白
                    max_scan_ratio = 0.20  # 最大扫描至 20% 范围
                    blank_threshold = 240   # 均值 > 240 判定为空白

                    def _edge_has_content(strip_mean: float) -> bool:
                        return strip_mean <= blank_threshold

                    def _find_content_boundary(start: int, end: int, step: int, axis: str) -> int:
                        """从 start 到 end 扫描，找内容起始位置；返回像素坐标；全空白返回 end"""
                        rng = range(start, end, step) if step > 0 else range(start, end, step)
                        for pos in rng:
                            strip = gray_img[:, pos] if axis == 'x' else gray_img[pos, :]
                            if _edge_has_content(strip.mean()):
                                return pos
                        return end

                    if x <= w_orig * self.edge_tolerance_ratio:
                        tol = int(w_orig * self.edge_tolerance_ratio)
                        scan_end = int(w_orig * max_scan_ratio)
                        content_x = _find_content_boundary(0, scan_end, 1, 'x')
                        if content_x <= tol:
                            x1 = 0  # 内容紧贴边缘，保护
                        else:
                            x1 = content_x  # 空白区在 tolerance 外，推进到内容起始

                    if (w_orig - (x + cw)) <= w_orig * self.edge_tolerance_ratio:
                        tol = int(w_orig * self.edge_tolerance_ratio)
                        scan_end = int(w_orig * (1 - max_scan_ratio))
                        content_x = _find_content_boundary(w_orig - 1, scan_end, -1, 'x')
                        if content_x >= w_orig - tol:
                            x2 = w_orig  # 内容紧贴边缘，保护
                        else:
                            x2 = content_x + 1  # 空白区在 tolerance 外，推进到内容起始

                    if y <= h_orig * self.edge_tolerance_ratio:
                        tol = int(h_orig * self.edge_tolerance_ratio)
                        scan_end = int(h_orig * max_scan_ratio)
                        content_y = _find_content_boundary(0, scan_end, 1, 'y')
                        if content_y <= tol:
                            y1 = 0  # 内容紧贴边缘，保护
                        else:
                            y1 = content_y  # 空白区在 tolerance 外，推进到内容起始

                    if (h_orig - (y + ch)) <= h_orig * self.edge_tolerance_ratio:
                        tol = int(h_orig * self.edge_tolerance_ratio)
                        scan_end = int(h_orig * (1 - max_scan_ratio))
                        content_y = _find_content_boundary(h_orig - 1, scan_end, -1, 'y')
                        if content_y >= h_orig - tol:
                            y2 = h_orig  # 内容紧贴边缘，保护
                        else:
                            y2 = content_y + 1  # 空白区在 tolerance 外，推进到内容起始

                    cropped = gray_img[y1:y2, x1:x2]
                else:
                    # 无法提取版心时保留原图
                    cropped = gray_img

            # ---------- 步骤4.5: 二阶段精裁 ----------
            # 在裁剪后图像上检测文字框实际边界，去除框内外大段空白
            # _detect_inner_content_rect 内部已包含 padding，与 Tester 逻辑对齐
            # simple 模式不做二阶段精裁
            if self.book_type != "simple":
                inner_rect = self._detect_inner_content_rect(cropped)
                if inner_rect:
                    ix, iy, iw, ih = inner_rect
                    cropped = cropped[iy:iy+ih, ix:ix+iw]

            # ---------- 步骤5: 尺寸缩放 ----------
            # 始终使用宽度约束按比例计算高度，保持内容宽高比不变形
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

            del cropped

            # ---------- 步骤6: USM锐化 ----------
            if self.sharpness != 1.0:
                resized = self._apply_usm_sharpening(resized, self.sharpness)

            # ---------- 步骤7: 极限压缩输出 ----------
            output_suffix = f".{self.compress_format}"
            output_path = self.output_dir / img_path.with_suffix(output_suffix).name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            final_output = self._compress_output(resized)

            # 使用安全方式写入（支持中文路径）
            cv2.imencode(
                output_suffix,
                final_output,
                self._get_encode_params()
            )[1].tofile(str(output_path))

            # 释放中间结果内存
            del resized, final_output

            if self.book_type == "simple":
                rect_info = "simple模式"
            elif core_rect:
                rect_info = f"{core_rect[2]}x{core_rect[3]}"
            else:
                rect_info = "原图"
            return f"[成功] {img_path.name} -> {output_path.name} (版心:{rect_info})"

        except Exception as e:
            return f"[失败] {img_path.name} - {str(e)}"

    # ======================================================
    # 图像读取（支持中文路径）
    # ======================================================

    def _read_image_safe(self, img_path: Path) -> Optional[np.ndarray]:
        """
        安全读取图像（支持中文路径）

        使用 np.fromfile + cv2.imdecode 替代 cv2.imread，
        确保中文路径正常工作。

        Args:
            img_path: 图像路径

        Returns:
            灰度图像矩阵，读取失败返回 None
        """
        try:
            # 使用 np.fromfile 读取，支持中文路径
            data = np.fromfile(str(img_path), dtype=np.uint8)

            # 使用 cv2.imdecode 解码
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if img is None:
                return None

            # 转换为灰度图
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            del img
            return gray

        except Exception as e:
            self._print(f"[警告] 读取图像失败: {img_path} - {e}")
            return None

    # ======================================================
    # 大津法二值化
    # ======================================================

    def _detect_inner_content_rect(self, gray_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        二阶段精裁：检测裁剪后图像中的文字框实际边界

        与 AncientBookCropTester._detect_inner_content_rect 逻辑对齐。

        算法：GaussianBlur → 反向Otsu → 形态学闭运算 → 连通域去噪
              → 边缘忽略区 → X/Y投影 → 预留padding

        Returns:
            (x, y, w, h) 文字框矩形，未检测到返回 None
        """
        h, w = gray_img.shape[:2]

        # 高斯模糊去底纹，反向二值化（白底黑字）
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学闭运算，将散碎文字和边框连成大块
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 连通域过滤：移除小噪点
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        sizes = stats[1:, -1]
        clean_binary = np.zeros_like(output)
        min_size = w * h * 0.001
        for i in range(nb_components - 1):
            if sizes[i] >= min_size:
                clean_binary[output == i + 1] = 255

        # 物理切断最外侧边缘干扰（跳过扫描仪黑边/物理阴影）
        ignore_x = int(w * self.edge_tolerance_ratio)
        ignore_y = int(h * self.edge_tolerance_ratio)
        clean_binary[:ignore_y, :] = 0
        clean_binary[-ignore_y:, :] = 0
        clean_binary[:, :ignore_x] = 0
        clean_binary[:, -ignore_x:] = 0

        # X/Y 轴投影
        white_ratio_per_row = clean_binary.sum(axis=1) / (255 * w)
        white_ratio_per_col = clean_binary.sum(axis=0) / (255 * h)

        THRESHOLD = 0.01  # 容错率（低于此比例视为空白）

        content_top, content_bottom = 0, h - 1
        for y in range(h):
            if white_ratio_per_row[y] > THRESHOLD:
                content_top = y
                break
        for y in range(h - 1, -1, -1):
            if white_ratio_per_row[y] > THRESHOLD:
                content_bottom = y
                break

        content_left, content_right = 0, w - 1
        for x in range(w):
            if white_ratio_per_col[x] > THRESHOLD:
                content_left = x
                break
        for x in range(w - 1, -1, -1):
            if white_ratio_per_col[x] > THRESHOLD:
                content_right = x
                break

        content_w = content_right - content_left + 1
        content_h = content_bottom - content_top + 1

        # 如果检测到的内容区域占图像比例 > 90%，不做额外裁剪（说明已经很紧凑）
        if content_w > w * 0.90 and content_h > h * 0.90:
            return None

        # 预留 padding 空余：左右各加 2%，上下各加 1.5%，避免紧贴文字边缘
        pad_x = max(4, int(content_w * 0.02))
        pad_y = max(4, int(content_h * 0.015))
        content_left = max(0, content_left - pad_x)
        content_top = max(0, content_top - pad_y)
        content_right = min(w - 1, content_right + pad_x)
        content_bottom = min(h - 1, content_bottom + pad_y)
        content_w = content_right - content_left + 1
        content_h = content_bottom - content_top + 1

        return (content_left, content_top, content_w, content_h)

    def _ensure_binary(self, gray_img: np.ndarray) -> np.ndarray:
        """
        确保图像为纯黑白二值图

        如果图像不是纯黑白（即存在灰阶），使用大津法自动阈值二值化。
        这是极限压缩前的必要降维步骤。

        Args:
            gray_img: 输入灰度图

        Returns:
            二值化后的图像
        """
        unique_vals = np.unique(gray_img)

        # 检查是否为纯黑白图
        if np.all(np.isin(unique_vals, [0, 255])):
            return gray_img

        # 使用大津法自动阈值二值化
        # 注意：OTSU阈值会对整个图像计算最优分割点
        otsu_thresh, binary = cv2.threshold(
            gray_img,
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 应用阈值缩放系数（用户可调）
        if self.otsu_scale != 1.0:
            adjusted_thresh = int(otsu_thresh * self.otsu_scale)
            _, binary = cv2.threshold(
                gray_img,
                adjusted_thresh,
                255,
                cv2.THRESH_BINARY
            )

        return binary

    # ======================================================
    # 版心提取（核心算法）
    # ======================================================

    def _extract_core_rect(self, gray_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        提取古籍版心区域

        根据 self.crop_mode 选择不同的提取策略:
        - smart: 通过Y轴投影分离页眉页脚，提取最大文字块
        - union: 所有文字连通域的全局最小包围盒
        - largest: 仅最大连通域

        Args:
            gray_img: 输入灰度图

        Returns:
            (x, y, width, height) 版心矩形区域，提取失败返回 None
        """
        h, w = gray_img.shape[:2]

        # 预处理：去噪 + 二值化
        denoised = cv2.medianBlur(gray_img, 7)
        binary = self._ensure_binary(denoised)

        del denoised

        # 动态形态学闭运算核
        # 【修复V2.1】原问题：kernel_w = max(60, w*0.08) 对窄幅图像产生过大核（2095*0.08=167px）
        # 导致竖排文字间隙全部被填满，连通域变成整图100%
        #
        # 修复方案：使用 min(宽高较小者 * 0.05) 限制核大小
        # 对于竖排古籍，闭运算核只需覆盖相邻文字间的间隙（通常10-30像素）
        min_dim = min(w, h)
        kernel_w = max(15, int(min_dim * 0.03))  # 宽度核：使用较小边的3%
        kernel_h = max(8, int(min_dim * 0.02))   # 高度核：使用较小边的2%

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))

        # 形态学闭运算：先膨胀后腐蚀，填充文字间隙
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        del binary

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            morphed, connectivity=8
        )

        del morphed, labels

        if num_labels <= 1:
            return None

        # 收集有效连通域
        valid_rects = []
        min_area = h * w * 0.003  # 最小面积阈值（过滤噪点）

        for i in range(1, num_labels):
            x, y, cw, ch, area = stats[i]

            # 面积过滤
            if area < min_area:
                continue

            # 伪影过滤（边缘细长黑块检测）
            if self._is_artifact_block(x, y, cw, ch, area, w, h):
                continue

            valid_rects.append((x, y, cw, ch, area))

        if not valid_rects:
            return None

        # 【修复V2.2】连通域版心检测后备：检测连通域版心是否覆盖95%以上图像
        # 如果是，说明闭运算核过大导致所有文字被合并，此时使用投影法替代
        cc_based_rect = None
        if self.crop_mode == 'largest':
            cc_based_rect = self._extract_largest(valid_rects)
        elif self.crop_mode == 'union':
            cc_based_rect = self._union_rects(valid_rects)
        elif self.crop_mode == 'smart':
            cc_based_rect = self._extract_smart(valid_rects, h)

        if cc_based_rect:
            cx, cy, cw, ch = cc_based_rect
            coverage = (cw * ch) / (w * h)
            # 如果连通域法得到的版心覆盖超过95%，使用投影法作为后备
            if coverage < 0.95:
                return cc_based_rect

        # 后备：投影法直接提取版心（不受闭运算核大小影响）
        proj_rect = self._extract_by_projection(self._ensure_binary(gray_img))
        if proj_rect:
            return proj_rect

        return cc_based_rect

    def _extract_by_projection(self, binary: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        【修复V2.4】投影法直接提取版心

        当连通域法因闭运算核过大而失效时，使用投影法作为后备。
        直接分析二值图的像素投影，不依赖连通域分析。

        算法：
        1. 检测并跳过边缘孤立高比例黑行/列（>90%，如边框线）
        2. 用滑动窗口平滑投影
        3. 用内容区的最大值归一化（避免边缘高值影响）
        4. 用15%阈值找内容边界

        Args:
            binary: 二值图像

        Returns:
            (x, y, width, height) 版心区域
        """
        h, w = binary.shape[:2]

        # 计算每行/列的黑色像素数量和比例
        black_per_row = (binary == 0).sum(axis=1).astype(np.float32)
        black_per_col = (binary == 0).sum(axis=0).astype(np.float32)
        black_ratio_row = black_per_row / w
        black_ratio_col = black_per_col / h

        # 【修复V2.4】第一步：检测并跳过边缘孤立高比例黑行/列（>90%）
        # 这些通常是扫描边框，不是内容
        border_threshold = 0.90

        # 找顶部边框结束位置（跳过连续>90%的行）
        y_border_end = 0
        consecutive_high = 0
        for y in range(h):
            if black_ratio_row[y] > border_threshold:
                consecutive_high += 1
                if consecutive_high >= 2:  # 连续2行>90%认为是边框
                    y_border_end = y
            else:
                consecutive_high = 0
                break

        # 找底部边框开始位置
        y_border_start = h - 1
        consecutive_high = 0
        for y in range(h - 1, -1, -1):
            if black_ratio_row[y] > border_threshold:
                consecutive_high += 1
                if consecutive_high >= 2:
                    y_border_start = y
            else:
                consecutive_high = 0
                break

        # 找左边框结束位置
        x_border_end = 0
        consecutive_high = 0
        for x in range(w):
            if black_ratio_col[x] > border_threshold:
                consecutive_high += 1
                if consecutive_high >= 2:
                    x_border_end = x
            else:
                consecutive_high = 0
                break

        # 找右边框开始位置
        x_border_start = w - 1
        consecutive_high = 0
        for x in range(w - 1, -1, -1):
            if black_ratio_col[x] > border_threshold:
                consecutive_high += 1
                if consecutive_high >= 2:
                    x_border_start = x
            else:
                consecutive_high = 0
                break

        # 第二步：滑动窗口平滑（50行/列）
        window = 50
        smoothed_rows = np.convolve(black_per_row, np.ones(window)/window, mode='same')
        smoothed_cols = np.convolve(black_per_col, np.ones(window)/window, mode='same')

        # 【修复V2.4】第三步：用内容区的最大值归一化
        # 跳过边缘区域，用内部区域的最大值来归一化
        margin = 100  # 忽略边缘100行/列
        if margin < h // 2 and margin < w // 2:
            inner_max_row = smoothed_rows[margin:-margin].max()
            inner_max_col = smoothed_cols[margin:-margin].max()
        else:
            inner_max_row = smoothed_rows.max()
            inner_max_col = smoothed_cols.max()

        inner_max_row = inner_max_row if inner_max_row > 0 else 1
        inner_max_col = inner_max_col if inner_max_col > 0 else 1

        # 【修复V2.5】阈值使用平滑后的内区最大值，但边界检测使用原始值
        # 这样可以避免平滑值的"渗透"效应导致空白区域被错误包含
        norm_rows = smoothed_rows / inner_max_row
        norm_cols = smoothed_cols / inner_max_col
        content_threshold = 0.15

        # 第四步：找Y轴边界
        # 【修复V2.8】上边界用黑色比例阈值（避免投影法回退），下边界用投影法（更准确）
        dense_ratio_threshold = 0.25

        # 上边界：直接用黑色比例阈值，避免投影法回退
        y_start = y_border_end + 1 if y_border_end > 0 else 0
        for y in range(int(y_start), h):
            if black_ratio_row[y] < dense_ratio_threshold:
                y_start = y
                break
        else:
            y_start = y_border_end + 1 if y_border_end > 0 else 0

        # 下边界：双重条件检测（归一化投影 + 原始黑色比例）
        # 避免文字较淡时底部空白被归一化后误判为有内容
        y_end = h - 1
        for y in range(h - 1, -1, -1):
            if norm_rows[y] > self.projection_threshold and black_ratio_row[y] > self.blank_ratio_threshold:
                y_end = y
                break

        # 第五步：找X轴边界（同样直接使用黑色比例阈值）
        x_start = x_border_end + 1 if x_border_end > 0 else 0
        for x in range(int(x_start), w):
            if black_ratio_col[x] < dense_ratio_threshold:
                x_start = x
                break
        else:
            x_start = x_border_end + 1 if x_border_end > 0 else 0

        x_end = x_border_start - 1 if x_border_start < w - 1 else w - 1
        for x in range(int(x_end), -1, -1):
            if black_ratio_col[x] < dense_ratio_threshold:
                x_end = x
                break
        else:
            x_end = x_border_start - 1 if x_border_start < w - 1 else w - 1

        # 确保边界有效
        content_h = y_end - y_start + 1
        content_w = x_end - x_start + 1

        # 如果内容区域太小，返回None
        if content_h < h * 0.1 or content_w < w * 0.1:
            return None

        return (x_start, y_start, content_w, content_h)

    def _is_artifact_block(
        self,
        x: int, y: int, w: int, h: int,
        area: int,
        img_w: int, img_h: int
    ) -> bool:
        """
        判断连通域是否为扫描伪影（边缘黑边/黑角）

        安全阀机制:
        - strict_artifact_filter=True: 严格模式，可能误杀贴边文字
        - strict_artifact_filter=False: 宽松模式，绝不误切文字

        Args:
            x, y, w, h: 连通域边界
            area: 连通域面积
            img_w, img_h: 图像尺寸

        Returns:
            True=是伪影应过滤，False=正常文字应保留
        """
        # 判断是否紧贴边缘（3%范围内）
        is_edge_block = (
            x < img_w * 0.03 or
            (x + w) > img_w * 0.97 or
            y < img_h * 0.03 or
            (y + h) > img_h * 0.97
        )

        if not is_edge_block:
            return False

        # 计算宽高比和面积占比
        aspect_ratio = h / float(w) if w > 0 else 0
        area_ratio = area / float(img_w * img_h)

        if self.strict_artifact_filter:
            # 严格模式：更激进地过滤
            # 条件：高度超过60% + 宽高比>15:1 + 面积<5%
            is_tall_thin = h > img_h * 0.6 and aspect_ratio > 15.0
            is_small = area_ratio < 0.05
            return is_tall_thin and is_small
        else:
            # 宽松模式：只过滤极度极端的情况
            # 条件：几乎覆盖整页高度 + 极度细长(>30:1) + 面积极小(<0.5%)
            is_tall_thin = h > img_h * 0.8 and aspect_ratio > 30.0
            is_tiny = area_ratio < 0.005
            return is_tall_thin and is_tiny

    def _extract_largest(self, valid_rects: List) -> Optional[Tuple[int, int, int, int]]:
        """
        最大连通模式：仅返回面积最大的连通域
        """
        if not valid_rects:
            return None

        best = max(valid_rects, key=lambda r: r[4])
        return (best[0], best[1], best[2], best[3])

    def _union_rects(self, rects: List) -> Optional[Tuple[int, int, int, int]]:
        """
        并集模式：计算所有连通域的全局最小包围盒

        适用于多栏古籍、竖排数列等场景，确保不遗漏任何孤立文字。
        """
        if not rects:
            return None

        min_x = min(r[0] for r in rects)
        min_y = min(r[1] for r in rects)
        max_x = max(r[0] + r[2] for r in rects)
        max_y = max(r[1] + r[3] for r in rects)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _extract_smart(
        self,
        valid_rects: List,
        img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        智能分离模式：通过Y轴垂直像素投影提取最大正文块

        算法步骤:
        1. 对所有有效连通域做Y轴像素投影（每行是否有文字）
        2. 使用闭运算连接断开的行（容忍竖排文字间隙）
        3. 寻找投影连续的"块"
        4. 选面积最大的块作为主正文
        5. 返回主正文区域内所有连通域的并集

        Args:
            valid_rects: 有效连通域列表
            img_h: 图像高度

        Returns:
            主正文区域的包围盒
        """
        # ---------- Y轴像素投影 ----------
        y_projection = np.zeros((img_h, 1), dtype=np.uint8)

        for rx, ry, rw, rh, _ in valid_rects:
            # 该连通域覆盖的行全部标记为有文字
            y_projection[ry:ry + rh, 0] = 255

        # ---------- 闭运算连接断开的行 ----------
        # gap_tolerance 控制竖排文字间隙的容忍度
        gap_tolerance = max(10, int(img_h * self.smart_gap_ratio))
        kernel_1d = np.ones((gap_tolerance, 1), dtype=np.uint8)
        y_projection_closed = cv2.morphologyEx(
            y_projection,
            cv2.MORPH_CLOSE,
            kernel_1d
        )

        # ---------- 寻找连续文字块 ----------
        contours, _ = cv2.findContours(
            y_projection_closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        del y_projection, y_projection_closed

        if not contours:
            # 无法找到连续块时， fallback 到并集模式
            return self._union_rects(valid_rects)

        # ---------- 选择面积最大的块 ----------
        best_block_area = -1
        best_y_range = (0, img_h)

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)

            # 找出属于这个块的连通域
            block_rects = [
                r for r in valid_rects
                if not (r[1] + r[3] < by or r[1] > by + bh)
            ]

            if not block_rects:
                continue

            # 计算块内文字总面积
            total_area = sum(r[4] for r in block_rects)

            if total_area > best_block_area:
                best_block_area = total_area
                best_y_range = (by, by + bh)

        # ---------- 提取主正文区域的连通域 ----------
        main_rects = [
            r for r in valid_rects
            if not (r[1] + r[3] < best_y_range[0] or r[1] > best_y_range[1])
        ]

        if main_rects:
            return self._union_rects(main_rects)
        else:
            # fallback: 返回最大连通域
            return self._extract_largest(valid_rects)

    # ======================================================
    # USM 锐化
    # ======================================================

    def _apply_usm_sharpening(self, img: np.ndarray, strength: float) -> np.ndarray:
        """
        USM (Unsharp Mask) 锐化

        通过高斯模糊加权叠加增强文字边缘对比度:
        result = img * (1 + s) - blurred * s  （s = strength）

        Args:
            img: 输入图像
            strength: 锐化强度（1.0=不锐化，>1.0增强锐化）

        Returns:
            锐化后的图像
        """
        if strength == 1.0:
            return img

        blurred = cv2.GaussianBlur(img, (0, 0), 3)

        # cv2.addWeighted: img*alpha + blurred*beta + gamma
        alpha = 1.0 + strength
        beta = -strength
        sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)

        del blurred
        return sharpened

    # ======================================================
    # 极限压缩
    # ======================================================

    def _compress_output(self, img: np.ndarray) -> np.ndarray:
        """
        对图像进行极限压缩处理

        PNG模式:
        - 二次大津法洗白（去除残留灰阶）
        - 输出 0/255 二值图（PNG 压缩后为伪彩色，不等于 1-bit）

        JPG模式:
        - 灰阶图像直接输出
        - 保留插值产生的灰阶边缘

        Args:
            img: 输入图像

        Returns:
            压缩后的图像
        """
        if self.compress_format == 'png':
            # 二次大津法洗白：彻底去除残留灰阶
            _, binary = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary
        else:
            # ---------- JPG: 灰阶高压模式 ----------
            # 确保图像为灰阶格式
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return gray

            return img

    def _get_encode_params(self) -> List:
        """
        获取 OpenCV 编码参数

        Returns:
            cv2.imencode 所需的参数列表
        """
        if self.compress_format == 'png':
            return [
                cv2.IMWRITE_PNG_COMPRESSION, self.png_compression,
            ]
        else:
            return [
                cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # 霍夫曼优化
            ]

    # ======================================================
    # 工具方法
    # ======================================================

    def _print(self, msg: str):
        """
        线程安全的打印输出

        Args:
            msg: 要打印的消息
        """
        if self.verbose:
            with self._print_lock:
                print(msg)

    def get_stats(self) -> Dict[str, int]:
        """
        获取处理统计信息

        Returns:
            包含 total/success/failed/skipped 的字典
        """
        return self._stats.copy()


# ======================================================
# 预设配置
# ======================================================

class AncientBookPresets:
    """
    预设配置类

    提供常用场景的优化参数组合
    """

    @staticmethod
    def kindle_vertical() -> Dict[str, Any]:
        """
        Kindle KPW6 竖排古籍预设

        特点：极限压缩，文字保护优先
        """
        return {
            "crop_mode": "smart",
            "smart_gap_ratio": 0.012,
            "strict_artifact_filter": False,
            "edge_tolerance_ratio": 0.02,
            "padding_ratio_w": 0.025,
            "padding_ratio_h": 0.025,
            "padding_min_pixel": 30,
            "sharpness": 1.3,
            "otsu_scale": 1.05,
            "compress_format": "png",
            "max_workers": 4,
        }

    @staticmethod
    def modern_reprint() -> Dict[str, Any]:
        """
        现代横排重印本预设

        特点：smart模式自动分离页眉页脚
        """
        return {
            "crop_mode": "smart",
            "smart_gap_ratio": 0.025,
            "strict_artifact_filter": True,
            "edge_tolerance_ratio": 0.04,
            "padding_ratio_w": 0.015,
            "padding_ratio_h": 0.015,
            "padding_min_pixel": 20,
            "sharpness": 1.1,
            "otsu_scale": 1.0,
            "compress_format": "png",
            "max_workers": 4,
        }

    @staticmethod
    def multicolumn() -> Dict[str, Any]:
        """
        多栏古籍/竖排数列预设

        特点：union模式不遗漏任何孤立文字
        """
        return {
            "crop_mode": "union",
            "strict_artifact_filter": False,
            "edge_tolerance_ratio": 0.08,
            "padding_ratio_w": 0.03,
            "padding_ratio_h": 0.03,
            "padding_min_pixel": 25,
            "sharpness": 1.2,
            "otsu_scale": 1.0,
            "compress_format": "png",
            "max_workers": 4,
        }

    @staticmethod
    def max_compression() -> Dict[str, Any]:
        """
        极致压缩预设

        特点：文件体积优先，牺牲部分文字完整性
        """
        return {
            "crop_mode": "largest",
            "strict_artifact_filter": True,
            "edge_tolerance_ratio": 0.03,
            "padding_ratio_w": 0.01,
            "padding_ratio_h": 0.01,
            "padding_min_pixel": 15,
            "sharpness": 1.0,
            "otsu_scale": 1.0,
            "compress_format": "png",
            "max_workers": 6,
        }


# ======================================================
# 入口点
# ======================================================

if __name__ == "__main__":
    # 示例：使用 Kindle 竖排古籍预设处理图像
    config = {
        "input_dir": "./input/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版",
        "output_dir": "./output_final/小檀栅室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本",
        "target_width": 1264,
        "target_height": 1680,
    }

    # 合并预设配置
    config.update(AncientBookPresets.kindle_vertical())

    # 创建引擎并处理
    engine = AncientBookEngine(**config)
    engine.process_directory("*.png")
