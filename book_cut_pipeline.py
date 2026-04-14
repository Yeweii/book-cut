# -*- coding: utf-8 -*-
"""
古籍裁剪工具 - 统一入口

支持两种工作模式：
  - from_pdf():    PDF → 提取图片 → 反色检测 → 裁剪压缩 → 生成 PDF → 复制书签 → PDF
  - from_images(): 图片文件夹 → 反色检测 → 裁剪压缩 → PDF

依赖：
  - extractPdf.py
  - invert_color.py (含 detect_inverted, batch_invert_if_needed)
  - ancient_book_engine.py (AncientBookEngine)
  - pic2pdf.py (images_to_pdf)
"""

import os
import tempfile
import time
from pathlib import Path

import fitz

from extractPdf import extract_images_from_pdf
from invert_color import detect_inverted, batch_invert_if_needed
from ancient_book_engine import AncientBookEngine, AncientBookPresets
from pic2pdf import images_to_pdf
from split_spread import batch_split


class AncientBookPipeline:
    """
    古籍图像裁剪流水线

    完整流程：
        PDF → extractPdf → 图片序列 → 反色检测 → AncientBookEngine → 裁剪压缩 → pic2pdf → PDF
        或
        图片文件夹 → (跨页拆分) → 反色检测 → 裁剪压缩 → PDF

    ==========================================
    参数详解与取值范围
    ==========================================

    【输出尺寸】
      target_width  : 统一缩放目标宽度（像素）
                     取值范围：≥ 100，推荐 1264～1680（300dpi A4 竖版）
                     参考值：1264（轻度压缩）、1680（高质量）

      target_height : 统一缩放目标高度（像素）
                     取值范围：≥ 100，推荐 1680～2240（300dpi A4 竖版）
                     参考值：1680（轻度压缩）、2240（高质量）

    【版心裁剪】
      crop_mode     : 版心提取模式
                     取值："smart"（智能合并）、"union"（外接矩形）、"largest"（最大连通域）
                     参考值："smart"（推荐，适用于大多数古籍）
                     - smart  : 检测所有连通域，取并集后加 padding，适用于多栏古籍
                     - union  : 取所有连通域的外接矩形
                     - largest: 仅保留最大连通域，适用于单页单栏古籍

      smart_gap_ratio: smart 模式下，小于该比例的空白间隙会被合并
                     取值范围：0.0 ～ 0.1
                     参考值：0.012（默认）

      strict_artifact_filter: 是否启用严格杂点过滤（对红色印章/批注更敏感）
                     取值：True/False
                     参考值：False（默认）

      edge_tolerance_ratio : 边缘留白比例（预防切到文字边缘）
                     取值范围：0.0 ～ 0.1
                     参考值：0.02（默认）

    【Padding（版心周围白边）】
      padding_ratio_w: 左右 padding 占版心宽度的比例
                     取值范围：0.0 ～ 0.2
                     参考值：0.025（默认），0.05（宽松）

      padding_ratio_h: 上下 padding 占版心高度的比例
                     取值范围：0.0 ～ 0.2
                     参考值：0.025（默认），0.05（宽松）

      padding_min_pixel: padding 最小像素值（绝对值兜底）
                     取值范围：≥ 0
                     参考值：30（默认）

    【图像增强】
      sharpness     : USM 锐化强度
                     取值范围：0.0 ～ 3.0，0=关闭
                     参考值：1.3（默认），1.8（增强），0（关闭）

    【二值化】
      otsu_scale    : Otsu 阈值的缩放系数（>1 则更白，<1 则更黑）
                     取值范围：0.5 ～ 2.0
                     参考值：1.05（默认）

    【投影法阈值】（控制版心检测精度）
      projection_threshold: 归一化投影阈值，判断是否为内容区域
                     取值范围：0.05 ～ 0.5
                     参考值：0.15（默认），越小越严格（可能切到文字）
      blank_ratio_threshold: 原始黑色比例下限，排除纯空白噪声
                     取值范围：0.01 ～ 0.2
                     参考值：0.05（默认）

    【压缩格式】
      compress_format: 输出图片格式
                     取值："png"（无损）、"jpg"（高压缩）
                     参考值："png"（默认，文字古籍推荐）

      png_compression: PNG 压缩级别
                     取值范围：0 ～ 9（0=最快，9=最小）
                     参考值：9（默认）

      jpeg_quality  : JPEG 输出质量
                     取值范围：1 ～ 100
                     参考值：85（默认）

    【性能】
      max_workers   : 最大并发线程数，None=自动（cpu_count）
                     取值范围：≥ 1
                     参考值：None（默认）

      verbose       : 是否打印详细日志
                     取值：True/False
                     参考值：True（默认）

    【反色控制】
      invert         : 是否对黑底白字图片执行反色
                     取值：None（自动检测）、True（强制反色）、False（跳过）
                     参考值：None（默认，首次处理建议用自动检测）

    【跨页拆分】
      split_spread   : 是否将跨页扫描拆分为左右独立页面
                     取值：True（拆分）、False（跳过，默认）
                     参考值：False（默认），True（处理古籍双页扫描时启用）

    【书籍类型】
      book_type      : 书籍类型，决定裁剪策略
                     取值："ancient"（古籍复杂版心提取）、"simple"（普通书籍仅裁白边）
                     参考值："ancient"（默认）
                     说明：
                       - ancient: 使用连通域分析、Y轴投影、形态学闭运算等复杂算法
                       - simple : 仅检测并裁剪四边空白，适用于无框线的普通书籍扫描件

    ==========================================
    使用示例
    ==========================================

    # 标准流程（PDF → 裁剪 → PDF）
    pipeline = AncientBookPipeline(
        target_width=1264,
        target_height=1680,
        crop_mode="smart",
    )
    pipeline.from_pdf("./input/古籍.pdf", "./output/古籍_裁剪.pdf")

    # 高质量输出
    pipeline = AncientBookPipeline(
        target_width=1680,
        target_height=2240,
        crop_mode="smart",
        compress_format="png",
        png_compression=9,
    )

    # 处理黑底白字（强制反色）
    pipeline = AncientBookPipeline(
        invert=True,  # 强制反色
    )

    # 处理跨页扫描（自动拆分左右页）
    pipeline = AncientBookPipeline(
        split_spread=True,  # 启用跨页拆分
    )

    # 从图片文件夹开始（跳过步骤1）
    pipeline = AncientBookPipeline()
    pipeline.from_images("./input/图片文件夹", "./output/图片_裁剪.pdf")

    # 断点续传（从步骤3开始，跳过已完成的1和2）
    pipeline = AncientBookPipeline()
    pipeline.from_images("./input/图片文件夹", start_step=3)

    # 普通书籍模式（仅裁白边，不做复杂版心提取）
    pipeline = AncientBookPipeline(
        book_type="simple",  # 普通书籍仅裁白边
    )
    pipeline.from_images("./input/普通书籍", "./output/普通书籍_裁剪.pdf")
    """

    def __init__(
        self,
        # 【输出尺寸】
        target_width: int = 1264,      # 输出宽度(px)，推荐 1264-1680
        target_height: int = 1680,      # 输出高度(px)，推荐 1680-2240

        # 【版心裁剪】
        crop_mode: str = "smart",       # "smart"(推荐)/"union"/"largest"
        smart_gap_ratio: float = 0.012,  # smart模式空白间隙合并阈值 0.0-0.1
        strict_artifact_filter: bool = False,  # 严格杂点过滤 True/False，推荐 False
        edge_tolerance_ratio: float = 0.02,  # 边缘留白防切边 0.0-0.1，推荐 0.02

        # 【Padding】
        padding_ratio_w: float = 0.025,    # 左右padding占版心宽度比例 0.0-0.2，推荐 0.025
        padding_ratio_h: float = 0.025,    # 上下padding占版心高度比例 0.0-0.2，推荐 0.025
        padding_min_pixel: int = 30,      # padding最小像素兜底值，推荐 30

        # 【图像增强】
        sharpness: float = 1.3,          # USM锐化强度 0.0-3.0，推荐 1.3

        # 【二值化】
        otsu_scale: float = 1.05,       # Otsu阈值缩放 >1更白<1更黑 0.5-2.0，推荐 1.05
        projection_threshold: float = 0.15,  # 投影法阈值 0.05-0.5，推荐 0.15
        blank_ratio_threshold: float = 0.05,  # 黑色比例下限 0.01-0.2，推荐 0.05

        # 【压缩格式】
        compress_format: str = "png",     # "png"(推荐)/"jpg"
        png_compression: int = 9,      # PNG压缩级别 0-9，9=最小，推荐 9
        jpeg_quality: int = 85,         # JPEG质量 1-100，推荐 85

        # 【性能】
        max_workers: int = None,         # 最大线程数，推荐 None(自动)
        verbose: bool = True,           # 详细日志 True/False

        # 【其他】
        invert: bool = None,            # 反色控制 None(自动)/True/False，推荐 None
        split_spread: bool = False,   # 跨页拆分 True/False，推荐 False
        split_mode: str = "gradient",  # 拆分模式 "gradient"(默认)/"frame"，仅在 split_spread=True 时有效
        book_type: str = "ancient",    # "ancient"/"simple"，推荐 "ancient"
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.crop_mode = crop_mode
        self.compress_format = compress_format
        self.invert = invert  # None=自动检测，True=强制反色，False=跳过反色
        self.split_spread = split_spread
        self.split_mode = split_mode
        self.book_type = book_type
        self.engine_params = dict(
            target_width=target_width,
            target_height=target_height,
            crop_mode=crop_mode,
            smart_gap_ratio=smart_gap_ratio,
            strict_artifact_filter=strict_artifact_filter,
            edge_tolerance_ratio=edge_tolerance_ratio,
            padding_ratio_w=padding_ratio_w,
            padding_ratio_h=padding_ratio_h,
            padding_min_pixel=padding_min_pixel,
            sharpness=sharpness,
            otsu_scale=otsu_scale,
            projection_threshold=projection_threshold,
            blank_ratio_threshold=blank_ratio_threshold,
            compress_format=compress_format,
            png_compression=png_compression,
            jpeg_quality=jpeg_quality,
            max_workers=max_workers,
            verbose=verbose,
            book_type=book_type,
        )

    def _get_engine(self, input_dir, output_dir):
        return AncientBookEngine(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            **self.engine_params,
        )

    def _resolve_output_pdf_path(self, output_pdf_path, pdf_path):
        """解析输出路径：支持仅传入目录，或 None（自动生成）"""
        if output_pdf_path is None:
            return str(pdf_path.with_suffix("_cut.pdf"))
        p = Path(output_pdf_path)
        if p.is_dir() or str(p).endswith(("/", "\\")):
            p.mkdir(parents=True, exist_ok=True)
            return str(p / f"{pdf_path.stem}_cut.pdf")
        return str(p)

    def from_pdf(self, pdf_path, output_pdf_path=None, start_step=1, split_spread=None):
        """
        模式 A：PDF → 裁剪 → 输出 PDF

        Args:
            pdf_path: 输入 PDF 路径
            output_pdf_path: 输出 PDF 路径，None 则自动生成在 PDF 同级目录
            start_step: 从第几步开始（跳过前置步骤）
                1 - 从 extractPdf 开始（默认）
                2 - 从反色检测开始（图片已在 pdf.with_suffix("") 目录）
                3 - 从裁剪压缩开始
                4 - 从生成 PDF 开始（裁剪图片在 cropped_dir）
                5 - 复制书签
            split_spread: 是否拆分跨页图像，None=使用构造函数的默认值，True=启用，False=跳过
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"错误：找不到文件 '{pdf_path}'")
            return

        # 解析输出路径：支持仅传入目录或文件名
        output_pdf_path = self._resolve_output_pdf_path(output_pdf_path, pdf_path)

        start_time = time.time()

        # 图片输出目录（extractPdf 的固定输出位置）
        extracted_imgs_dir = pdf_path.with_suffix("")

        with tempfile.TemporaryDirectory() as tmpdir:
            invert_dir = Path(tmpdir) / "inverted"
            invert_dir.mkdir()
            split_dir = Path(tmpdir) / "split"
            split_dir.mkdir()
            cropped_dir = Path(tmpdir) / "cropped"
            cropped_dir.mkdir()

            # ---------- 步骤 1：提取 PDF 图片 ----------
            if start_step <= 1:
                if not extracted_imgs_dir.exists():
                    extracted_imgs_dir.mkdir(parents=True, exist_ok=True)
                print("\n========== 步骤 1：提取 PDF 图片 ==========")
                extract_images_from_pdf(str(pdf_path.resolve()))
                if not extracted_imgs_dir.exists():
                    print("错误：未能提取到图片")
                    return
            elif start_step <= 2:
                # 验证前置目录
                if not extracted_imgs_dir.exists():
                    print(f"错误：跳过了步骤 1，请确认图片已在 {extracted_imgs_dir}")
                    return
                print(f"[跳过] 步骤 1（已提取）")

            # ---------- 步骤 0：跨页拆分（在提取之后） ----------
            pre_split_dir = None
            _do_split = split_spread if split_spread is not None else self.split_spread
            if _do_split:
                print("\n========== 步骤 0：跨页拆分 ==========")
                batch_split(extracted_imgs_dir, str(split_dir), base_num=1, split_mode=self.split_mode)
                pre_split_dir = split_dir
            else:
                print("[跳过] 步骤 0（split_spread=False）")

            # ---------- 步骤 2：反色处理 ----------
            # split_spread 的输出目录优先作为反色处理的输入
            invert_src = pre_split_dir if pre_split_dir else extracted_imgs_dir
            if start_step <= 2:
                print("\n========== 步骤 2：反色处理 ==========")
                if self.invert is True:
                    print("强制反色...")
                    batch_invert_if_needed(invert_src, invert_dir)
                    process_dir = invert_dir
                elif self.invert is False:
                    print("跳过反色（invert=False）")
                    process_dir = invert_src
                else:
                    # None：自动检测
                    sample_img = next(invert_src.glob("*.png"), None) \
                              or next(invert_src.glob("*.jpg"), None)
                    if sample_img and detect_inverted(sample_img):
                        print("检测到黑底白字图片，开始反色...")
                        batch_invert_if_needed(invert_src, invert_dir)
                        process_dir = invert_dir
                    else:
                        print("未检测到反色需求，直接使用原图")
                        process_dir = invert_src
            else:
                print(f"[跳过] 步骤 2（跳过反色处理）")
                process_dir = invert_src

            # ---------- 步骤 3：裁剪压缩 ----------
            if start_step <= 3:
                print("\n========== 步骤 3：裁剪压缩 ==========")
                engine = self._get_engine(process_dir, cropped_dir)
                engine.process_directory()
            else:
                print(f"[跳过] 步骤 3（使用 cropped_dir）")

            # ---------- 步骤 4：生成 PDF ----------
            if start_step <= 4:
                print("\n========== 步骤 4：生成 PDF ==========")
                try:
                    images_to_pdf(str(cropped_dir), output_pdf_path)
                except Exception:
                    print("PDF 生成失败，跳过后续步骤")
                    return
            else:
                print(f"[跳过] 步骤 4（使用已生成的 PDF）")

            # ---------- 步骤 5：复制书签 ----------
            if start_step <= 5:
                print("\n========== 步骤 5：复制书签 ==========")
                src_doc = fitz.open(str(pdf_path.resolve()))
                toc = src_doc.get_toc()
                src_doc.close()
                if toc:
                    dst_doc = fitz.open(output_pdf_path)
                    dst_doc.set_toc(toc)
                    tmp = output_pdf_path + ".tmp"
                    dst_doc.save(tmp)
                    dst_doc.close()
                    os.replace(tmp, output_pdf_path)
                    print(f"已复制 {len(toc)} 条书签")
                else:
                    print("源 PDF 无书签，跳过")

            elapsed = time.time() - start_time
            print(f"\n========== 完成！输出：{output_pdf_path} | 耗时：{elapsed:.1f}s ==========")

    def from_images(self, image_dir, output_pdf_path=None, start_step=1, split_spread=None):
        """
        模式 B：图片文件夹 → 裁剪 → 输出 PDF

        Args:
            image_dir: 输入图片文件夹路径
            output_pdf_path: 输出 PDF 路径，None 则自动生成在同级目录
            start_step: 从第几步开始
                1 - 从反色检测开始（默认）
                2 - 从裁剪压缩开始
                3 - 从生成 PDF 开始
                4 - 书签复制
            split_spread: 是否拆分跨页图像，None=使用构造函数的默认值，True=启用，False=跳过
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"错误：找不到文件夹 '{image_dir}'")
            return

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            invert_dir = Path(tmpdir) / "inverted"
            invert_dir.mkdir()
            split_dir = Path(tmpdir) / "split"
            split_dir.mkdir()
            cropped_dir = Path(tmpdir) / "cropped"
            cropped_dir.mkdir()

            # ---------- 步骤 0：跨页拆分 ----------
            pre_split_dir = None
            _do_split = split_spread if split_spread is not None else self.split_spread
            if _do_split:
                print("\n========== 步骤 0：跨页拆分 ==========")
                batch_split(image_dir, str(split_dir), base_num=1, split_mode=self.split_mode)
                pre_split_dir = split_dir
            else:
                print("[跳过] 步骤 0（split_spread=False）")

            # ---------- 步骤 1：反色处理 ----------
            if start_step <= 1:
                print("\n========== 步骤 1：反色处理 ==========")
                invert_src = pre_split_dir if pre_split_dir else image_dir
                if self.invert is True:
                    print("强制反色...")
                    batch_invert_if_needed(invert_src, invert_dir)
                    process_dir = invert_dir
                elif self.invert is False:
                    print("跳过反色（invert=False）")
                    process_dir = invert_src
                else:
                    # None：自动检测
                    sample_img = next(invert_src.glob("*.png"), None) \
                              or next(invert_src.glob("*.jpg"), None)
                    if sample_img and detect_inverted(sample_img):
                        print("检测到黑底白字图片，开始反色...")
                        batch_invert_if_needed(invert_src, invert_dir)
                        process_dir = invert_dir
                    else:
                        print("未检测到反色需求，直接使用原图")
                        process_dir = invert_src
            else:
                print(f"[跳过] 步骤 1（跳过反色处理）")
                process_dir = pre_split_dir if pre_split_dir else image_dir

            # ---------- 步骤 2：裁剪压缩 ----------
            if start_step <= 2:
                print("\n========== 步骤 2：裁剪压缩 ==========")
                engine = self._get_engine(process_dir, cropped_dir)
                engine.process_directory()
            else:
                print(f"[跳过] 步骤 2（使用 cropped_dir）")

            # ---------- 步骤 3：生成 PDF ----------
            if start_step <= 3:
                print("\n========== 步骤 3：生成 PDF ==========")
                if output_pdf_path is None:
                    output_pdf_path = str(image_dir.parent / f"{image_dir.name}_cut.pdf")
                else:
                    p = Path(output_pdf_path)
                    if p.is_dir() or str(p).endswith(("/", "\\")):
                        p.mkdir(parents=True, exist_ok=True)
                        output_pdf_path = str(p / f"{image_dir.name}_cut.pdf")
                try:
                    images_to_pdf(str(cropped_dir), output_pdf_path)
                except Exception:
                    print("PDF 生成失败，跳过后续步骤")
                    return
            else:
                if output_pdf_path is None:
                    output_pdf_path = str(image_dir.parent / f"{image_dir.name}_cut.pdf")
                print(f"[跳过] 步骤 3（使用已生成的 PDF）")

            elapsed = time.time() - start_time
            print(f"\n========== 完成！输出：{output_pdf_path} | 耗时：{elapsed:.1f}s ==========")


if __name__ == "__main__":
    # ==========================================
    # 示例用法
    # ==========================================

    # ---- 模式 A：从 PDF 开始 ----
    # 标准裁剪（黑底白字自动检测）
    pipeline = AncientBookPipeline(
        target_width=1264,
        target_height=1680,
        crop_mode="smart",
        book_type="simple",
        # compress_format="jpg",
        # split_mode="frame",

    )
    pipeline.from_pdf(
        "./input/"
        "苏轼全集校注（1-9 诗词集） (苏轼) .pdf",
        "./output/",
        # split_spread= False,  # 启用跨页拆分
        # split_spread=True,
        # start_step=2,
    )

    # ---- 模式 B：从图片文件夹开始 ----
    # 图片文件夹 → 反色检测 → 裁剪 → PDF
    # pipeline = AncientBookPipeline(
    #     target_width=1264,
    #     target_height=1680,
    #     crop_mode="smart",
    #     invert=None,  # 自动检测（默认）
    # )
    # pipeline.from_images("./input/图片文件夹", "./output/图片文件夹_裁剪.pdf")

    # ---- 处理跨页扫描 ----
    # 自动拆分双页 → 反色检测 → 裁剪 → PDF
    # pipeline = AncientBookPipeline(
    #     split_spread=True,  # 启用跨页拆分
    #     invert=True,        # 黑底白字古籍通常需要反色
    # )
    # pipeline.from_images("./input/跨页扫描件文件夹", "./output/跨页扫描件_裁剪.pdf")

    # ---- 断点续传（跳过已完成步骤） ----
    # pipeline.from_images("./input/图片文件夹", start_step=3)  # 从生成 PDF 开始

    # ---- 高质量输出（更大尺寸） ----
    # pipeline = AncientBookPipeline(
    #     target_width=1680,
    #     target_height=2240,
    #     compress_format="png",
    #     png_compression=9,
    #     sharpness=1.8,
    # )
    # pipeline.from_pdf("./input/古籍.pdf", "./output/古籍_高质量_裁剪.pdf")
