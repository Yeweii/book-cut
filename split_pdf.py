# -*- coding: utf-8 -*-
"""
PDF 按页码范围切分为多个 PDF，保留书签

支持两种使用方式：
  1. 函数调用：split_pdf(input_pdf, output_dir, splits)
  2. YAML 配置：split_pdf_from_config(config_path)
"""

import fitz
import yaml
import os
from pathlib import Path
from typing import List, Dict


def split_pdf(
    input_pdf: str,
    output_dir: str,
    splits: List[Dict],
    preserve_bookmarks: bool = True,
) -> List[str]:
    """
    按页码范围切分 PDF

    Args:
        input_pdf: 输入 PDF 路径
        output_dir: 输出目录
        splits: 切分配置列表，每项包含:
            - range: [start, end]，1-indexed 页码范围
            - name: 输出文件名（不含 .pdf 扩展名）
        preserve_bookmarks: 是否保留书签

    Returns:
        输出文件路径列表
    """
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_pdf}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取源 PDF 书签
    src_doc = fitz.open(str(input_path))
    src_toc = src_doc.get_toc() if preserve_bookmarks else []
    total_pages = src_doc.page_count
    src_doc.close()

    print(f"输入 PDF: {input_path.name}")
    print(f"总页数: {total_pages}")
    print(f"书签数量: {len(src_toc)}")
    print(f"切分份数: {len(splits)}")
    print("-" * 50)

    output_paths = []

    for i, split in enumerate(splits, 1):
        range_start, range_end = split["range"]
        name = split["name"]

        # 验证页码范围
        if range_start < 1 or range_end > total_pages or range_start > range_end:
            print(f"[警告] 跳过无效范围 [{range_start}, {range_end}]: 超出 PDF 页数 (1-{total_pages})")
            continue

        # 切分 PDF（fitz 使用 0-indexed）
        src_doc = fitz.open(str(input_path))
        out_doc = fitz.open()
        out_doc.insert_pdf(src_doc, from_page=range_start - 1, to_page=range_end - 1)
        src_doc.close()

        # 处理书签
        if preserve_bookmarks and src_toc:
            split_toc = _filter_toc_for_range(src_toc, range_start, range_end)
            if split_toc:
                out_doc.set_toc(split_toc)

        # 保存输出
        output_path = output_dir / f"{name}.pdf"
        out_doc.save(str(output_path), encryption=fitz.PDF_ENCRYPT_NONE)
        out_doc.close()

        page_count = range_end - range_start + 1
        bookmark_count = len(_filter_toc_for_range(src_toc, range_start, range_end)) if src_toc else 0
        print(f"[{i}] {name}.pdf - 页码 {range_start}-{range_end} ({page_count} 页), 书签 {bookmark_count} 条")

        output_paths.append(str(output_path))

    print("-" * 50)
    print(f"完成！输出 {len(output_paths)} 个文件至: {output_dir}")

    return output_paths


def _filter_toc_for_range(toc: list, range_start: int, range_end: int) -> list:
    """
    过滤书签，保留落在指定页码范围内的书签，并调整页码

    Args:
        toc: 源 PDF 书签列表
        range_start: 起始页（1-indexed）
        range_end: 结束页（1-indexed）

    Returns:
        调整后的书签列表
    """
    offset = range_start - 1  # 0-indexed 偏移量
    filtered_toc = []

    for item in toc:
        if len(item) >= 3:
            level = item[0]
            title = item[1]
            page = item[2]

            # 检查书签页码是否在范围内
            if range_start <= page <= range_end:
                # 调整页码（减去偏移量）
                new_item = [level, title, page - offset]
                filtered_toc.append(new_item)

    return filtered_toc


def split_pdf_from_config(config_path: str) -> list[str]:
    """
    从 YAML 配置文件加载并执行切分

    Args:
        config_path: YAML 配置文件路径

    Returns:
        输出文件路径列表
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_pdf = config.get("input_pdf")
    output_dir = config.get("output_dir", "./output")
    splits = config.get("splits", [])
    preserve_bookmarks = config.get("preserve_bookmarks", True)

    if not input_pdf:
        raise ValueError("配置文件缺少 input_pdf 字段")

    if not splits:
        raise ValueError("配置文件缺少 splits 字段")

    return split_pdf(
        input_pdf=input_pdf,
        output_dir=output_dir,
        splits=splits,
        preserve_bookmarks=preserve_bookmarks,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF 按页码范围切分")
    parser.add_argument("input_pdf", nargs="?", help="输入 PDF 文件")
    parser.add_argument("--config", "-c", help="YAML 配置文件路径")
    parser.add_argument("--output-dir", "-o", default="./output", help="输出目录")
    parser.add_argument("--ranges", help="页码范围，如 1-10,11-20,21-30")
    parser.add_argument("--names", help="输出文件名，如 卷一,卷二,卷三")
    parser.add_argument("--no-bookmarks", action="store_true", help="不保留书签")

    args = parser.parse_args()

    if args.config:
        # 从配置文件加载
        split_pdf_from_config(args.config)
    elif args.input_pdf and args.ranges:
        # 从命令行参数构建
        ranges = args.ranges.split(",")
        names = args.names.split(",") if args.names else [f"part_{i+1}" for i in range(len(ranges))]

        splits = []
        for r, n in zip(ranges, names):
            start, end = map(int, r.split("-"))
            splits.append({"range": [start, end], "name": n.strip()})

        split_pdf(
            args.input_pdf,
            args.output_dir,
            splits,
            preserve_bookmarks=not args.no_bookmarks,
        )
    else:
        # 无参数时，尝试使用默认配置文件
        default_config = "split_pdf_config.yaml"
        if Path(default_config).exists():
            print(f"使用默认配置文件: {default_config}")
            split_pdf_from_config(default_config)
        else:
            parser.print_help()
