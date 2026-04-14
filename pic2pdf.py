import os
import re
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed


def natural_sort_key(s):
    """自然排序 key：数字部分按数值比较"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def images_to_pdf(folder_path, output_pdf_path=None):
    """
    将图片文件夹合并为单一 PDF

    Args:
        folder_path: 图片文件夹路径
        output_pdf_path: 输出 PDF 路径，None 则自动生成在同级目录
    """
    if not os.path.isdir(folder_path):
        print(f"错误：找不到文件夹 '{folder_path}'")
        return

    folder_path = os.path.normpath(folder_path)
    folder_name = os.path.basename(folder_path)

    if output_pdf_path is None:
        parent_dir = os.path.dirname(folder_path)
        output_pdf_path = os.path.join(parent_dir, f"{folder_name}.pdf")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = sorted(
        [f for f in os.listdir(folder_path)
         if os.path.splitext(f)[1].lower() in valid_extensions],
        key=natural_sort_key
    )

    if not image_files:
        print(f"文件夹 '{folder_path}' 中没有找到支持的图片文件。")
        return

    print(f"正在读取 '{folder_name}' 文件夹...")
    print(f"共找到 {len(image_files)} 张图片，准备合并...")
    print("-" * 30)

    # 步骤 1：多线程获取图片尺寸（I/O 密集型，并行加速）
    img_meta_list = []
    failed = []

    def load_img_meta(filename):
        img_path = os.path.join(folder_path, filename)
        try:
            pix = fitz.Pixmap(img_path)
            w, h = pix.width, pix.height
            return filename, img_path, w, h, None
        except Exception as e:
            return filename, img_path, 0, 0, str(e)

    max_workers = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_img_meta, f): f for f in image_files}
        for future in as_completed(futures):
            filename, img_path, w, h, err = future.result()
            if err:
                print(f"读取图片 {filename} 时跳过 (错误: {err})")
                failed.append(filename)
            else:
                img_meta_list.append((filename, img_path, w, h))

    # 保持原始顺序（as_completed 不保证顺序）
    img_meta_list.sort(key=lambda x: natural_sort_key(x[0]))

    if not img_meta_list:
        print("没有成功加载任何图片，合并取消。")
        return

    # 步骤 2：顺序构建 PDF（fitz 文档写入必须串行）
    doc = fitz.open()
    for filename, img_path, w, h in img_meta_list:
        try:
            page = doc.new_page(width=w, height=h)
            page.insert_image(page.rect, filename=img_path)
            print(f"已加载: {filename}")
        except Exception as e:
            print(f"插入图片 {filename} 时跳过 (错误: {e})")

    if len(doc) == 0:
        print("PDF 中没有页面，保存取消。")
        doc.close()
        return

    # 确保输出目录存在
    out_dir = os.path.dirname(output_pdf_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    try:
        doc.save(output_pdf_path, deflate=True)
        print("-" * 30)
        print(f"合并成功！PDF 已保存至: {output_pdf_path}")
    except Exception as e:
        print(f"保存 PDF 时发生错误: {e}")
        raise
    finally:
        doc.close()


if __name__ == "__main__":
    # image_folder_path = "./output_final/小檀栅室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本"
    image_folder_path = "./output_final_invert/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版"  # 反色后的输出目录

    images_to_pdf(image_folder_path)
