import fitz
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def _render_page(args):
    """线程 worker：渲染单个页面并写入文件"""
    page_num, output_dir, digits, pdf_path = args
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        doc.close()

        image_filename = f"{page_num + 1:0{digits}}.jpg"
        image_filepath = os.path.join(output_dir, image_filename)
        # 使用 JPG 压缩，quality=85
        pix.save(image_filepath, jpg_quality=85)

        return f"成功提取: {image_filename}"
    except Exception as e:
        return f"提取第 {page_num + 1} 页时出错: {e}"


def extract_images_from_pdf(pdf_path):
    # 1. 检查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"错误：找不到文件 '{pdf_path}'")
        return

    # 2. 获取 PDF 所在目录、不带扩展名的文件名
    pdf_dir = os.path.dirname(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # 3. 构建同名输出文件夹路径并创建
    output_dir = os.path.join(pdf_dir, base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"正在读取 PDF: {pdf_path}")
    print(f"图片将被保存至: {output_dir}")
    print("-" * 30)

    # 4. 打开 PDF 文件
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"打开 PDF 时发生错误: {e}")
        return

    page_count = len(pdf_document)
    digits = len(str(page_count))
    pdf_document.close()

    if page_count == 0:
        print("PDF 没有页面")
        return

    # 5. 准备多线程任务参数（文件名在提交时就确定，顺序与完成顺序无关）
    task_args = [
        (page_num, output_dir, digits, pdf_path)
        for page_num in range(page_count)
    ]

    print(f"共 {page_count} 页，开始多线程渲染...")
    print("-" * 30)

    # 6. 多线程渲染
    success_count = 0
    fail_count = 0
    max_workers = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_render_page, arg): arg for arg in task_args}
        for future in as_completed(futures):
            msg = future.result()
            print(msg)
            if "成功" in msg:
                success_count += 1
            else:
                fail_count += 1

    print("-" * 30)
    print(f"提取完成！成功: {success_count} | 失败: {fail_count}")


if __name__ == "__main__":
    pdf_file_path = "./input/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版.pdf"
    extract_images_from_pdf(pdf_file_path)
