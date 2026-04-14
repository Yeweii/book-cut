import fitz


def copy_pdf_bookmarks(src_pdf, dst_pdf, output_pdf):
    """
    从源 PDF 读取书签，复制到目标 PDF 并保存

    Args:
        src_pdf: 源 PDF（书签来源）
        dst_pdf: 目标 PDF（书签目标，与源 PDF 页面结构应一致）
        output_pdf: 输出 PDF 路径
    """
    src = fitz.open(src_pdf)
    toc = src.get_toc()
    src.close()

    if not toc:
        print(f"源 PDF '{src_pdf}' 没有书签")
        return

    print(f"读取到 {len(toc)} 条书签")

    dst = fitz.open(dst_pdf)
    try:
        dst.set_toc(toc)
        dst.save(output_pdf)
        print(f"书签已复制，输出至: {output_pdf}")
    except Exception as e:
        print(f"保存失败: {e}")
    finally:
        dst.close()


if __name__ == "__main__":
    copy_pdf_bookmarks(
        "./input/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版.pdf",
        "./output_final_invert/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版.pdf",
        "./output_final_invert/小檀栾室汇刻闺秀词.十集.附.闺秀词钞十六卷.补遗一卷.清.徐乃昌编.清末南陵徐乃昌刊本.黑白版.裁剪版.pdf",
    )
