# Ancient Book Cut Pipeline

古籍扫描件智能裁剪与压缩工具。

## 功能特性

- **PDF 批量提取**：多线程并行渲染，保留书签
- **跨页拆分**：自动识别并拆分双页扫描件（支持普通跨页和双框古籍两种模式）
- **反色检测**：自动识别墨渍/阴雨天扫描件并校正
- **智能版心裁剪**：三种模式适应不同版式
  - `smart`：多栏古籍首选
  - `union`：所有文字外接矩形
  - `largest`：单栏古籍
- **白边裁剪**：自动识别框线/白边模式
- **极限压缩**：PNG / JPG 多级压缩优化

## 安装依赖

```bash
pip install opencv-python numpy pymupdf pillow
```

## 快速开始

### Pipeline 一键处理

```python
from book_cut_pipeline import AncientBookPipeline

# 处理 PDF
pipeline = AncientBookPipeline(
    split_spread=True,
    split_mode="frame",
    invert=True,
    target_width=1264,
)
pipeline.from_pdf("input.pdf", "output.pdf")

# 或处理图片文件夹
pipeline = AncientBookPipeline(split_spread=True)
pipeline.from_images("./input_folder", "output.pdf")
```

### 单步处理

```python
# 1. PDF 提取为图片
from extractPdf import extract_images_from_pdf
extract_images_from_pdf("input.pdf")

# 2. 跨页拆分
from split_spread import batch_split
batch_split("./images", "./split", split_mode="frame")

# 3. 反色检测与校正
from invert_color import batch_invert_if_needed
batch_invert_if_needed("./split", "./inverted")

# 4. 版心裁剪与压缩
from ancient_book_engine import AncientBookEngine
engine = AncientBookEngine("./inverted", "./cropped")
engine.process()

# 5. 生成 PDF
from pic2pdf import images_to_pdf
images_to_pdf("./cropped", "output.pdf")
```

## 核心参数

### AncientBookPipeline

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `target_width` | 输出宽度 (px) | 1264-1680 |
| `target_height` | 输出高度 (px) | 1680-2240 |
| `crop_mode` | 版心裁剪模式 | "smart" |
| `split_spread` | 启用跨页拆分 | False |
| `split_mode` | 拆分模式 `"gradient"` / `"frame"` | "gradient" |
| `invert` | 反色控制 None(自动)/True/False | None |
| `book_type` | `"ancient"` 古籍 / `"simple"` 普通书籍 | "ancient" |
| `compress_format` | 输出格式 `"png"` / `"jpg"` | "png" |

### SimpleCropper（简版白边裁剪）

```python
from simple_crop import SimpleCropper

cropper = SimpleCropper(
    padding_ratio=0.02,       # 保留空白比例
    padding_min_pixel=25,     # 最小空白像素
    content_density_min=0.005, # 内容密度阈值
)
cropped = cropper.crop(image)  # 自动识别框线或白边
```

## 文件结构

| 文件 | 说明 |
|------|------|
| `book_cut_pipeline.py` | 主流水线入口 |
| `ancient_book_engine.py` | 古籍版心裁剪引擎 |
| `split_spread.py` | 双页拆分工具 |
| `extractPdf.py` | PDF 图片提取 |
| `invert_color.py` | 反色检测与校正 |
| `simple_crop.py` | 简版白边裁剪器 |
| `pic2pdf.py` | 图片转 PDF |
| `split_pdf.py` | PDF 按页拆分 |

## 算法设计

### 跨页拆分 - 双框检测模式

适用于两边都有框线的古籍扫描件：

```
1. 统计每列暗色像素比例
2. 左半边从右向左扫描，找右侧框线
3. 右半边从左向右扫描，找左侧框线
4. 切分位置 = (右边界 + 左边界) / 2
```

### 版心提取 - Smart 模式

```
1. 去噪 + 二值化
2. 闭运算合并相邻文字块
3. 连通域分析提取所有文字区域
4. Y轴投影分离页眉页脚
5. 提取最大正文块
```

### 框线/白边自动识别

```
1. 边缘区暗色像素统计
2. 四边都有 > 5% 暗色 → 有框线
3. 有框线 → 框线外侧裁切
4. 无框线 → 内容边界 + padding 裁切
```

## License

MIT
