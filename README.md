# 📦 OCR TensorRT 部署 & 推理说明

本项目提供 OCR 模型在 **TensorRT** 上的部署方案，支持 **RTX 5070 Ti / sm_120** 架构，包含：
- 运行推理所需的文件
- 重新构建 TensorRT 引擎的方法
- 推理运行示例

---

## 📂 运行必需（推理阶段用）
> 在另一台电脑直接运行 TensorRT 推理时必须携带的文件。

| 文件名            | 类型               | 用途说明 |
|-------------------|--------------------|----------|
| `det_fp16.engine` | TensorRT Engine    | OCR 检测模型（文本框检测）编译好的 FP16 引擎，直接加载推理 |
| `rec_fp16.engine` | TensorRT Engine    | OCR 识别模型（文字识别）编译好的 FP16 引擎，直接加载推理 |
| `japan_dict.txt`  | 字典文件           | 识别阶段的字符映射表（CTC 解码用）。不带的话输出为字符索引而不是日文/英文 |

---

## 📂 可选携带（重建/调试用）
> 仅推理可不带；如需重新构建 TensorRT 引擎则必须携带。

| 文件名                  | 类型                    | 用途说明 |
|-------------------------|-------------------------|----------|
| `det_dbpp.onnx`         | ONNX 模型               | Paddle 检测模型导出的 ONNX 版（重建 TensorRT 引擎用） |
| `rec_japan.onnx`        | ONNX 模型               | Paddle 识别模型导出的 ONNX 版（重建 TensorRT 引擎用） |
| `trt_cache.cache`       | TensorRT Timing Cache   | 构建引擎的性能调优缓存（同架构 GPU 上可大幅加快重建速度） |
| `trt_cache.cache.lock`  | Cache 锁文件            | 构建时生成的锁文件，无实际运行意义，可不带 |

---

## 📂 Paddle 原始模型（仅 Paddle 推理或再次导出 ONNX 时用）
> 使用 `paddle2onnx` 导出 ONNX 时，就是从这些目录导出的。

| 文件名/文件夹                    | 类型                | 用途说明 |
|----------------------------------|---------------------|----------|
| `ch_PP-OCRv4_det_infer/`         | Paddle 模型目录     | 中文+多语言检测模型（包含 `.pdmodel` 和 `.pdiparams`） |
| `japan_PP-OCRv3_rec_infer/`      | Paddle 模型目录     | 日文识别模型（包含 `.pdmodel` 和 `.pdiparams`） |
| `ch_PP-OCRv4_det_infer.tar`      | 压缩包              | `ch_PP-OCRv4_det_infer/` 的打包文件 |
| `japan_PP-OCRv3_rec_infer.tar`   | 压缩包              | `japan_PP-OCRv3_rec_infer/` 的打包文件 |

---

## 🚀 构建 TensorRT 引擎（支持 RTX 5070 Ti / sm_120）

以下步骤使用 `trtexec` 将 ONNX 模型转换为 FP16 TensorRT 引擎，并生成/复用 Timing Cache  
（路径请根据你的实际情况修改）

### 1️⃣ 路径准备
```powershell
# TensorRT 可执行文件路径
$trt = 'D:\desk\TensorRT-10.13.2.6\bin\trtexec.exe'

# 模型存放路径
$models = 'C:\Users\musika\models'
### 2️⃣ 构建检测引擎（det_dbpp.onnx → det_fp16.engine）
```powershell
& $trt `
  --onnx="$models\det_dbpp.onnx" `
  --saveEngine="$models\det_fp16.engine" `
  --fp16 `
  --minShapes=x:1x3x640x640 `
  --optShapes=x:4x3x960x960 `
  --maxShapes=x:8x3x1280x1280 `
  --memPoolSize=workspace:4096M `
  --timingCacheFile="$models\trt_cache.cache"

### 3️⃣ 构建识别引擎（rec_japan.onnx → rec_fp16.engine）
会复用/追加同一份 trt_cache.cache，加快构建速度。

```powershell
& $trt `
  --onnx="$models\rec_japan.onnx" `
  --saveEngine="$models\rec_fp16.engine" `
  --fp16 `
  --minShapes=x:1x3x48x80 `
  --optShapes=x:8x3x48x320 `
  --maxShapes=x:16x3x48x640 `
  --memPoolSize=workspace:4096M `
  --timingCacheFile="$models\trt_cache.cache"
###▶️ 推理运行示例
以下示例假设你已有 det_fp16.engine、rec_fp16.engine、japan_dict.txt，并在脚本中调用 TensorRT 进行 OCR 推理。

```python
from trt_ocr import OCRTRT

# 初始化 OCRTRT
ocr = OCRTRT(
    det_engine_path="models/det_fp16.engine",
    rec_engine_path="models/rec_fp16.engine",
    dict_path="models/japan_dict.txt"
)

# 运行推理
results = ocr.run("test.jpg")

# 打印结果
for box, text, score in results:
    print(f"[{score:.2f}] {text} - {box}")
运行命令：

```bash
python run_ocr.py --image test.jpg
