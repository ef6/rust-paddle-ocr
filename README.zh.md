# Rust PaddleOCR

[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

一个基于PaddleOCR模型的轻量级高效OCR（光学字符识别）Rust库。该库利用MNN推理框架提供高性能的文本检测和识别功能，并提供完整的C API接口。

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## 特性

- **文本检测**：准确定位图像中的文本区域
- **文本识别**：识别检测区域中的文本内容
- **多版本模型支持**：支持 PP-OCRv4 和 PP-OCRv5 模型，灵活选择使用
- **多语言支持**：PP-OCRv5 支持简体中文、繁体中文、英文、日文、中文拼音等多种文字类型
- **复杂场景识别**：增强的手写体、竖排文本、生僻字识别能力
- **高性能**：使用MNN推理框架优化
- **最小依赖**：轻量级且易于集成
- **可自定义**：针对不同用例的可调整参数
- **命令行工具**：提供简单的命令行界面进行OCR识别
- **C API支持**：提供完整的C语言接口，支持跨语言调用
- **内存安全**：自动内存管理，防止内存泄漏

## 模型版本

该库支持三个PaddleOCR模型版本：

### PP-OCRv4
- **稳定版本**：经过充分验证，兼容性好
- **适用场景**：常规文档识别，对准确性要求较高的场景
- **模型文件**：
  - 检测模型：`ch_PP-OCRv4_det_infer.mnn`
  - 识别模型：`ch_PP-OCRv4_rec_infer.mnn`
  - 字符集：`ppocr_keys_v4.txt`

### PP-OCRv5 ⭐️ 推荐
- **最新版本**：新一代文字识别解决方案
- **多语言支持**：默认模型（`PP-OCRv5_mobile_rec.mnn`）支持简体中文、繁体中文、英文、日文、中文拼音
- **专用语言模型**：提供11+种语言的专用模型以获得最佳性能：
  - 阿拉伯语、西里尔语、梵文字母、希腊语、英语
  - 东斯拉夫语、韩语、拉丁语、泰米尔语、泰卢固语、泰语
- **共享检测模型**：所有V5语言模型使用相同的检测模型（`PP-OCRv5_mobile_det.mnn`）
- **增强场景识别**：
  - 中英复杂手写体识别能力显著提升
  - 竖排文本识别优化
  - 生僻字识别能力增强
- **性能提升**：相比PP-OCRv4端到端提升13个百分点
- **模型文件**（默认多语言）：
  - 检测模型：`PP-OCRv5_mobile_det.mnn`（所有语言共享）
  - 识别模型：`PP-OCRv5_mobile_rec.mnn`（默认，支持中文/英文/日文）
  - 字符集：`ppocr_keys_v5.txt`
- **专用语言模型文件**（可选）：
  - 识别模型：`{lang}_PP-OCRv5_mobile_rec_infer.mnn`
  - 字符集：`ppocr_keys_{lang}.txt`
  - 可用语言：`arabic`（阿拉伯语）、`cyrillic`（西里尔语）、`devanagari`（梵文字母）、`el`（希腊语）、`en`（英语）、`eslav`（东斯拉夫语）、`korean`（韩语）、`latin`（拉丁语）、`ta`（泰米尔语）、`te`（泰卢固语）、`th`（泰语）

### PP-OCRv5 FP16 ⭐️ 新增
- **高效版本**：在不牺牲准确率的情况下提供更快的推理速度和更低的内存使用
- **适用场景**：需要高性能和低内存使用的场景
- **性能提升**：
  - 推理速度提升约9%
  - 内存使用减少约8%
  - 模型大小减半
- **模型文件**：
  - 检测模型：`PP-OCRv5_mobile_det_fp16.mnn`
  - 识别模型：`PP-OCRv5_mobile_rec_fp16.mnn`
  - 字符集：`ppocr_keys_v5.txt`

### 模型性能对比

| 特性               | PP-OCRv4 | PP-OCRv5 | PP-OCRv5 FP16 |
|--------------------|----------|----------|---------------|
| 语言支持           | 中文、英文 | 多语言（默认支持中文/英文/日文，提供11+种专用语言模型） | 多语言（默认支持中文/英文/日文，提供11+种专用语言模型） |
| 文字类型支持       | 中文、英文 | 简体中文、繁体中文、英文、日文、中文拼音 | 简体中文、繁体中文、英文、日文、中文拼音 |
| 手写体识别         | 基础支持  | 显著增强  | 显著增强       |
| 竖排文本           | 基础支持  | 优化提升  | 优化提升       |
| 生僻字识别         | 有限支持  | 增强识别  | 增强识别       |
| 推理速度 (FPS)     | 1.1      | 1.2      | 1.2           |
| 内存使用 (峰值)    | 422.22MB | 388.41MB | 388.41MB      |
| 模型大小           | 标准      | 标准      | 减半           |
| 推荐场景           | 常规文档  | 复杂场景与多语言 | 高性能场景与多语言 |

### PP-OCRv5 FP16 测试数据

#### 标准模型
```
============================================================
测试报告: 推理速度测试
============================================================
总时间: 44.23s
平均推理时间: 884.64ms
最快推理时间: 744.99ms
最慢推理时间: 954.03ms
成功次数: 50
失败次数: 0
推理速度: 1.1 FPS
内存使用 - 开始: 14.94MB
内存使用 - 结束: 422.22MB
内存使用 - 峰值: 422.22MB
内存变化: +407.28MB
```

#### FP16 模型
```
============================================================
测试报告: 推理速度测试
============================================================
总时间: 43.33s
平均推理时间: 866.66ms
最快推理时间: 719.41ms
最慢推理时间: 974.93ms
成功次数: 50
失败次数: 0
推理速度: 1.2 FPS
内存使用 - 开始: 15.70MB
内存使用 - 结束: 388.41MB
内存使用 - 峰值: 388.41MB
内存变化: +372.70MB
```

### 测试方法

运行以下命令以执行测试并验证性能数据（基于Mac Mini M4）：

```bash
python test_ffi.py test
```

## 安装

在`Cargo.toml`中添加：

```toml
[dependencies.rust-paddle-ocr]
git = "https://github.com/zibo-chen/rust-paddle-ocr.git"

```

您也可以指定特定分支或标签：

```toml
[dependencies.rust-paddle-ocr]
git = "https://github.com/zibo-chen/rust-paddle-ocr.git"
branch = "main"
```

### 前提条件

该库需要：
- 转换为MNN格式的预训练PaddleOCR模型
- 用于文本识别的字符集文件

## 使用方式

### 作为Rust库使用

可以灵活选择使用 PP-OCRv4 或 PP-OCRv5 模型,只需加载不同模型文件即可：

```rust
use rust_paddle_ocr::{Det, Rec};
use image::open;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // === 使用 PP-OCRv5 模型（推荐） ===
    let mut det = Det::from_file("./models/PP-OCRv5_mobile_det.mnn")?;
    let mut rec = Rec::from_file(
        "./models/PP-OCRv5_mobile_rec.mnn", 
        "./models/ppocr_keys_v5.txt"
    )?;
    
    // === 或使用 PP-OCRv4 模型 ===
    // let mut det = Det::from_file("./models/ch_PP-OCRv4_det_infer.mnn")?;
    // let mut rec = Rec::from_file(
    //     "./models/ch_PP-OCRv4_rec_infer.mnn", 
    //     "./models/ppocr_keys_v4.txt"
    // )?;
    
    // 自定义检测参数（可选）
    let det = det
        .with_rect_border_size(12)  // PP-OCRv5 推荐参数
        .with_merge_boxes(false)    // PP-OCRv5 推荐参数
        .with_merge_threshold(1);   // PP-OCRv5 推荐参数
    
    // 自定义识别参数（可选）
    let rec = rec
        .with_min_score(0.6)
        .with_punct_min_score(0.1);
    
    // 打开图像
    let img = open("path/to/image.jpg")?;
    
    // 检测文本区域
    let text_images = det.find_text_img(&img)?;
    
    // 识别每个检测区域中的文本
    for text_img in text_images {
        let text = rec.predict_str(&text_img)?;
        println!("识别的文本: {}", text);
    }
    
    Ok(())
}
```

### 使用专用语言模型

对于特定语言，使用专用语言模型可以获得更好的识别准确度：

```rust
use rust_paddle_ocr::{Det, Rec};
use image::open;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 所有V5语言模型共享相同的检测模型
    let mut det = Det::from_file("./models/PP-OCRv5_mobile_det.mnn")?;
    
    // === 示例1: 英语专用模型 ===
    let mut rec_en = Rec::from_file(
        "./models/en_PP-OCRv5_mobile_rec_infer.mnn",
        "./models/ppocr_keys_en.txt"
    )?;
    
    // === 示例2: 韩语模型 ===
    let mut rec_ko = Rec::from_file(
        "./models/korean_PP-OCRv5_mobile_rec_infer.mnn",
        "./models/ppocr_keys_korean.txt"
    )?;
    
    // === 示例3: 阿拉伯语模型 ===
    let mut rec_ar = Rec::from_file(
        "./models/arabic_PP-OCRv5_mobile_rec_infer.mnn",
        "./models/ppocr_keys_arabic.txt"
    )?;
    
    // 处理图像
    let img = open("path/to/image.jpg")?;
    let text_images = det.find_text_img(&img)?;
    
    for text_img in text_images {
        let text = rec_en.predict_str(&text_img)?;
        println!("识别的文本: {}", text);
    }
    
    Ok(())
}
```

#### 可用的语言模型

| 语言 | 模型文件 | 字符集文件 | 适用场景 |
|------|---------|-----------|---------|
| 默认（多语言） | `PP-OCRv5_mobile_rec.mnn` | `ppocr_keys_v5.txt` | 中文、英文、日文（推荐通用使用） |
| 阿拉伯语 | `arabic_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_arabic.txt` | 阿拉伯语文本识别 |
| 西里尔语 | `cyrillic_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_cyrillic.txt` | 俄语、保加利亚语、塞尔维亚语等 |
| 梵文字母 | `devanagari_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_devanagari.txt` | 印地语、马拉地语、尼泊尔语等 |
| 希腊语 | `el_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_el.txt` | 希腊语文本识别 |
| 英语 | `en_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_en.txt` | 纯英文文档 |
| 东斯拉夫语 | `eslav_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_eslav.txt` | 乌克兰语、白俄罗斯语等 |
| 韩语 | `korean_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_korean.txt` | 韩语文本识别 |
| 拉丁语 | `latin_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_latin.txt` | 拉丁字母语言 |
| 泰米尔语 | `ta_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_ta.txt` | 泰米尔语文本识别 |
| 泰卢固语 | `te_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_te.txt` | 泰卢固语文本识别 |
| 泰语 | `th_PP-OCRv5_mobile_rec_infer.mnn` | `ppocr_keys_th.txt` | 泰语文本识别 |

**注意**：所有专用语言模型使用相同的检测模型（`PP-OCRv5_mobile_det.mnn`）。根据目标语言选择合适的识别模型可获得最佳准确度。

### 作为C库使用

该库提供了完整的C API接口，可以在C/C++项目中使用：

#### 编译C动态库

```bash
# 编译生成动态库
cargo build --release

# 生成的动态库位置（根据系统不同）：
# Linux: target/release/librust_paddle_ocr.so
# macOS: target/release/librust_paddle_ocr.dylib  
# Windows: target/release/rust_paddle_ocr.dll

# C头文件会自动生成到项目根目录：rocr.h
```

#### C API 使用示例

```c
#include "rocr.h"
#include <stdio.h>

int main() {
    // 获取版本信息
    printf("OCR库版本: %s\n", rocr_version());
    
    // 创建OCR引擎
    ROCR_RocrHandle engine = rocr_create_engine(
        "./models/PP-OCRv5_mobile_det.mnn",
        "./models/PP-OCRv5_mobile_rec.mnn", 
        "./models/ppocr_keys_v5.txt"
    );
    
    if (engine == 0) {
        printf("OCR引擎创建失败\n");
        return 1;
    }
    
    // 简单模式识别 - 只获取文本内容
    struct ROCR_RocrSimpleResult simple_result = 
        rocr_recognize_simple(engine, "./image.jpg");
    
    if (simple_result.STATUS == ROCR_RocrStatus_Success) {
        printf("识别出 %zu 个文本:\n", simple_result.COUNT);
        for (size_t i = 0; i < simple_result.COUNT; i++) {
            printf("- %s\n", simple_result.TEXTS[i]);
        }
    }
    
    // 释放简单结果内存
    rocr_free_simple_result(&simple_result);
    
    // 详细模式识别 - 获取文本和位置信息
    struct ROCR_RocrResult detailed_result = 
        rocr_recognize_detailed(engine, "./image.jpg");
    
    if (detailed_result.STATUS == ROCR_RocrStatus_Success) {
        printf("详细识别出 %zu 个文本框:\n", detailed_result.COUNT);
        for (size_t i = 0; i < detailed_result.COUNT; i++) {
            struct ROCR_RocrTextBox* box = &detailed_result.BOXES[i];
            printf("文本: %s\n", box->TEXT);
            printf("置信度: %.2f\n", box->CONFIDENCE);
            printf("位置: (%d, %d, %u, %u)\n", 
                   box->LEFT, box->TOP, box->WIDTH, box->HEIGHT);
        }
    }
    
    // 释放详细结果内存
    rocr_free_result(&detailed_result);
    
    // 销毁引擎
    rocr_destroy_engine(engine);
    
    // 清理所有资源
    rocr_cleanup();
    
    return 0;
}
```

#### 编译和运行C Demo

项目提供了完整的C语言演示程序：

```bash
# 进入demo目录
cd demo

# 编译C demo（Linux/macOS）
gcc -o c_demo c_demo.c -L../target/release -lrust_paddle_ocr -ldl

# 运行demo
./c_demo

# Windows编译示例
# gcc -o c_demo.exe c_demo.c -L../target/release -lrust_paddle_ocr -lws2_32 -luserenv
```

#### C API 高级配置

```c
// 使用自定义配置创建引擎
ROCR_RocrHandle engine = rocr_create_engine_with_config(
    det_model_path,
    rec_model_path, 
    keys_path,
    12,    // rect_border_size - 边界框扩展尺寸
    0,     // merge_boxes - 是否合并文本框 (0=false, 1=true)
    1      // merge_threshold - 合并阈值
);

// 使用内存中的模型数据创建引擎
ROCR_RocrHandle engine = rocr_create_engine_with_bytes(
    det_model_data, det_model_size,
    rec_model_data, rec_model_size,
    keys_data, keys_size,
    12, 0, 1
);
```

## 命令行工具

该库提供了一个内置的命令行工具，可以直接进行OCR识别：

```bash
# 基本用法
./ocr -p path/to/image.jpg

# 输出JSON格式（包含详细信息和位置）
./ocr -p path/to/image.jpg -m json

# 显示详细日志信息
./ocr -p path/to/image.jpg -v

# 显示当前使用的模型版本
./ocr --version-info
```

### 编译不同版本

```bash
# 编译使用 PP-OCRv4 模型的版本（默认）
cargo build --release

# 编译使用 PP-OCRv5 模型的版本（推荐）
cargo build --release --features v5
```

### 命令行选项

```
选项:
  -p, --path <IMAGE_PATH>  要识别的图像路径
  -s, --server             交互模式
  -m, --mode <MODE>        输出模式: json(详细) 或 text(简单) [默认: text]
  -v, --verbose            是否显示详细日志
      --version-info       显示模型版本信息
  -h, --help               显示帮助信息
  -V, --version            显示版本信息
```

## 模型文件获取

您可以从以下渠道获取预训练的MNN模型：

1. **官方模型**：从 PaddleOCR 官方仓库下载并转换为 MNN 格式
2. **项目提供**：本项目的 `models/` 目录包含了转换好的模型文件

## PP-OCRv5 vs PP-OCRv4 性能对比

| 特性 | PP-OCRv4 | PP-OCRv5 |
|------|----------|----------|
| 文字类型支持 | 中文、英文 | 简体中文、繁体中文、英文、日文、中文拼音 |
| 手写体识别 | 基础支持 | 显著增强 |
| 竖排文本 | 基础支持 | 优化提升 |
| 生僻字识别 | 有限支持 | 增强识别 |
| 端到端准确率 | 基准 | 提升 13% |
| 推荐场景 | 常规文档 | 复杂多样场景 |

## API 参考

### Rust API

#### 文本检测 (Det)

```rust
// 创建新的检测器
let mut det = Det::from_file("path/to/det_model.mnn")?;

// 查找文本区域并返回矩形
let rects = det.find_text_rect(&img)?;

// 查找文本区域并返回裁剪后的图像
let text_images = det.find_text_img(&img)?;

// 自定义选项
let det = det
    .with_rect_border_size(12)
    .with_merge_boxes(false)
    .with_merge_threshold(1);
```

#### 文本识别 (Rec)

```rust
// 创建新的识别器
let mut rec = Rec::from_file("path/to/rec_model.mnn", "path/to/keys.txt")?;

// 识别文本并返回字符串
let text = rec.predict_str(&text_img)?;

// 识别文本并返回带有置信度分数的字符
let char_scores = rec.predict_char_score(&text_img)?;

// 自定义选项
let rec = rec
    .with_min_score(0.6)           // 设置普通字符的最低置信度
    .with_punct_min_score(0.1);    // 设置标点符号的最低置信度
```

### C API

#### 核心函数

```c
// 引擎管理
ROCR_RocrHandle rocr_create_engine(const char* det_model, 
                                   const char* rec_model, 
                                   const char* keys_file);
ROCR_RocrHandle rocr_create_engine_with_config(...);
ROCR_RocrHandle rocr_create_engine_with_bytes(...);
enum ROCR_RocrStatus rocr_destroy_engine(ROCR_RocrHandle handle);

// 文本识别
struct ROCR_RocrSimpleResult rocr_recognize_simple(ROCR_RocrHandle handle, 
                                                   const char* image_path);
struct ROCR_RocrResult rocr_recognize_detailed(ROCR_RocrHandle handle, 
                                               const char* image_path);

// 内存管理
void rocr_free_simple_result(struct ROCR_RocrSimpleResult* result);
void rocr_free_result(struct ROCR_RocrResult* result);
void rocr_cleanup(void);

// 工具函数
const char* rocr_version(void);
```

#### 数据结构

```c
// 状态码
typedef enum ROCR_RocrStatus {
    ROCR_RocrStatus_Success = 0,
    ROCR_RocrStatus_InitError = 1,
    ROCR_RocrStatus_FileNotFound = 2,
    ROCR_RocrStatus_ImageLoadError = 3,
    ROCR_RocrStatus_ProcessError = 4,
    ROCR_RocrStatus_MemoryError = 5,
    ROCR_RocrStatus_InvalidParam = 6,
    ROCR_RocrStatus_NotInitialized = 7,
} ROCR_RocrStatus;

// 文本框
typedef struct ROCR_RocrTextBox {
    char* TEXT;              // 识别的文本
    float CONFIDENCE;        // 置信度 (0.0-1.0)
    int LEFT;               // 左边界
    int TOP;                // 上边界  
    unsigned int WIDTH;     // 宽度
    unsigned int HEIGHT;    // 高度
} ROCR_RocrTextBox;

// 详细结果
typedef struct ROCR_RocrResult {
    enum ROCR_RocrStatus STATUS;     // 状态码
    size_t COUNT;                    // 文本框数量
    struct ROCR_RocrTextBox* BOXES;  // 文本框数组
} ROCR_RocrResult;

// 简单结果
typedef struct ROCR_RocrSimpleResult {
    enum ROCR_RocrStatus STATUS;     // 状态码
    size_t COUNT;                    // 文本数量
    char** TEXTS;                    // 文本数组
} ROCR_RocrSimpleResult;
```

#### 内存管理注意事项

1. **结果释放**：必须调用相应的释放函数释放结果内存
2. **引擎销毁**：使用完毕后必须销毁引擎实例
3. **全局清理**：程序结束前调用 `rocr_cleanup()` 清理所有资源
4. **线程安全**：引擎实例不是线程安全的，多线程使用需要额外同步

## 演示程序

项目在 `demo/` 目录下提供了完整的演示程序：

- **C Demo** (`demo/c_demo.c`)：完整的C语言调用示例，展示简单和详细模式的使用
- **模型文件**：`models/` 目录包含示例模型文件
- **测试图片**：`res/` 目录包含测试图片

运行演示：
```bash
# 进入demo目录并运行
cd demo && ./c_demo
```

## 许可证

该项目采用Apache许可证2.0版 - 详情请参阅[LICENSE](LICENSE)文件。

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 提供原始OCR模型和研究
- [MNN](https://github.com/alibaba/MNN) - 提供高效的神经网络推理框架
- [mnn-rs](https://github.com/aftershootco/mnn-rs) - 为Rust提供了MNN绑定
