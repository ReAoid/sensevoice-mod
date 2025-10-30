# SenseVoice + 3D-Speaker 集成系统使用指南

## 📖 目录

- [系统简介](#系统简介)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [环境安装](#环境安装)
- [快速开始](#快速开始)
- [详细使用教程](#详细使用教程)
  - [1. 声纹注册](#1-声纹注册)
  - [2. 语音识别](#2-语音识别)
  - [3. 说话人识别](#3-说话人识别)
  - [4. 切换功能开关](#4-切换功能开关)
- [命令行使用](#命令行使用)
- [Web UI使用](#web-ui使用)
- [配置文件说明](#配置文件说明)
- [API接口](#api接口)
- [常见问题](#常见问题)
- [性能优化](#性能优化)
- [技术细节](#技术细节)

---

## 系统简介

本系统将 **3D-Speaker（说话人识别）** 和 **SenseVoice（语音识别）** 两个强大的模型集成在一起，提供完整的语音理解解决方案。

### 核心能力

- 🎯 **多语言语音识别**：支持中文、英文、粤语、日语、韩语
- 😊 **情感识别**：识别7种情绪（开心、悲伤、愤怒、中性、害怕、厌恶、惊讶）
- 🎵 **事件检测**：检测音乐、掌声、笑声、哭声、咳嗽等声音事件
- 👤 **说话人识别**：识别和验证说话人身份
- 🔄 **灵活切换**：可以独立使用或组合使用各项功能

---

## 功能特性

### ✨ SenseVoice 功能

| 功能 | 描述 | 支持语言 |
|------|------|---------|
| **语音识别** | 高精度多语言ASR | 中、英、粤、日、韩 |
| **情感识别** | 7种情绪分类 | 所有语言 |
| **事件检测** | 8种常见声音事件 | 所有语言 |
| **文本规范化** | 自动添加标点符号 | 中、英 |
| **低延迟** | 10s音频仅需70ms | - |

### 🎙️ 3D-Speaker 功能

| 功能 | 描述 | 模型选择 |
|------|------|---------|
| **声纹注册** | 注册说话人声纹特征 | CAM++, ERes2Net, ERes2NetV2 |
| **说话人识别** | 识别已注册的说话人 | 同上 |
| **相似度计算** | 计算声纹相似度 | 同上 |
| **批量注册** | 批量注册多个说话人 | 同上 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    集成ASR系统                            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌────────────────┐         ┌──────────────────┐        │
│  │  3D-Speaker    │         │   SenseVoice     │        │
│  │  说话人识别     │         │   语音识别        │        │
│  ├────────────────┤         ├──────────────────┤        │
│  │ • 声纹提取      │         │ • 多语言ASR       │        │
│  │ • 特征匹配      │         │ • 情感识别        │        │
│  │ • 身份验证      │         │ • 事件检测        │        │
│  └────────────────┘         └──────────────────┘        │
│           ↓                          ↓                   │
│  ┌─────────────────────────────────────────┐            │
│  │          统一处理流程                      │            │
│  │  1. 音频输入                               │            │
│  │  2. 说话人识别 (可选)                       │            │
│  │  3. 语音识别                               │            │
│  │  4. 结果输出                               │            │
│  └─────────────────────────────────────────┘            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 环境安装

### 系统要求

- Python >= 3.8
- CUDA >= 11.0 (推荐，用于GPU加速)
- 8GB+ RAM (CPU模式)
- 4GB+ GPU显存 (GPU模式)

### 安装步骤

#### 1. 克隆项目

```bash
cd /Users/mingy/Documents/python/SenseVoice
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果缺少某些依赖，手动安装：

```bash
pip install torch torchaudio
pip install modelscope funasr gradio
pip install pyyaml numpy scipy scikit-learn
pip install soundfile librosa
```

#### 3. 验证安装

```bash
# 测试SenseVoice
python -c "from funasr import AutoModel; print('SenseVoice OK')"

# 测试3D-Speaker
python -c "from speakerlab.utils.builder import dynamic_import; print('3D-Speaker OK')"
```

---

## 快速开始

### 方式一：Web UI（推荐新手）

```bash
# 启动Web界面
python webui_integrated.py
```

浏览器打开 http://localhost:7860

### 方式二：命令行

#### 基础语音识别（不使用说话人识别）

```bash
python integrated_asr.py --audio test.wav
```

#### 启用说话人识别

```bash
python integrated_asr.py --audio test.wav --enable-speaker
```

### 方式三：Python API

```python
from integrated_asr import IntegratedASRSystem

# 创建系统
system = IntegratedASRSystem(enable_speaker_verification=False)

# 处理音频
result = system.process_audio("test.wav", language="auto")
print(result['asr_results'][0]['text'])
```

---

## 详细使用教程

### 1. 声纹注册

#### 1.1 为什么要注册声纹？

在使用说话人识别功能之前，需要先注册说话人的声纹特征。系统会提取音频中的声纹特征并保存到声纹库中。

#### 1.2 注册单个声纹

**命令行方式：**

```bash
python register_voiceprint.py \
  --action register \
  --speaker-id user001 \
  --speaker-name "张三" \
  --audio voice_samples/zhangsan.wav
```

**参数说明：**
- `--speaker-id`: 说话人唯一ID（必需，不可重复）
- `--speaker-name`: 说话人名称（可选，便于识别）
- `--audio`: 音频文件路径（建议3-10秒纯净人声）

**Web UI方式：**

1. 打开Web界面
2. 切换到"声纹管理"标签页
3. 填写说话人ID和名称
4. 上传音频文件
5. 点击"注册声纹"按钮
6. 等待注册完成

#### 1.3 批量注册声纹

如果有多个音频文件需要注册：

```bash
# 准备目录结构
voice_samples/
  ├── zhangsan.wav
  ├── lisi.wav
  └── wangwu.wav

# 批量注册（文件名作为speaker_id）
python register_voiceprint.py \
  --action batch-register \
  --audio-dir voice_samples \
  --pattern "*.wav"
```

#### 1.4 注册最佳实践

✅ **推荐做法：**
- 使用3-10秒的纯净人声
- 在安静环境录制
- 包含自然的语调变化
- WAV或MP3格式，16kHz采样率

❌ **避免：**
- 音频太短（<2秒）
- 背景噪音太大
- 音量太小或失真
- 多人同时说话

#### 1.5 查看已注册的声纹

```bash
# 列出所有已注册的声纹
python register_voiceprint.py --action list
```

输出示例：
```
============================================================
说话人ID: user001
说话人名称: 张三
音频文件: voice_samples/zhangsan.wav
注册时间: 2024-10-30T10:30:00
模型类型: eres2net

============================================================
说话人ID: user002
说话人名称: 李四
音频文件: voice_samples/lisi.wav
注册时间: 2024-10-30T10:35:00
模型类型: eres2net

============================================================
[INFO] 共有 2 个已注册的说话人
```

#### 1.6 删除声纹

```bash
python register_voiceprint.py \
  --action unregister \
  --speaker-id user001
```

---

### 2. 语音识别

#### 2.1 基础语音识别（不使用说话人识别）

**命令行：**

```bash
python integrated_asr.py --audio test.wav
```

**Python API：**

```python
from integrated_asr import IntegratedASRSystem

# 创建系统（关闭说话人识别）
system = IntegratedASRSystem(enable_speaker_verification=False)

# 识别
result = system.process_audio("test.wav", language="auto")

# 查看结果
for item in result['asr_results']:
    print(item['text'])
```

#### 2.2 指定语言识别

```bash
# 中文
python integrated_asr.py --audio test.wav --language zh

# 英文
python integrated_asr.py --audio test.wav --language en

# 粤语
python integrated_asr.py --audio test.wav --language yue
```

支持的语言：`auto`, `zh`, `en`, `yue`, `ja`, `ko`

#### 2.3 批量处理多个文件

```bash
python integrated_asr.py \
  --audio-list file1.wav file2.wav file3.wav \
  --output results.json
```

结果会保存到 `results.json` 文件中。

#### 2.4 识别结果解析

识别结果包含：
- **文本内容**：识别的文字
- **情感标签**：😊😔😡等表情
- **事件标签**：🎼👏😀等事件

示例输出：
```
🎼很高兴认识你😊
👏大家好，我是张三
这是一个测试音频😔
```

---

### 3. 说话人识别

#### 3.1 识别已注册的说话人

**前提条件：** 已经注册了至少一个说话人的声纹

**命令行：**

```bash
python integrated_asr.py \
  --audio test.wav \
  --enable-speaker
```

**Python API：**

```python
from integrated_asr import IntegratedASRSystem

# 创建系统（启用说话人识别）
system = IntegratedASRSystem(enable_speaker_verification=True)

# 处理音频
result = system.process_audio("test.wav")

# 查看说话人信息
if result['speaker_info']:
    print(f"说话人: {result['speaker_info']['speaker_name']}")
    print(f"相似度: {result['speaker_info']['similarity']:.4f}")
else:
    print("未识别到注册的说话人")

# 查看识别文本
for item in result['asr_results']:
    print(item['text'])
```

#### 3.2 调整识别阈值

识别阈值决定了多相似才算匹配成功。阈值范围：0-1

- **0.3-0.4**：宽松，容易匹配，可能误识别
- **0.5-0.6**：适中（默认0.5）
- **0.7-0.9**：严格，不易匹配，更准确

**命令行调整：**

```bash
python integrated_asr.py \
  --audio test.wav \
  --enable-speaker \
  --speaker-threshold 0.7
```

**配置文件调整：**

编辑 `config_3dspeaker.yaml`：

```yaml
speaker_verification:
  threshold: 0.7  # 修改这里
```

#### 3.3 识别流程详解

```
输入音频 → 提取声纹特征 → 与库中所有声纹对比 → 计算相似度 → 判断是否超过阈值
```

如果最高相似度 >= 阈值，返回匹配的说话人信息，否则返回 None。

---

### 4. 切换功能开关

#### 4.1 动态启用/禁用说话人识别

**Python API：**

```python
from integrated_asr import IntegratedASRSystem

# 创建系统
system = IntegratedASRSystem(enable_speaker_verification=False)

# 处理第一个音频（不识别说话人）
result1 = system.process_audio("test1.wav")

# 启用说话人识别
system.toggle_speaker_verification(True)

# 处理第二个音频（识别说话人）
result2 = system.process_audio("test2.wav")

# 禁用说话人识别
system.toggle_speaker_verification(False)

# 处理第三个音频（不识别说话人）
result3 = system.process_audio("test3.wav")
```

#### 4.2 配置文件控制

编辑 `config_3dspeaker.yaml`：

```yaml
speaker_verification:
  # 修改这里来控制默认是否启用
  enabled: true   # true=启用, false=禁用
```

#### 4.3 Web UI中切换

在Web界面的"语音识别"标签页：
- 勾选"启用说话人识别"复选框 → 启用
- 取消勾选 → 禁用

---

## 命令行使用

### 声纹管理命令

#### 注册声纹

```bash
# 基础注册
python register_voiceprint.py \
  --action register \
  --speaker-id user001 \
  --speaker-name "张三" \
  --audio voice.wav

# 指定模型类型
python register_voiceprint.py \
  --action register \
  --speaker-id user001 \
  --speaker-name "张三" \
  --audio voice.wav \
  --model-type campplus

# 使用CPU
python register_voiceprint.py \
  --action register \
  --speaker-id user001 \
  --speaker-name "张三" \
  --audio voice.wav \
  --device cpu
```

#### 批量注册

```bash
python register_voiceprint.py \
  --action batch-register \
  --audio-dir ./voice_samples \
  --pattern "*.wav"
```

#### 识别说话人

```bash
python register_voiceprint.py \
  --action identify \
  --audio test.wav \
  --threshold 0.5
```

#### 列出所有声纹

```bash
python register_voiceprint.py --action list
```

#### 删除声纹

```bash
python register_voiceprint.py \
  --action unregister \
  --speaker-id user001
```

### 集成识别命令

#### 基础语音识别

```bash
# 自动语言识别
python integrated_asr.py --audio test.wav

# 指定语言
python integrated_asr.py --audio test.wav --language zh

# 保存结果
python integrated_asr.py --audio test.wav --output result.json
```

#### 启用说话人识别

```bash
# 基础使用
python integrated_asr.py --audio test.wav --enable-speaker

# 调整阈值
python integrated_asr.py \
  --audio test.wav \
  --enable-speaker \
  --speaker-threshold 0.7

# 指定模型
python integrated_asr.py \
  --audio test.wav \
  --enable-speaker \
  --speaker-model-type campplus
```

#### 批量处理

```bash
# 处理多个文件
python integrated_asr.py \
  --audio-list file1.wav file2.wav file3.wav \
  --enable-speaker \
  --output batch_results.json

# 使用通配符（需要shell支持）
python integrated_asr.py \
  --audio-list *.wav \
  --enable-speaker \
  --output results.json
```

#### 高级选项

```bash
python integrated_asr.py \
  --audio test.wav \
  --enable-speaker \
  --language zh \
  --use-itn \
  --speaker-threshold 0.6 \
  --speaker-model-type eres2net \
  --device cuda:0 \
  --output result.json
```

### 完整参数列表

#### register_voiceprint.py 参数

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--action` | 操作类型 | 是 | - |
| `--speaker-id` | 说话人ID | 部分 | - |
| `--speaker-name` | 说话人名称 | 否 | speaker-id |
| `--audio` | 音频路径 | 部分 | - |
| `--audio-dir` | 音频目录 | 部分 | - |
| `--pattern` | 文件模式 | 否 | `*.wav` |
| `--voiceprint-dir` | 声纹库目录 | 否 | `./voiceprint_db` |
| `--model-dir` | 模型目录 | 否 | `./pretrained_models` |
| `--model-type` | 模型类型 | 否 | `eres2net` |
| `--device` | 计算设备 | 否 | 自动选择 |
| `--threshold` | 识别阈值 | 否 | `0.5` |

action 可选值：
- `register`: 注册单个声纹
- `batch-register`: 批量注册
- `unregister`: 注销声纹
- `list`: 列出所有声纹
- `identify`: 识别说话人

#### integrated_asr.py 参数

| 参数 | 说明 | 必需 | 默认值 |
|------|------|------|--------|
| `--audio` | 音频文件 | 部分 | - |
| `--audio-list` | 音频列表 | 部分 | - |
| `--enable-speaker` | 启用说话人识别 | 否 | False |
| `--voiceprint-dir` | 声纹库目录 | 否 | `./voiceprint_db` |
| `--speaker-model-dir` | 模型目录 | 否 | `./pretrained_models` |
| `--speaker-model-type` | 模型类型 | 否 | `eres2net` |
| `--speaker-threshold` | 识别阈值 | 否 | `0.5` |
| `--sensevoice-model` | SenseVoice模型 | 否 | `iic/SenseVoiceSmall` |
| `--vad-model` | VAD模型 | 否 | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| `--language` | 语言 | 否 | `auto` |
| `--use-itn` | 文本规范化 | 否 | True |
| `--device` | 计算设备 | 否 | `cuda:0` |
| `--output` | 输出文件 | 否 | - |
| `--show-info` | 显示系统信息 | 否 | False |

---

## Web UI使用

### 启动Web界面

```bash
python webui_integrated.py
```

启动后，浏览器会自动打开 http://localhost:7860

### 界面功能介绍

#### 1. 语音识别标签页

**功能：** 进行语音识别、情感识别和事件检测

**使用步骤：**
1. 上传音频文件或使用麦克风录音
2. 选择语言（建议auto）
3. 选择是否启用说话人识别
4. 选择是否使用文本规范化
5. 点击"开始识别"按钮
6. 查看识别结果和说话人信息

**界面元素：**
- 音频输入：支持拖拽上传或点击上传
- 语言选择：下拉菜单
- 说话人识别开关：复选框
- 文本规范化开关：复选框
- 识别按钮：开始处理
- 结果显示：文本框显示结果

#### 2. 声纹管理标签页

**功能：** 注册、查看和删除说话人声纹

**注册声纹：**
1. 填写说话人ID（必填）
2. 填写说话人名称（可选）
3. 上传声纹音频
4. 点击"注册声纹"按钮
5. 查看注册结果

**管理声纹：**
- 点击"查看所有声纹"按钮：列出所有已注册的声纹
- 填写ID并点击"删除声纹"：删除指定声纹

#### 3. 使用说明标签页

**功能：** 查看完整的使用教程和常见问题

包含：
- 基础使用教程
- 命令行使用说明
- 高级配置说明
- 常见问题解答

### Web UI配置

编辑 `config_3dspeaker.yaml` 的 webui 部分：

```yaml
webui:
  port: 7860              # 端口号
  server_name: "0.0.0.0"  # 监听地址
  share: false            # 是否创建公开链接
  theme: "soft"           # 界面主题
  max_concurrent: 5       # 最大并发数
```

### 自定义主题

支持的主题：
- `default`: 默认主题
- `soft`: 柔和主题（推荐）
- `huggingface`: HuggingFace主题
- `grass`: 草绿主题
- `peach`: 桃红主题

修改配置文件：
```yaml
webui:
  theme: "soft"  # 修改这里
```

---

## 配置文件说明

配置文件：`config_3dspeaker.yaml`

### 完整配置示例

```yaml
# 说话人识别配置
speaker_verification:
  enabled: false                                    # 默认是否启用
  voiceprint_dir: "./voiceprint_db"                # 声纹库目录
  model_dir: "./pretrained_models"                 # 模型目录
  model_type: "eres2net"                           # 模型类型
  threshold: 0.5                                   # 识别阈值

# SenseVoice配置
sensevoice:
  model: "iic/SenseVoiceSmall"                     # 模型名称
  vad_model: "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
  use_itn: true                                    # 文本规范化
  batch_size_s: 60                                 # 批处理大小
  default_language: "auto"                         # 默认语言

# 系统配置
system:
  device: "cuda:0"                                 # 计算设备
  log_level: "INFO"                                # 日志级别
  output_format: "json"                            # 输出格式

# Web UI配置
webui:
  port: 7860                                       # 端口
  server_name: "0.0.0.0"                          # 地址
  share: false                                     # 公开链接
  theme: "soft"                                    # 主题
```

### 配置项详解

#### speaker_verification 配置

| 配置项 | 说明 | 可选值 |
|-------|------|--------|
| `enabled` | 默认启用状态 | `true`, `false` |
| `voiceprint_dir` | 声纹数据库目录 | 任意路径 |
| `model_dir` | 模型缓存目录 | 任意路径 |
| `model_type` | 模型类型 | `campplus`, `eres2net`, `eres2netv2` |
| `threshold` | 识别阈值 | 0.0 - 1.0 |

#### sensevoice 配置

| 配置项 | 说明 | 可选值 |
|-------|------|--------|
| `model` | SenseVoice模型 | modelscope模型ID或本地路径 |
| `vad_model` | VAD模型 | modelscope模型ID或本地路径 |
| `use_itn` | 文本规范化 | `true`, `false` |
| `batch_size_s` | 批处理大小（秒） | 正整数 |
| `default_language` | 默认语言 | `auto`, `zh`, `en`, `yue`, `ja`, `ko` |

#### system 配置

| 配置项 | 说明 | 可选值 |
|-------|------|--------|
| `device` | 计算设备 | `cuda:0`, `cuda:1`, `cpu` |
| `log_level` | 日志级别 | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `output_format` | 输出格式 | `json`, `txt` |

---

## API接口

### Python API

#### 1. 创建系统实例

```python
from integrated_asr import IntegratedASRSystem

# 基础创建
system = IntegratedASRSystem()

# 启用说话人识别
system = IntegratedASRSystem(enable_speaker_verification=True)

# 完整参数
system = IntegratedASRSystem(
    enable_speaker_verification=True,
    voiceprint_dir="./voiceprint_db",
    speaker_model_dir="./pretrained_models",
    speaker_model_type="eres2net",
    speaker_threshold=0.5,
    sensevoice_model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    device="cuda:0"
)
```

#### 2. 处理单个音频

```python
# 基础使用
result = system.process_audio("test.wav")

# 指定参数
result = system.process_audio(
    audio_path="test.wav",
    language="zh",
    use_itn=True,
    identify_speaker=True
)

# 结果结构
{
    'audio_path': 'test.wav',
    'speaker_info': {
        'speaker_id': 'user001',
        'speaker_name': '张三',
        'similarity': 0.856
    },
    'asr_results': [
        {'text': '这是识别的文本😊'}
    ],
    'processing_time': 1.23
}
```

#### 3. 批量处理

```python
audio_list = ['file1.wav', 'file2.wav', 'file3.wav']

results = system.batch_process(
    audio_list=audio_list,
    language="auto",
    use_itn=True,
    output_file="results.json"
)
```

#### 4. 切换功能

```python
# 启用说话人识别
system.toggle_speaker_verification(True)

# 禁用说话人识别
system.toggle_speaker_verification(False)
```

#### 5. 获取系统信息

```python
info = system.get_system_info()
print(info)

# 输出示例
{
    'speaker_verification_enabled': True,
    'speaker_threshold': 0.5,
    'device': 'cuda:0',
    'registered_speakers_count': 3,
    'registered_speakers': ['user001', 'user002', 'user003']
}
```

### 声纹管理API

```python
from register_voiceprint import VoiceprintManager

# 创建管理器
manager = VoiceprintManager(
    voiceprint_dir="./voiceprint_db",
    model_dir="./pretrained_models",
    model_type="eres2net"
)

# 注册声纹
success = manager.register_speaker(
    speaker_id="user001",
    audio_path="voice.wav",
    speaker_name="张三"
)

# 识别说话人
result = manager.identify_speaker(
    audio_path="test.wav",
    threshold=0.5
)

# 列出所有声纹
speakers = manager.list_speakers(verbose=True)

# 删除声纹
manager.unregister_speaker("user001")

# 批量注册
manager.batch_register(
    audio_dir="./voice_samples",
    pattern="*.wav"
)
```

---

## 常见问题

### Q1: 首次运行很慢，怎么办？

**A:** 首次运行需要从ModelScope下载模型文件，这是正常现象。

**解决方案：**
- 耐心等待下载完成
- 下载后的模型会缓存，后续使用会很快
- 如果下载失败，可以手动下载模型：
  ```bash
  # 设置ModelScope缓存目录
  export MODELSCOPE_CACHE=~/.cache/modelscope
  
  # 手动下载模型（可选）
  modelscope download --model iic/SenseVoiceSmall
  modelscope download --model iic/speech_eres2net_sv_zh-cn_16k-common
  ```

### Q2: 说话人识别准确率不高？

**A:** 可能的原因和解决方案：

**原因1: 注册音频质量差**
- ✅ 使用3-10秒纯净人声
- ✅ 在安静环境录制
- ✅ 避免背景噪音

**原因2: 测试音频与注册音频差异大**
- ✅ 使用相似的录音设备
- ✅ 保持相似的说话方式
- ✅ 避免声音失真

**原因3: 阈值设置不当**
- ✅ 调高阈值（如0.7）提高准确率
- ✅ 调低阈值（如0.4）提高召回率

**原因4: 模型选择**
- ✅ 尝试不同的模型类型
- ✅ `eres2netv2` 通常效果最好

### Q3: GPU内存不够怎么办？

**A:** 有几种解决方案：

**方案1: 使用CPU**
```bash
python integrated_asr.py --audio test.wav --device cpu
```

**方案2: 修改配置文件**
```yaml
system:
  device: "cpu"
```

**方案3: 减少批处理大小**
```yaml
sensevoice:
  batch_size_s: 30  # 从60减少到30
```

### Q4: 支持实时识别吗？

**A:** 当前版本主要支持离线音频文件处理。实时流式识别功能在开发中。

### Q5: 如何提高识别速度？

**A:** 优化建议：

1. **使用GPU**
   ```yaml
   system:
     device: "cuda:0"
   ```

2. **关闭VAD**（对短音频）
   ```python
   # 修改代码，不使用VAD模型
   system = IntegratedASRSystem(vad_model=None)
   ```

3. **禁用不需要的功能**
   ```python
   # 不需要说话人识别时关闭
   system = IntegratedASRSystem(enable_speaker_verification=False)
   ```

4. **批量处理**
   ```python
   # 批量处理比逐个处理快
   system.batch_process(audio_list)
   ```

### Q6: 如何更换模型？

**A:** 修改配置文件：

```yaml
speaker_verification:
  model_type: "eres2netv2"  # 更换说话人识别模型

sensevoice:
  model: "iic/SenseVoiceSmall"  # 更换语音识别模型
```

支持的说话人识别模型：
- `campplus`: CAM++模型，速度快
- `eres2net`: ERes2Net模型，平衡
- `eres2netv2`: ERes2NetV2模型，效果最好（推荐）

### Q7: 声纹库存储在哪里？

**A:** 默认存储在 `./voiceprint_db` 目录：

```
voiceprint_db/
  ├── embeddings/          # 声纹特征文件
  │   ├── user001.npy
  │   └── user002.npy
  └── metadata.json        # 元数据文件
```

可以通过配置文件或命令行参数更改目录位置。

### Q8: 如何备份声纹库？

**A:** 直接复制整个 `voiceprint_db` 目录：

```bash
# 备份
cp -r voiceprint_db voiceprint_db_backup

# 恢复
cp -r voiceprint_db_backup voiceprint_db
```

### Q9: 不同机器的声纹库可以共享吗？

**A:** 可以，但需要注意：

✅ **可以共享的情况：**
- 使用相同的模型类型
- metadata.json 和 embeddings/ 目录完整

❌ **不能共享的情况：**
- 使用了不同的模型类型
- 模型版本不一致

### Q10: 如何添加新语言支持？

**A:** SenseVoice已支持主流语言，如需添加其他语言：

1. 查看SenseVoice官方文档
2. 可能需要微调模型
3. 参考官方微调教程

---

## 性能优化

### 1. 硬件选择

**推荐配置：**

| 场景 | CPU | 内存 | GPU | 性能 |
|------|-----|------|-----|------|
| 开发测试 | 4核 | 8GB | - | 慢 |
| 小规模部署 | 8核 | 16GB | GTX 1060 | 中 |
| 生产环境 | 16核 | 32GB | RTX 3090 | 快 |
| 大规模部署 | 32核 | 64GB | A100 | 很快 |

### 2. 软件优化

#### 优化1: 使用GPU加速

```yaml
system:
  device: "cuda:0"
  use_gpu: true
```

**性能提升：** 5-10倍

#### 优化2: 调整批处理大小

```yaml
sensevoice:
  batch_size_s: 60  # 根据GPU显存调整
```

- 显存4GB: batch_size_s=30
- 显存8GB: batch_size_s=60
- 显存16GB+: batch_size_s=120

#### 优化3: 预加载模型

```python
# 系统启动时预加载
system = IntegratedASRSystem(enable_speaker_verification=True)
system.speaker_manager.load_model()  # 预加载

# 后续使用更快
```

#### 优化4: 音频预处理

```python
import torchaudio

# 提前转换音频格式
wav, sr = torchaudio.load("input.mp3")
torchaudio.save("input.wav", wav, 16000)

# 使用转换后的wav文件
result = system.process_audio("input.wav")
```

### 3. 批量处理优化

```python
# 不推荐：逐个处理
for audio in audio_list:
    system.process_audio(audio)  # 慢

# 推荐：批量处理
system.batch_process(audio_list)  # 快
```

### 4. 部署优化

**方案1: 容器化部署**

```dockerfile
# Dockerfile示例
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "webui_integrated.py"]
```

**方案2: 使用ONNX加速**

```python
# 导出ONNX模型（开发中）
# 可以获得1.5-2倍加速
```

---

## 技术细节

### 1. 系统架构

```
IntegratedASRSystem
├── SpeakerManager (3D-Speaker)
│   ├── 模型加载
│   ├── 特征提取
│   ├── 声纹匹配
│   └── 数据库管理
└── SenseVoice
    ├── VAD切分
    ├── 语音识别
    ├── 情感识别
    └── 事件检测
```

### 2. 处理流程

```
输入音频
    ↓
[可选] 说话人识别
    ├→ 提取声纹特征
    ├→ 与数据库匹配
    └→ 返回说话人信息
    ↓
语音识别
    ├→ VAD切分
    ├→ 特征提取
    ├→ 模型推理
    └→ 后处理
    ↓
输出结果
    ├→ 识别文本
    ├→ 情感标签
    ├→ 事件标签
    └→ 说话人信息
```

### 3. 声纹特征

**特征维度：** 192维（eres2net）

**提取方法：**
1. 音频 → FBank特征 (80维)
2. FBank → ERes2Net → Embedding (192维)
3. Embedding 归一化
4. 保存为 .npy 文件

**匹配方法：**
- 余弦相似度计算
- 公式: `similarity = (A·B) / (||A||·||B||)`
- 范围: [-1, 1]，越接近1越相似

### 4. 模型详情

#### 3D-Speaker模型

| 模型 | 参数量 | 特征维度 | 训练数据 | 适用场景 |
|------|--------|----------|---------|---------|
| CAM++ | 7M | 192 | 20万人 | 中文通用 |
| ERes2Net | 6.6M | 192 | 20万人 | 中文通用 |
| ERes2NetV2 | 7.1M | 192 | 20万人 | 中文通用（推荐） |

#### SenseVoice模型

| 能力 | 详情 |
|------|------|
| 模型大小 | ~220M |
| 参数量 | ~220M |
| 语言支持 | 5种 |
| 情感类别 | 7种 |
| 事件类别 | 8种 |
| 推理速度 | 10s音频 70ms |

### 5. 数据格式

#### 声纹元数据 (metadata.json)

```json
{
  "user001": {
    "speaker_id": "user001",
    "speaker_name": "张三",
    "audio_path": "voice.wav",
    "embedding_file": "voiceprint_db/embeddings/user001.npy",
    "register_time": "2024-10-30T10:30:00",
    "model_type": "eres2net",
    "metadata": {}
  }
}
```

#### 识别结果

```json
{
  "audio_path": "test.wav",
  "speaker_info": {
    "speaker_id": "user001",
    "speaker_name": "张三",
    "similarity": 0.856
  },
  "asr_results": [
    {"text": "这是识别的文本😊"}
  ],
  "processing_time": 1.23
}
```

---

## 完整工作流程示例

### 场景1: 新系统首次使用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 注册第一个说话人
python register_voiceprint.py \
  --action register \
  --speaker-id user001 \
  --speaker-name "张三" \
  --audio voice_zhangsan.wav

# 3. 注册第二个说话人
python register_voiceprint.py \
  --action register \
  --speaker-id user002 \
  --speaker-name "李四" \
  --audio voice_lisi.wav

# 4. 查看已注册的声纹
python register_voiceprint.py --action list

# 5. 测试识别（启用说话人识别）
python integrated_asr.py \
  --audio test.wav \
  --enable-speaker

# 6. 启动Web界面
python webui_integrated.py
```

### 场景2: 批量处理会议录音

```bash
# 1. 准备音频文件
meeting_audio/
  ├── speaker1.wav    # 用于注册
  ├── speaker2.wav    # 用于注册
  ├── meeting.wav     # 会议录音

# 2. 批量注册参会人员声纹
python register_voiceprint.py \
  --action batch-register \
  --audio-dir meeting_audio \
  --pattern "speaker*.wav"

# 3. 处理会议录音
python integrated_asr.py \
  --audio meeting_audio/meeting.wav \
  --enable-speaker \
  --language zh \
  --output meeting_result.json

# 4. 查看结果
cat meeting_result.json
```

### 场景3: Python脚本集成

```python
#!/usr/bin/env python3
from integrated_asr import IntegratedASRSystem
from register_voiceprint import VoiceprintManager
import json

# 初始化系统
system = IntegratedASRSystem(enable_speaker_verification=True)

# 场景A: 新用户注册
def register_new_user(user_id, user_name, voice_file):
    manager = system.speaker_manager
    success = manager.register_speaker(
        speaker_id=user_id,
        audio_path=voice_file,
        speaker_name=user_name
    )
    return success

# 场景B: 识别并处理音频
def process_with_speaker_id(audio_file):
    result = system.process_audio(audio_file)
    
    if result['speaker_info']:
        speaker = result['speaker_info']['speaker_name']
        text = result['asr_results'][0]['text']
        print(f"{speaker}: {text}")
    else:
        text = result['asr_results'][0]['text']
        print(f"未知说话人: {text}")
    
    return result

# 场景C: 批量处理
def batch_process_directory(audio_dir):
    import os
    from pathlib import Path
    
    audio_files = list(Path(audio_dir).glob("*.wav"))
    results = system.batch_process(
        audio_list=[str(f) for f in audio_files],
        output_file="batch_results.json"
    )
    
    return results

# 使用示例
if __name__ == '__main__':
    # 注册新用户
    register_new_user("user001", "张三", "voice1.wav")
    
    # 处理单个音频
    result = process_with_speaker_id("test1.wav")
    
    # 批量处理
    results = batch_process_directory("./audio_files")
```

---

## 项目文件说明

### 核心文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `register_voiceprint.py` | 声纹注册管理脚本 | 注册、管理声纹 |
| `integrated_asr.py` | 集成ASR系统 | 语音识别+说话人识别 |
| `webui_integrated.py` | Web界面 | 图形化操作界面 |
| `config_3dspeaker.yaml` | 配置文件 | 系统配置 |
| `README_3DSpeaker.md` | 使用文档 | 本文档 |

### 目录结构

```
SenseVoice/
├── register_voiceprint.py      # 声纹管理脚本
├── integrated_asr.py            # 集成系统
├── webui_integrated.py          # Web界面
├── config_3dspeaker.yaml        # 配置文件
├── README_3DSpeaker.md          # 使用文档
├── voiceprint_db/               # 声纹数据库
│   ├── embeddings/              # 声纹特征
│   └── metadata.json            # 元数据
├── pretrained_models/           # 模型缓存
│   └── ...
├── speakerlab/                  # 3D-Speaker代码
│   ├── models/
│   ├── process/
│   └── utils/
└── requirements.txt             # 依赖列表
```

---

## 更新日志

### v1.0.0 (2024-10-30)

✨ **新功能：**
- ✅ 集成3D-Speaker和SenseVoice
- ✅ 声纹注册和管理功能
- ✅ 说话人识别功能
- ✅ 可切换的功能开关
- ✅ Web UI界面
- ✅ 命令行工具
- ✅ Python API
- ✅ 配置文件支持
- ✅ 批量处理功能

🐛 **问题修复：**
- 无

📝 **文档：**
- ✅ 完整的使用文档
- ✅ 详细的示例代码
- ✅ 常见问题解答

---

## 技术支持

### 问题反馈

- **GitHub Issues**: 在项目页面提交问题
- **钉钉群**: 扫描README中的二维码加入交流群

### 参考资源

- [SenseVoice 官方文档](https://github.com/FunAudioLLM/SenseVoice)
- [3D-Speaker 官方文档](https://github.com/modelscope/3D-Speaker)
- [FunASR 文档](https://github.com/modelscope/FunASR)
- [ModelScope 平台](https://www.modelscope.cn/)

---

## 许可证

本项目遵循原始项目的许可证：
- SenseVoice: Apache License 2.0
- 3D-Speaker: Apache License 2.0

---

## 致谢

感谢以下项目：
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 语音识别模型
- [3D-Speaker](https://github.com/modelscope/3D-Speaker) - 说话人识别模型
- [FunASR](https://github.com/modelscope/FunASR) - 语音识别工具包
- [ModelScope](https://www.modelscope.cn/) - 模型社区

---

**最后更新：** 2024-10-30
**文档版本：** v1.0.0

