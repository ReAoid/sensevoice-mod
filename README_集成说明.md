# SenseVoice + 3D-Speaker 集成系统

## 🎉 新功能：说话人识别

本项目已成功集成 **3D-Speaker** 说话人识别功能！现在您可以在语音识别的同时识别说话人身份。

---

## ⚡ 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Web界面

```bash
python webui_integrated.py
```

浏览器打开：http://localhost:7860

### 3. 开始使用

#### 基础语音识别（原SenseVoice功能）

1. 上传音频
2. 选择语言
3. 点击"开始识别"

**功能：** 多语言识别 + 情感识别 + 事件检测

#### 说话人识别（新增功能）

1. 切换到"声纹管理"标签页
2. 注册说话人声纹
3. 回到"语音识别"标签页
4. 勾选"启用说话人识别"
5. 开始识别

**功能：** 识别说话人身份（基于3D-Speaker）

---

## 📁 核心文件

| 文件 | 说明 |
|------|------|
| `webui_integrated.py` | 集成Web界面（**推荐使用**） |
| `integrated_asr.py` | 集成系统（命令行/API） |
| `register_voiceprint.py` | 声纹管理工具 |
| `config_3dspeaker.yaml` | 配置文件 |
| `README_3DSpeaker.md` | **完整使用文档** |
| `快速使用指南.md` | 快速上手指南 |
| `example_usage.py` | 代码示例 |

---

## 🎯 功能对比

### 原SenseVoice功能

- ✅ 多语言语音识别（中、英、粤、日、韩）
- ✅ 情感识别（😊😔😡😰🤢😮）
- ✅ 事件检测（🎼👏😀😭🤧）
- ✅ 低延迟推理

### 新增3D-Speaker功能

- 🆕 说话人声纹注册
- 🆕 说话人身份识别
- 🆕 声纹相似度计算
- 🆕 声纹数据库管理
- 🆕 灵活的功能开关

---

## 🔄 使用方式

### 方式1: Web界面（推荐）

```bash
# 集成界面（支持说话人识别）
python webui_integrated.py

# 原始界面（仅SenseVoice）
python webui.py
```

### 方式2: 命令行

```bash
# 基础语音识别
python integrated_asr.py --audio test.wav

# 语音识别 + 说话人识别
python integrated_asr.py --audio test.wav --enable-speaker

# 声纹管理
python register_voiceprint.py --action register --speaker-id user001 --audio voice.wav
```

### 方式3: Python API

```python
from integrated_asr import IntegratedASRSystem

# 创建系统（启用说话人识别）
system = IntegratedASRSystem(enable_speaker_verification=True)

# 处理音频
result = system.process_audio("test.wav")

# 查看结果
print(result['speaker_info'])  # 说话人信息
print(result['asr_results'])   # 识别文本
```

---

## 📚 详细文档

### 完整文档

📖 **[README_3DSpeaker.md](README_3DSpeaker.md)** - 超详细的使用文档

包含：
- 系统架构
- 安装配置
- 功能说明
- 使用教程
- API文档
- 常见问题
- 性能优化
- 技术细节

### 快速指南

📄 **[快速使用指南.md](快速使用指南.md)** - 5分钟快速上手

包含：
- 快速开始
- 基础使用
- 功能切换
- 流程图
- 问题速查

### 代码示例

💻 **[example_usage.py](example_usage.py)** - 7个完整示例

包含：
- 基础语音识别
- 声纹注册
- 说话人识别
- 功能切换
- 批量处理
- 多语言识别
- 系统信息

---

## 🎮 功能切换

### 说话人识别功能可以灵活开启/关闭

**方法1: Web界面**
- 勾选"启用说话人识别"复选框

**方法2: 命令行**
```bash
# 开启
python integrated_asr.py --audio test.wav --enable-speaker

# 关闭
python integrated_asr.py --audio test.wav
```

**方法3: 配置文件**
```yaml
# config_3dspeaker.yaml
speaker_verification:
  enabled: true  # true=开启, false=关闭
```

**方法4: Python代码**
```python
system = IntegratedASRSystem(enable_speaker_verification=True)
system.toggle_speaker_verification(False)  # 动态切换
```

---

## 🌟 应用场景

### 场景1: 纯语音识别
**使用：** 原SenseVoice功能  
**开关：** 不启用说话人识别  
**适合：** 单人录音、字幕生成、语音转文字

### 场景2: 会议纪要
**使用：** 集成功能  
**开关：** 启用说话人识别  
**适合：** 多人会议、识别发言人、生成带名字的纪要

### 场景3: 智能客服
**使用：** 集成功能  
**开关：** 启用说话人识别  
**适合：** 识别来电客户、个性化服务、历史记录关联

### 场景4: 身份验证
**使用：** 3D-Speaker功能  
**开关：** 仅使用声纹识别  
**适合：** 声纹验证、安全认证、身份确认

---

## 🔧 配置说明

### 主要配置项

```yaml
# config_3dspeaker.yaml

# 说话人识别
speaker_verification:
  enabled: false              # 默认启用状态
  model_type: "eres2net"     # 模型类型
  threshold: 0.5              # 识别阈值

# SenseVoice
sensevoice:
  model: "iic/SenseVoiceSmall"
  default_language: "auto"

# 系统
system:
  device: "cuda:0"            # 或 "cpu"
  
# Web UI
webui:
  port: 7860
```

---

## 💡 常见问题

### Q: 首次运行很慢？
**A:** 需要下载模型（约2GB），后续会很快。

### Q: 如何提高识别准确率？
**A:** 
- 使用3-10秒纯净人声注册
- 在安静环境录制
- 调整识别阈值（配置文件）

### Q: GPU内存不够？
**A:** 修改配置文件：`device: "cpu"`

### Q: 不需要说话人识别功能？
**A:** 直接使用原版：`python webui.py`

---

## 📞 技术支持

- **详细文档**: [README_3DSpeaker.md](README_3DSpeaker.md)
- **快速指南**: [快速使用指南.md](快速使用指南.md)
- **GitHub Issues**: 项目页面
- **钉钉群**: 原SenseVoice交流群

---

## 📊 技术栈

- **SenseVoice**: 语音识别、情感识别、事件检测
- **3D-Speaker**: 说话人识别、声纹提取
- **FunASR**: 语音识别框架
- **Gradio**: Web界面
- **PyTorch**: 深度学习框架
- **ModelScope**: 模型仓库

---

## 📜 许可证

- SenseVoice: Apache License 2.0
- 3D-Speaker: Apache License 2.0

---

## 🎊 快速命令参考

```bash
# Web界面
python webui_integrated.py              # 集成界面
python webui.py                         # 原始界面

# 声纹管理
python register_voiceprint.py --action register --speaker-id ID --audio file.wav
python register_voiceprint.py --action list
python register_voiceprint.py --action identify --audio file.wav

# 语音识别
python integrated_asr.py --audio test.wav                    # 基础识别
python integrated_asr.py --audio test.wav --enable-speaker   # 包含说话人识别

# 查看帮助
python register_voiceprint.py --help
python integrated_asr.py --help

# 运行示例
python example_usage.py

# 快速启动
bash quick_start.sh
```

---

**更新日期**: 2024-10-30  
**集成版本**: v1.0.0  
**原SenseVoice**: 保持不变，完全兼容

---

🎉 **开始使用吧！运行 `python webui_integrated.py` 体验新功能！**

