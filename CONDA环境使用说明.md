# Conda环境使用说明

## 🐍 重要提示

本项目**需要在conda环境中运行**！所有Python命令都必须先激活`sensevoice`环境。

---

## 📦 环境准备

### 方式1: 使用快速启动脚本（推荐）

```bash
cd /Users/mingy/Documents/python/SenseVoice
bash quick_start_conda.sh
```

脚本会自动：
- ✅ 检查conda是否安装
- ✅ 检查sensevoice环境是否存在
- ✅ 激活环境
- ✅ 安装依赖
- ✅ 创建必要目录
- ✅ 提供启动选项

### 方式2: 手动配置

#### 步骤1: 创建环境（如果不存在）

```bash
# 检查环境是否存在
conda env list

# 如果不存在，创建新环境
conda create -n sensevoice python=3.8 -y
```

#### 步骤2: 激活环境

```bash
conda activate sensevoice
```

**验证环境已激活：**
```bash
# 命令行提示符应该显示: (sensevoice)
# 检查Python路径
which python
# 应该显示conda环境中的python路径
```

#### 步骤3: 安装依赖

```bash
pip install -r requirements.txt
```

#### 步骤4: 创建目录

```bash
mkdir -p voiceprint_db/embeddings
mkdir -p pretrained_models
mkdir -p results
```

---

## 🚀 正确的使用方式

### ⚠️ 错误示例（不要这样做）

```bash
# ❌ 错误：直接运行，没有激活环境
python webui_integrated.py

# ❌ 错误：使用系统Python
/usr/bin/python3 webui_integrated.py
```

### ✅ 正确示例（必须这样做）

```bash
# ✅ 正确：先激活环境
conda activate sensevoice

# 然后运行命令
python webui_integrated.py
```

---

## 💻 常用命令（Conda版本）

### 启动Web界面

```bash
# 1. 激活环境
conda activate sensevoice

# 2. 启动Web界面
python webui_integrated.py
```

### 注册声纹

```bash
# 1. 激活环境
conda activate sensevoice

# 2. 注册
python register_voiceprint.py \
  --action register \
  --speaker-id user001 \
  --speaker-name "张三" \
  --audio voice.wav
```

### 语音识别

```bash
# 1. 激活环境
conda activate sensevoice

# 2. 基础识别
python integrated_asr.py --audio test.wav

# 3. 启用说话人识别
python integrated_asr.py --audio test.wav --enable-speaker
```

### 批量处理

```bash
# 1. 激活环境
conda activate sensevoice

# 2. 批量处理
python integrated_asr.py \
  --audio-list file1.wav file2.wav file3.wav \
  --enable-speaker \
  --output results.json
```

### 查看声纹列表

```bash
# 1. 激活环境
conda activate sensevoice

# 2. 列出声纹
python register_voiceprint.py --action list
```

---

## 🔧 Conda环境管理

### 查看所有环境

```bash
conda env list
```

### 激活环境

```bash
conda activate sensevoice
```

### 退出环境

```bash
conda deactivate
```

### 删除环境（谨慎）

```bash
# 先退出环境
conda deactivate

# 删除环境
conda remove -n sensevoice --all
```

### 导出环境配置

```bash
# 激活环境
conda activate sensevoice

# 导出配置
conda env export > environment.yml
```

### 从配置文件创建环境

```bash
conda env create -f environment.yml
```

---

## 📝 创建别名（可选）

为了方便使用，可以创建shell别名：

### Bash/Zsh

编辑 `~/.bashrc` 或 `~/.zshrc`：

```bash
# SenseVoice别名
alias sv-activate='conda activate sensevoice'
alias sv-webui='conda activate sensevoice && cd /Users/mingy/Documents/python/SenseVoice && python webui_integrated.py'
alias sv-register='conda activate sensevoice && cd /Users/mingy/Documents/python/SenseVoice && python register_voiceprint.py'
alias sv-asr='conda activate sensevoice && cd /Users/mingy/Documents/python/SenseVoice && python integrated_asr.py'
```

然后重新加载配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

使用别名：

```bash
# 激活环境
sv-activate

# 启动Web界面
sv-webui

# 注册声纹
sv-register --action register --speaker-id user001 --audio voice.wav

# 语音识别
sv-asr --audio test.wav --enable-speaker
```

---

## 🐛 常见问题

### Q1: conda: command not found

**原因：** Conda未安装或未添加到PATH

**解决方案：**

1. 安装Miniconda或Anaconda
2. 初始化conda：
   ```bash
   conda init bash  # 或 conda init zsh
   ```
3. 重启终端

### Q2: 环境激活失败

**错误信息：** `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.`

**解决方案：**

```bash
# 方式1: 初始化conda
conda init bash  # 或 zsh

# 方式2: 使用source命令
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sensevoice
```

### Q3: 模块导入错误

**错误信息：** `ModuleNotFoundError: No module named 'torch'`

**原因：** 依赖未安装或不在正确的环境中

**解决方案：**

```bash
# 1. 确认环境已激活
conda activate sensevoice

# 2. 检查Python路径
which python
# 应该显示：/Users/mingy/anaconda3/envs/sensevoice/bin/python

# 3. 重新安装依赖
pip install -r requirements.txt
```

### Q4: 使用了系统Python

**问题：** 运行时使用的是系统Python而不是conda环境中的Python

**检查：**

```bash
# 查看当前Python
which python

# 如果显示 /usr/bin/python3，说明没有激活环境
# 应该显示 conda环境路径
```

**解决方案：**

```bash
# 激活环境
conda activate sensevoice

# 再次检查
which python
```

### Q5: GPU不可用

**检查：**

```bash
conda activate sensevoice
python -c "import torch; print(torch.cuda.is_available())"
```

**如果显示False，可能的原因：**

1. 没有NVIDIA GPU
2. CUDA未安装
3. PyTorch版本不匹配

**解决方案：**

```bash
# 如果有GPU但CUDA不可用，重新安装PyTorch
conda activate sensevoice
pip uninstall torch torchaudio
# 根据CUDA版本安装（查看：nvidia-smi）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果没有GPU，使用CPU版本
# 修改 config_3dspeaker.yaml:
#   system:
#     device: "cpu"
```

---

## 📋 环境检查清单

在运行项目前，请检查：

- [ ] Conda已安装：`conda --version`
- [ ] sensevoice环境已创建：`conda env list | grep sensevoice`
- [ ] 环境已激活：提示符显示 `(sensevoice)`
- [ ] Python路径正确：`which python` 显示conda环境路径
- [ ] 依赖已安装：`pip list | grep torch`
- [ ] 目录已创建：`ls voiceprint_db pretrained_models`

**快速检查脚本：**

```bash
#!/bin/bash
echo "=== 环境检查 ==="
echo "1. Conda版本: $(conda --version)"
echo "2. 当前环境: $CONDA_DEFAULT_ENV"
echo "3. Python路径: $(which python)"
echo "4. Python版本: $(python --version)"
echo "5. PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "6. CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "================"
```

---

## 🎯 推荐工作流

### 首次使用

```bash
# 1. 运行快速启动脚本
cd /Users/mingy/Documents/python/SenseVoice
bash quick_start_conda.sh

# 2. 脚本会引导你完成所有设置

# 3. 启动Web界面
# （如果脚本中选择了自动启动，会直接打开）
```

### 日常使用

```bash
# 方式1: 每次手动激活
conda activate sensevoice
cd /Users/mingy/Documents/python/SenseVoice
python webui_integrated.py

# 方式2: 使用别名（如果配置了）
sv-webui

# 方式3: 创建启动脚本
# 创建 start.sh:
#!/bin/bash
conda activate sensevoice
cd /Users/mingy/Documents/python/SenseVoice
python webui_integrated.py
```

---

## 📚 相关文档

- **详细使用文档**: [README_3DSpeaker.md](README_3DSpeaker.md)
- **快速使用指南**: [快速使用指南.md](快速使用指南.md)
- **集成说明**: [README_集成说明.md](README_集成说明.md)
- **项目总结**: [项目完成总结_3DSpeaker.md](项目完成总结_3DSpeaker.md)

---

## 🎊 一键启动命令

保存到你喜欢的位置（如 `~/start_sensevoice.sh`）：

```bash
#!/bin/bash
# SenseVoice一键启动脚本

# 获取conda路径
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    echo "错误: Conda未安装"
    exit 1
fi

# 激活conda
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 激活sensevoice环境
conda activate sensevoice

if [ $? -ne 0 ]; then
    echo "错误: 无法激活sensevoice环境"
    echo "请先创建环境: conda create -n sensevoice python=3.8"
    exit 1
fi

# 进入项目目录
cd /Users/mingy/Documents/python/SenseVoice

# 启动Web界面
echo "正在启动SenseVoice + 3D-Speaker集成系统..."
echo "浏览器将打开 http://localhost:7860"
python webui_integrated.py
```

使用：

```bash
chmod +x ~/start_sensevoice.sh
~/start_sensevoice.sh
```

---

**记住：始终先 `conda activate sensevoice`，然后再运行任何Python命令！** 🎯

