# CPU模式使用说明

## ✅ 已修复

系统已默认配置为**CPU模式**，可以直接使用。

---

## 🔧 配置说明

### 当前配置（CPU模式）

配置文件 `config_3dspeaker.yaml` 已设置为：

```yaml
system:
  device: "cpu"
```

这是因为您的PyTorch没有CUDA支持。

---

## 💻 使用方式（不变）

```bash
# 激活环境
conda activate sensevoice

# 启动Web界面
python webui_integrated.py
```

---

## ⚡ 如果想使用GPU加速

如果您有NVIDIA GPU并想使用GPU加速，需要：

### 1. 检查是否有NVIDIA GPU

```bash
nvidia-smi
```

如果看到GPU信息，说明有GPU。

### 2. 安装支持CUDA的PyTorch

```bash
# 激活环境
conda activate sensevoice

# 卸载当前PyTorch
pip uninstall torch torchaudio

# 安装CUDA版本（根据您的CUDA版本选择）
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 验证CUDA是否可用

```bash
conda activate sensevoice
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
```

如果显示 `CUDA可用: True`，说明安装成功。

### 4. 修改配置文件

编辑 `config_3dspeaker.yaml`：

```yaml
system:
  device: "cuda:0"  # 改为cuda:0
```

### 5. 重启Web界面

```bash
conda activate sensevoice
python webui_integrated.py
```

---

## 📊 CPU vs GPU 性能对比

| 设备 | 10秒音频处理时间 | 说话人识别 | 适用场景 |
|------|-----------------|-----------|---------|
| CPU | 5-10秒 | 支持 | 测试、小规模使用 |
| GPU (如RTX 3090) | 0.5-1秒 | 支持 | 生产环境、大规模处理 |

---

## ⚠️ 注意事项

### CPU模式下

- ✅ 所有功能都可以正常使用
- ✅ 准确率不变
- ⚠️ 处理速度较慢
- ⚠️ 首次加载模型需要较长时间

### 性能优化建议

1. **减少批处理大小**
   ```yaml
   sensevoice:
     batch_size_s: 30  # 从60改为30
   ```

2. **关闭不需要的功能**
   - 如果不需要说话人识别，不要勾选启用

3. **批量处理**
   - 多个文件一起处理比逐个处理效率更高

---

## 🐛 常见问题

### Q: 启动很慢？

A: 首次启动需要下载模型（约2GB），之后会快很多。

### Q: 处理速度慢？

A: CPU模式下这是正常的。建议：
- 使用GPU
- 处理较短的音频
- 关闭不需要的功能

### Q: 如何确认当前使用的是CPU还是GPU？

A: 查看启动日志：
```
计算设备: cpu  # 或 cuda:0
```

---

## 🎯 推荐配置

### 开发测试环境（CPU）

```yaml
system:
  device: "cpu"

sensevoice:
  batch_size_s: 30
```

### 生产环境（GPU）

```yaml
system:
  device: "cuda:0"

sensevoice:
  batch_size_s: 60
```

---

现在可以正常使用了！🎉

```bash
conda activate sensevoice
python webui_integrated.py
```

