# Ollama 模型加载指南

## 📋 已创建的脚本

### 单个模型加载脚本
每个模型目录下都有对应的加载脚本：

| 模型 | 脚本位置 | 执行命令 |
|------|---------|---------|
| Gemma-4-E4B | `models/gemma-4-E4B/gemma4_e4b_ollama.ps1` | `.\models\gemma-4-E4B\gemma4_e4b_ollama.ps1` |
| Gemma-4-E2B | `models/gemma-4-E2B/gemma4_e2b_ollama.ps1` | `.\models\gemma-4-E2B\gemma4_e2b_ollama.ps1` |
| GLM Edge 4B | `models/glm-edge-4b-chat/glm_edge_ollama.ps1` | `.\models\glm-edge-4b-chat\glm_edge_ollama.ps1` |
| Phi-4-Mini | `models/Phi-4-mini-instruct/phi4_ollama.ps1` | `.\models\Phi-4-mini-instruct\phi4_ollama.ps1` |
| Qwen 3.5 4B | `models/Qwen3.5-4B/qwen_ollama.ps1` | `.\models\Qwen3.5-4B\qwen_ollama.ps1` |

### 批量加载脚本
一次性加载所有 5 个模型：

```powershell
.\load_all_models.ps1
```

## 🚀 使用步骤

### 前置条件
1. **安装 Ollama**
   ```powershell
   winget install -e --id Ollama.Ollama
   ```

2. **启动 Ollama 服务**（新终端窗口）
   ```powershell
   ollama serve
   ```

### 方式一：一键加载所有模型（推荐）

```powershell
cd C:\local_llm_context_compaction
.\load_all_models.ps1
```

**耗时**：取决于网络和 GPU，每个模型 2-5 分钟

**输出**：
```
[HH:mm:ss] Processing: gemma4-e4b
  Found: gemma-4-E4B-it-Q4_K_M.gguf
  Building model...
  SUCCESS: Model loaded

[HH:mm:ss] Processing: gemma4-e2b
  ...

Summary:
  Loaded: 5 models
  Failed: 0 models
```

### 方式二：单个加载

```powershell
# 例如加载 Gemma-4-E4B
.\models\gemma-4-E4B\gemma4_e4b_ollama.ps1
```

## 📊 模型配置

所有模型配置相同：
- **上下文窗口**：32k tokens (`num_ctx 32768`)
- **GPU 推理**：全量 (`num_gpu -1`)
- **温度**：0.7（可在脚本中修改）

## 🎯 使用加载的模型

加载完成后，可以直接使用：

```powershell
# 对话模式
ollama run gemma4-e4b

# 一次性推理
ollama run gemma4-e4b "What is machine learning?"
ollama run glm-edge-4b "请解释深度学习"
ollama run phi4-mini "Explain quantum computing"
ollama run qwen-3.5-4b "介绍自然语言处理"
```

## 📈 查看已加载的模型

```powershell
ollama list
```

输出示例：
```
NAME                 ID              SIZE     MODIFIED
gemma4-e4b          xxx...xxx       4.6GB    5 minutes ago
gemma4-e2b          xxx...xxx       3.2GB    4 minutes ago
glm-edge-4b         xxx...xxx       4.5GB    3 minutes ago
phi4-mini           xxx...xxx       2.8GB    2 minutes ago
qwen-3.5-4b         xxx...xxx       4.1GB    1 minute ago
```

## ⚠️ 故障排除

### 问题 1：模型加载失败
```
Error: 500 Internal Server Error: unable to load model
```
**原因**：Ollama 的 llama.cpp 不支持某些架构（如 Gemma-4）
**解决**：等待 Ollama 更新或使用其他工具（CTransformers）

### 问题 2：Ollama 未运行
```
Error: Could not connect to ollama
```
**解决**：在新终端运行 `ollama serve`

### 问题 3：显存不足
**现象**：推理时崩溃或很慢
**解决**：修改脚本中的 `num_gpu` 参数（减少 GPU 层数）

## 💾 磁盘占用

所有 5 个模型总体积：
- 总计：**~20GB**（包括量化和元数据）
- Ollama 缓存：`C:\Users\{username}\.ollama\models\`

## 🔧 自定义配置

要修改任何模型的参数，编辑对应脚本中的 Modelfile 部分：

```powershell
$modelfileContent = "FROM ./$GGUF_FILE`nPARAMETER num_ctx 32768`nPARAMETER num_gpu -1`nPARAMETER temperature 0.7`nSYSTEM `"Custom system prompt`""
```

可调参数：
- `num_ctx`: 上下文大小（默认 32768）
- `num_gpu`: GPU 层数（-1 = 全量，0 = CPU only）
- `temperature`: 温度（0.0-2.0，默认 0.7）
- `top_p`: 核采样（默认 0.9）
- `top_k`: Top-K 采样（默认 40）

## 📝 脚本自动化

创建一个快捷批处理文件 `quick_load.bat`（Windows）：

```batch
@echo off
cd /d C:\local_llm_context_compaction
powershell -ExecutionPolicy Bypass -File .\load_all_models.ps1
pause
```

然后直接双击运行。
