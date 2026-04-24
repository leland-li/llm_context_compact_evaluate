# ==========================================
# 为压缩任务构建优化的 Ollama 模型
# ==========================================
# 功能：基于不同模型创建压缩专用的 Modelfile
# 使用：.\create_compression_models.ps1
# ==========================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 英文压缩任务提示词
$COMPRESS_PROMPT = @"
You are a professional context compression engine specialized in processing conversational histories with tool calls and decision-making logic.

Compression Guidelines:
1. Multi-granularity Retention: Preserve all tool execution results, IDs, key numerical values, and dates.
2. Logical Distillation: Maintain causal links and decision reasoning.
3. Aggressive Denoising: Remove chitchat, system logs, and redundant documentation.
4. Output Constraint: Output only the structured summary. No meta-commentary or self-introduction.
5. Format: Preserve the original structure where possible, use bullet points for clarity.
"@

$TEMPLATE_CHATML = '{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>'
$TEMPLATE_GEMMA  = '{{ if .System }}<|turn|>system\n{{ .System }}<turn|>\n{{ end }}{{ if .Prompt }}<|turn|>user\n{{ .Prompt }}<turn|>\n{{ end }}<|turn|>assistant\n{{ .Response }}<turn|>'
$TEMPLATE_GLM    = '{{ if .System }}<|system|>\n{{ .System }}{{ end }}{{ if .Prompt }}<|user|>\n{{ .Prompt }}{{ end }}<|assistant|>\n{{ .Response }}'

# 模型配置：模型名称 -> (本地路径, 生成参数)
$MODEL_CONFIGS = @{
    "gemma4-e4b-compress" = @{
        gguf_path = "models\gemma-4-E4B\gemma-4-E4B-it-Q4_K_M.gguf"
        num_ctx = 32768
        template  = $TEMPLATE_GEMMA
        temperature = 0.1
        top_p = 0.85
        top_k = 40
        repeat_penalty = 1.15
        stop_tokens = @("<turn|>", "<eos>", "<channel|>", "<tool_call|>", "<|tool_response|>")
    }
    "gemma4-e2b-compress" = @{
        gguf_path = "models\gemma-4-E2B\gemma-4-E2B-it-Q8_0.gguf"
        num_ctx = 32768
        template  = $TEMPLATE_GEMMA
        temperature = 0.1
        top_p = 0.85
        top_k = 40
        repeat_penalty = 1.15
        stop_tokens = @("<turn|>", "<eos>", "<channel|>", "<tool_call|>", "<|tool_response|>")
    }
    "glm-edge-4b-compress" = @{
        gguf_path = "models\glm-edge-4b-chat\ggml-model-Q4_K_M.gguf"
        num_ctx = 16384
        template  = $TEMPLATE_GLM
        temperature = 0.1
        top_p = 0.85
        top_k = 40
        repeat_penalty = 1.15
        stop_tokens = @("<|endoftext|>", "<|user|>", "<|observation|>")
    }
    "qwen3.5-4b-compress" = @{
        gguf_path = "models\Qwen3.5-4B\Qwen3.5-4B-Q4_K_M.gguf"
        num_ctx = 32768
        template  = $TEMPLATE_CHATML
        temperature = 0.1
        top_p = 0.85
        top_k = 40
        repeat_penalty = 1.15
        stop_tokens = @("<|im_end|>", "<|endoftext|>", "</tool_call>", "</think>")
    }
    "phi4-mini-compress" = @{
        gguf_path = "models\Phi-4-mini-instruct\Phi-4-mini-instruct-Q4_K_M.gguf"
        num_ctx = 32768
        template  = $TEMPLATE_CHATML
        temperature = 0.1
        top_p = 0.85
        top_k = 40
        repeat_penalty = 1.15
        stop_tokens = @("<|end|>", "<|endoftext|>", "<|user|>", "<|system|>")
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "创建压缩优化的 Ollama 模型" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$successCount = 0
$failCount = 0

foreach ($modelName in $MODEL_CONFIGS.Keys) {
    $config = $MODEL_CONFIGS[$modelName]
    
    # 获取绝对路径并检查
    try {
        $absoluteGgufPath = (Resolve-Path $config.gguf_path -ErrorAction Stop).Path
    } catch {
        Write-Host "  ❌ 错误: 找不到 GGUF 文件 - $($config.gguf_path)" -ForegroundColor Red
        $failCount++
        continue
    }

    Write-Host "🔧 处理模型: $modelName" -ForegroundColor Yellow
    Write-Host "   参数: T=$($config.temperature), P=$($config.top_p), Ctx=$($config.num_ctx)" -ForegroundColor Gray
    
    # 生成 Modelfile 内容
    $stopArray = $config.stop_tokens | ForEach-Object { "PARAMETER stop `"$_`"" }
    $stopTokensLines = [string]::Join("`n", $stopArray) | Join-String -Separator "`n"
    
    $modelfileContent = @"
FROM "$absoluteGgufPath"
TEMPLATE "$($config.template)"
SYSTEM """$COMPRESS_PROMPT"""
PARAMETER num_ctx $($config.num_ctx)
PARAMETER temperature $($config.temperature)
PARAMETER top_p $($config.top_p)
PARAMETER top_k $($config.top_k)
PARAMETER repeat_penalty $($config.repeat_penalty)
$stopTokensLines
"@

    # 写入并执行
    $tempModelfile = Join-Path $pwd "Modelfile.tmp"
    [System.IO.File]::WriteAllText($tempModelfile, $modelfileContent)
    
    Write-Host "   📝 正在向 Ollama 提交构建请求..." -ForegroundColor Cyan
    ollama create $modelName -f "$tempModelfile" 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ 成功创建模型: $modelName" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "   ❌ 创建失败: $modelName" -ForegroundColor Red
        $failCount++
    }
    
    if (Test-Path $tempModelfile) { Remove-Item $tempModelfile -Force }
    Write-Host ""
}

# 输出总结
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "构建总结" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ 成功: $successCount 个模型" -ForegroundColor Green
Write-Host "❌ 失败: $failCount 个模型" -ForegroundColor Red
Write-Host ""

# 列出已创建的模型
Write-Host "已创建的压缩模型:" -ForegroundColor Yellow
ollama list | Select-String "compress"

Write-Host ""
Write-Host "使用示例:" -ForegroundColor Cyan
Write-Host "  ollama run gemma4-e4b-compress '你的压缩任务提示词'"
