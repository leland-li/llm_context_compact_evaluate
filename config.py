import os
from pathlib import Path

# 获取项目根目录：使用 pathlib 自动处理 Windows/Linux 路径差异
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_CONFIG = [
    {
        "name": "gemma-4-E4B", 
        "gguf_path": str(PROJECT_ROOT / "models" / "gemma-4-E4B" / "gemma-4-E4B-it-Q4_K_M.gguf")
    },
    {
        "name": "glm-edge-4b-chat", 
        "gguf_path": str(PROJECT_ROOT / "models" / "glm-edge-4b-chat" / "ggml-model-Q4_K_M.gguf")
    },
    {
        "name": "Phi-4-mini-instruct", 
        "gguf_path": str(PROJECT_ROOT / "models" / "Phi-4-mini-instruct" / "phi-4-mini-reasoning.Q4_K_M.gguf")
    },
    {
        "name": "Qwen3.5-4B", 
        "gguf_path": str(PROJECT_ROOT / "models" / "Qwen3.5-4B" / "Qwen3.5-4B-Q4_K_M.gguf")
    },
]

# =====================
# 文本压缩专家场景统一参数
# =====================
SYSTEM_PROMPT = (
    """You are a professional context compression engine specialized in processing conversational histories with tool calls and decision-making logic.

Compression Guidelines:
1. Multi-granularity Retention: Preserve all tool execution results, IDs, key numerical values, and dates.
2. Logical Distillation: Maintain causal links and decision reasoning.
3. Aggressive Denoising: Remove chitchat, system logs, and redundant documentation.
4. Output Constraint: Output only the structured summary. No meta-commentary or self-introduction.
5. Format: Preserve the original structure where possible, use bullet points for clarity.
"""
)


# =====================
# llama-cpp 加载配置 (RTX 5060 8GB 专属优化)
# =====================
LLAMA_CPP_CONFIG = {
    "n_gpu_layers": -1,    # RTX 5060 性能强劲，设置为 -1 表示全量 offload 到 GPU
    "n_ctx": 32768,        # 满足你 32k 的业务需求
    "n_threads": 8,
    "flash_attn": True,    # 50 系列显卡必开，大幅降低显存压力
    
    # KV Cache 量化：这是 8GB 显存跑 32k 的核心！
    # type_k/v = 8 表示使用 Q8_0 量化，显存占用比 FP16 减少近一半
    "type_k": 8,           
    "type_v": 8,
    
    "n_batch": 512,        # 减少瞬时显存峰值，防止 TDR 崩溃
    "verbose": False
}

# =====================
# 模型请求级补丁参数（注入 Ollama /api/generate 请求 body 顶层）
# =====================
# 只影响指定模型，其他模型不受影响
MODEL_REQUEST_PATCHES = {
    # Qwen3 系列默认开启思考模式（CoT），对压缩任务无意义且会导致卡死
    # think=False 通过 Ollama API 顶层字段关闭，不写入 Modelfile
    "qwen3.5-4b-compress": {
        "think": False
    }
}

