"""
ppl_evaluator.py
使用 llama-3.1-8B (GGUF) 评估文本的困惑度 (Perplexity, PPL)。

困惑度越低 = 语言越流畅、越符合模型所学的语言分布。
适用于评估压缩后文本的可读性和语言质量。

注意：
- llama-3.1-8B 以英文为主，对英文文本的 PPL 评估最为准确
- 中文文本的 PPL 会天然偏高，仅供参考
- 需要本地存放 llama-3.1-8B 的 GGUF 文件

依赖安装：
    已有 llama_cpp_python，无需额外安装

模型文件放置示例：
    models/llama-3.1-8B/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from llama_cpp import Llama
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


# ──────────────────────────────────────────────
# 配置：修改为你实际的 llama-3.1-8B GGUF 路径
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = str(
    PROJECT_ROOT / "models" / "llama-3.1-8B" / "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)


class PPLEvaluator:
    """
    基于 llama.cpp 的困惑度 (PPL) 评估器。
    PPL = exp(-1/N * sum(log P(w_i | w_1..w_{i-1})))
    """

    def __init__(self, model_path: str = None, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        初始化 llama 模型

        Args:
            model_path: GGUF 模型路径，为 None 时使用默认路径
            n_ctx:       上下文窗口大小
            n_gpu_layers: GPU 加速层数，-1 表示全部使用 GPU
        """
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "缺少依赖: llama_cpp_python 未安装"
            )

        path = model_path or DEFAULT_MODEL_PATH
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到模型文件: {path}\n"
                f"请将 llama-3.1-8B 的 GGUF 文件放置到: {DEFAULT_MODEL_PATH}"
            )

        print(f"加载 llama 模型: {path}")
        self.model = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,   # 必须开启，才能获取所有 token 的 logits
            verbose=False
        )
        print("llama 模型加载完成")

    def evaluate(self, text: str, max_tokens: int = 2048) -> dict:
        """
        计算文本的困惑度 (PPL)

        Args:
            text:       待评估文本
            max_tokens: 最大 token 数量（超出则截断）

        Returns:
            dict: {
                'ppl': float,          # 困惑度，越低越好
                'avg_log_prob': float, # 平均 log 概率
                'token_count': int     # token 数量
            }
        """
        if not text or len(text.strip()) < 5:
            return {"ppl": float("inf"), "error": "输入文本过短"}

        # tokenize 文本
        tokens = self.model.tokenize(text.encode("utf-8"))

        # 超出 max_tokens 时截断
        if len(tokens) > max_tokens:
            print(f"  ⚠️  文本过长({len(tokens)} tokens)，截断至 {max_tokens}")
            tokens = tokens[:max_tokens]

        if len(tokens) < 2:
            return {"ppl": float("inf"), "error": "token 数量不足"}

        # 使用 logits_all=True 模式，获取每个 token 的 log prob
        output = self.model(
            text,
            max_tokens=1,            # 不需要生成新 token
            echo=True,               # 输出原始 token 的 logprobs
            logprobs=1               # 返回 top-1 logprobs
        )

        # 提取每个 token 的 log probability
        choices = output.get("choices", [])
        if not choices:
            return {"ppl": float("inf"), "error": "模型未返回 logprobs"}

        logprobs_info = choices[0].get("logprobs", {})
        token_logprobs = logprobs_info.get("token_logprobs", [])

        # 过滤掉 None（通常是第一个 token）
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]

        if not valid_logprobs:
            return {"ppl": float("inf"), "error": "无有效的 logprobs"}

        avg_log_prob = sum(valid_logprobs) / len(valid_logprobs)
        ppl = math.exp(-avg_log_prob)

        return {
            "ppl": round(ppl, 4),
            "avg_log_prob": round(avg_log_prob, 6),
            "token_count": len(valid_logprobs),
        }

    def batch_evaluate(self, texts: list) -> list:
        """
        批量评估多个文本的 PPL

        Args:
            texts: 文本列表

        Returns:
            list of result dicts
        """
        results = []
        for text in texts:
            results.append(self.evaluate(text))
        return results


# ──────────────────────────────────────────────
# 单独运行时的快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import json

    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = project_root / "results"
    compressed_files = list(results_dir.glob("test_a_compress_*.txt"))

    if not compressed_files:
        print("未找到压缩结果文件")
        sys.exit(1)

    print("=" * 80)
    print("困惑度评估 (llama-3.1-8B)")
    print("=" * 80)

    evaluator = PPLEvaluator()

    for cf in sorted(compressed_files):
        with open(cf, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取 "压缩结果:" 之后的部分
        if "压缩结果:" in content:
            compressed_text = content.split("压缩结果:")[-1].strip()
            compressed_text = compressed_text.lstrip("=" * 10).strip()
        else:
            compressed_text = content

        if not compressed_text or len(compressed_text) < 10:
            print(f"{cf.stem}: ⚠️  压缩结果为空，跳过")
            continue

        result = evaluator.evaluate(compressed_text)
        print(f"\n模型: {cf.stem}")
        if "error" in result:
            print(f"  ❌ 错误: {result['error']}")
        else:
            print(f"  PPL:         {result['ppl']:.4f}  (越低越好)")
            print(f"  Token数:     {result['token_count']}")
            print(f"  平均LogProb: {result['avg_log_prob']:.6f}")
