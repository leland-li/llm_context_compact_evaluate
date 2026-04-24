"""
重建 qwen3.5-4b-compress 的 Modelfile
修复：
  - Qwen: TEMPLATE 中 assistant 预填充 <think>\n\n</think>，在 token 序列层面关闭思考模式
  - /no_think 软提示对 GGUF 无效，改用预填充硬约束
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a professional context compression engine specialized in processing "
    "conversational histories with tool calls and decision-making logic.\n\n"
    "Compression Guidelines:\n"
    "1. Multi-granularity Retention: Preserve all tool execution results, IDs, key numerical values, and dates.\n"
    "2. Logical Distillation: Maintain causal links and decision reasoning.\n"
    "3. Aggressive Denoising: Remove chitchat, system logs, and redundant documentation.\n"
    "4. Output Constraint: Output only the structured summary. No meta-commentary or self-introduction.\n"
    "5. Format: Preserve the original structure where possible, use bullet points for clarity."
)

MODELS = [
    {
        "name": "qwen3.5-4b-compress",
        "gguf": ROOT / "models" / "Qwen3.5-4B" / "Qwen3.5-4B-Q4_K_M.gguf",
        # Qwen3.5 ChatML template：assistant 预填充 <think>\n\n</think>
        # 模型看到思考块已关闭（空），直接输出答案，token 级硬约束
        "template": (
            r"{{ if .System }}<|im_start|>system" + "\n"
            r"{{ .System }}<|im_end|>" + "\n"
            r"{{ end }}"
            r"{{ if .Prompt }}<|im_start|>user" + "\n"
            r"{{ .Prompt }}<|im_end|>" + "\n"
            r"{{ end }}<|im_start|>assistant" + "\n"
            "<think>\n\n</think>\n"
            r"{{ .Response }}<|im_end|>"
        ),
        "num_ctx": 32768,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>", "</tool_call>"],
    },
]


def build_modelfile(model: dict) -> str:
    gguf = model["gguf"]
    stop_lines = "\n".join(f'PARAMETER stop "{t}"' for t in model["stop_tokens"])
    return (
        f'FROM "{gguf}"\n'
        f'TEMPLATE "{model["template"]}"\n'
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
        f'PARAMETER num_ctx {model["num_ctx"]}\n'
        f'PARAMETER temperature 0.1\n'
        f'PARAMETER top_p 0.85\n'
        f'PARAMETER top_k 40\n'
        f'PARAMETER repeat_penalty 1.15\n'
        f'{stop_lines}\n'
    )


def main():
    for model in MODELS:
        name = model["name"]
        print(f"\n{'='*60}")
        print(f"重建模型: {name}")

        if not model["gguf"].exists():
            print(f"  ERROR: GGUF 文件不存在: {model['gguf']}")
            continue

        content = build_modelfile(model)
        tmp = ROOT / "Modelfile.tmp"
        tmp.write_text(content, encoding="utf-8")

        print("  Modelfile stop tokens:")
        for t in model["stop_tokens"]:
            print(f"    PARAMETER stop \"{t}\"")
        if "<think>\n\n</think>" in content:
            print("  TEMPLATE: <think>\\n\\n</think> 预填充已注入 ✓")

        print("  正在重建...")
        result = subprocess.run(
            ["ollama", "create", name, "-f", str(tmp)],
            capture_output=False,
        )
        tmp.unlink(missing_ok=True)

        if result.returncode == 0:
            print(f"  ✅ 重建成功: {name}")
        else:
            print(f"  ❌ 重建失败: {name} (exit code {result.returncode})")

    print(f"\n{'='*60}")
    print("验证重建结果:")
    subprocess.run(["ollama", "list"])


if __name__ == "__main__":
    main()
