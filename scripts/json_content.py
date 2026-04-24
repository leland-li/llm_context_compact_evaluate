import json
from pathlib import Path

system_prompt = """You are a professional context compression engine specialized in processing conversational histories with tool calls and decision-making logic.

Compression Guidelines:
1. Multi-granularity Retention: Preserve all tool execution results, IDs, key numerical values, and dates.
2. Logical Distillation: Maintain causal links and decision reasoning.
3. Aggressive Denoising: Remove chitchat, system logs, and redundant documentation.
4. Output Constraint: Output only the structured summary. No meta-commentary or self-introduction.
5. Format: Preserve the original structure where possible, use bullet points for clarity.
"""

try:
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    test_file = project_root / "data" / "test_a.json"
    
    print(f"📂 项目根目录: {project_root}")
    print(f"📄 尝试读取: {test_file}")
    
    if not test_file.exists():
        print(f"❌ 文件不存在: {test_file}")
        exit(1)
    
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = f.read()
    
    print(f"✓ 已读取 {len(test_data)} 字符")
    
    full_question = system_prompt + "\n\n" + test_data
    print(f"✓ 拼接完成，总长度: {len(full_question)} 字符")
    
    payload = {
        "question": full_question,
        "uuid": "test-uuid",
        "kbqa": False
    }
    
    # 输出到控制台
    json_output = json.dumps(payload, ensure_ascii=False, indent=2)
    print("\n" + "="*80)
    print("生成的 Postman Body:")
    print("="*80)
    print(json_output)
    
    # 同时保存到文件
    output_file = project_root / "postman_payload.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json_output)
    
    print(f"\n✅ 已保存到: {output_file}")
    
except Exception as e:
    print(f"❌ 出错: {str(e)}")
    import traceback
    traceback.print_exc()