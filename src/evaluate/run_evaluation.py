"""
run_evaluation.py
评估统一入口：读取 results/ 中所有压缩结果，对每个结果运行三项评估，
最终输出汇总报告到 results/evaluation_report.json。

用法：
    python -m src.evaluate.run_evaluation

可选参数（修改文件开头的常量）：
    ENABLE_SEMANTIC  = True   # 是否运行语义相似度评估 (BGE-M3)
    ENABLE_PPL       = True   # 是否运行困惑度评估 (llama-3.1-8B)
    ENABLE_RECALL    = True   # 是否运行召回率评估 (Qwen API)
    TEST_FILES       = None   # 为 None 时评估所有 test_*.json，可指定如 ["test_a"]
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ──────────────────────────────────────────────
# 配置开关：按需开启/关闭评估项
# ──────────────────────────────────────────────
ENABLE_SEMANTIC = True     # BGE-M3 语义相似度
ENABLE_PPL      = True     # llama-3.1-8B 困惑度
ENABLE_RECALL   = True     # Qwen API 召回率

# 指定要评估的测试文件前缀，None = 全部
TEST_FILES = None          # 例如: ["test_a", "test_b"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"


def load_original(test_name: str) -> str:
    """读取原始测试文件并序列化为字符串"""
    path = DATA_DIR / f"{test_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), ensure_ascii=False, indent=2)


def extract_compressed_text(txt_path: Path) -> str:
    """从 results/*.txt 中提取压缩结果部分"""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "压缩结果:" in content:
        compressed = content.split("压缩结果:")[-1]
        # 去掉分隔线
        lines = [l for l in compressed.splitlines() if not set(l.strip()) <= {"=", ""}]
        return "\n".join(lines).strip()
    return content.strip()


def get_model_name(filename_stem: str) -> str:
    """从文件名中提取模型名称"""
    # test_a_compress_gemma4-e4b-compress → gemma4-e4b-compress
    parts = filename_stem.split("_compress_")
    return parts[-1] if len(parts) > 1 else filename_stem


def run_evaluation():
    print("=" * 80)
    print("压缩质量综合评估")
    print("=" * 80)

    # ── 初始化评估器 ──────────────────────────────
    semantic_evaluator = None
    ppl_evaluator = None
    recall_evaluator = None

    if ENABLE_SEMANTIC:
        try:
            from src.evaluate.semantic_evaluator import SemanticEvaluator
            semantic_evaluator = SemanticEvaluator()
            print("✅ BGE-M3 语义相似度评估器已加载")
        except Exception as e:
            print(f"⚠️  BGE-M3 加载失败，跳过语义评估: {e}")

    if ENABLE_PPL:
        try:
            from src.evaluate.ppl_evaluator import PPLEvaluator
            ppl_evaluator = PPLEvaluator()
            print("✅ llama-3.1-8B PPL 评估器已加载")
        except Exception as e:
            print(f"⚠️  llama 加载失败，跳过 PPL 评估: {e}")

    if ENABLE_RECALL:
        try:
            from src.evaluate.recall_evaluator import RecallEvaluator
            recall_evaluator = RecallEvaluator()
            print("✅ Qwen API 召回率评估器已加载")
        except Exception as e:
            print(f"⚠️  Qwen API 初始化失败，跳过召回率评估: {e}")

    # ── 收集要评估的测试文件 ──────────────────────
    if TEST_FILES:
        test_names = TEST_FILES
    else:
        test_names = sorted({
            p.stem.split("_compress_")[0]
            for p in RESULTS_DIR.glob("*_compress_*.txt")
        })

    if not test_names:
        print("❌ 未找到任何压缩结果文件")
        sys.exit(1)

    print(f"\n📂 待评估测试集: {test_names}")

    # ── 开始评估 ──────────────────────────────────
    report = {}
    total_start = time.time()

    for test_name in test_names:
        print(f"\n{'─' * 60}")
        print(f"📄 评估测试集: {test_name}")
        print(f"{'─' * 60}")

        try:
            original_text = load_original(test_name)
        except FileNotFoundError:
            print(f"⚠️  原始文件不存在: {DATA_DIR / test_name}.json，跳过")
            continue

        report[test_name] = {}

        # 找到该测试集的所有压缩结果
        compress_files = sorted(RESULTS_DIR.glob(f"{test_name}_compress_*.txt"))

        if not compress_files:
            print(f"⚠️  未找到 {test_name} 的压缩结果")
            continue

        for cf in compress_files:
            model_name = get_model_name(cf.stem)
            print(f"\n  🔍 模型: {model_name}")

            compressed_text = extract_compressed_text(cf)

            if not compressed_text or len(compressed_text) < 10:
                print(f"    ⚠️  压缩结果为空，跳过")
                report[test_name][model_name] = {"error": "压缩结果为空"}
                continue

            model_result = {
                "compressed_length": len(compressed_text),
                "original_length": len(original_text),
                "compression_ratio": round(
                    1 - len(compressed_text) / len(original_text), 4
                ),
            }

            # 语义相似度
            if semantic_evaluator:
                try:
                    sem = semantic_evaluator.evaluate(original_text, compressed_text)
                    model_result["semantic_similarity"] = sem.get("semantic_similarity")
                    print(f"    语义相似度: {model_result['semantic_similarity']:.4f}")
                except Exception as e:
                    model_result["semantic_similarity_error"] = str(e)
                    print(f"    ❌ 语义评估失败: {e}")

            # 困惑度
            if ppl_evaluator:
                try:
                    ppl = ppl_evaluator.evaluate(compressed_text)
                    model_result["ppl"] = ppl.get("ppl")
                    model_result["ppl_token_count"] = ppl.get("token_count")
                    print(f"    PPL:         {model_result['ppl']:.4f}")
                except Exception as e:
                    model_result["ppl_error"] = str(e)
                    print(f"    ❌ PPL 评估失败: {e}")

            # 召回率
            if recall_evaluator:
                try:
                    recall = recall_evaluator.evaluate(original_text, compressed_text)
                    model_result["recall_rate"] = recall.get("recall_rate")
                    model_result["total_key_items"] = recall.get("total_items")
                    model_result["retained_key_items"] = recall.get("retained_items")
                    print(f"    召回率:      {model_result['recall_rate']:.4f}")
                except Exception as e:
                    model_result["recall_error"] = str(e)
                    print(f"    ❌ 召回率评估失败: {e}")

            report[test_name][model_name] = model_result

    # ── 保存报告 ──────────────────────────────────
    total_time = time.time() - total_start
    report["_meta"] = {
        "total_time_s": round(total_time, 2),
        "test_files": test_names,
        "evaluators": {
            "semantic": ENABLE_SEMANTIC and semantic_evaluator is not None,
            "ppl":      ENABLE_PPL and ppl_evaluator is not None,
            "recall":   ENABLE_RECALL and recall_evaluator is not None,
        }
    }

    report_path = RESULTS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✅ 评估完成！总耗时: {total_time:.2f}s")
    print(f"📊 报告已保存: {report_path}")
    print(f"{'=' * 80}")

    # ── 输出汇总表格 ──────────────────────────────
    print("\n📊 汇总结果:")
    header = f"{'测试集':<10} {'模型':<35} {'语义相似度':<14} {'PPL':<12} {'召回率':<10} {'压缩率'}"
    print(header)
    print("-" * len(header))

    for test_name, models in report.items():
        if test_name == "_meta":
            continue
        for model_name, result in models.items():
            if "error" in result:
                continue
            sem  = f"{result.get('semantic_similarity', 'N/A'):.4f}" if isinstance(result.get('semantic_similarity'), float) else "N/A"
            ppl  = f"{result.get('ppl', 'N/A'):.2f}" if isinstance(result.get('ppl'), float) else "N/A"
            rec  = f"{result.get('recall_rate', 'N/A'):.4f}" if isinstance(result.get('recall_rate'), float) else "N/A"
            comp = f"{result.get('compression_ratio', 0):.2%}"
            print(f"{test_name:<10} {model_name:<35} {sem:<14} {ppl:<12} {rec:<10} {comp}")


if __name__ == "__main__":
    run_evaluation()
