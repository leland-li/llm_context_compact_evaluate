"""
semantic_evaluator.py
使用 BGE-M3 模型评估原文与压缩文本的语义相似度。

依赖安装：
    pip install FlagEmbedding scikit-learn

BGE-M3 模型会在首次运行时自动从 HuggingFace 下载。
如需离线使用，可手动下载并指定本地路径：
    model = BGEM3FlagModel('path/to/bge-m3', use_fp16=True)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

try:
    from FlagEmbedding import BGEM3FlagModel
    from sklearn.metrics.pairwise import cosine_similarity
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class SemanticEvaluator:
    """
    基于 BGE-M3 的语义相似度评估器。
    支持中英文混合文本。
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, model_path: str = None, use_fp16: bool = True):
        """
        初始化 BGE-M3 模型

        Args:
            model_path: 本地模型路径，为 None 时使用 HuggingFace 自动下载
            use_fp16: 是否使用半精度浮点加速，推荐开启
        """
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "缺少依赖，请运行: pip install FlagEmbedding scikit-learn"
            )

        path = model_path or self.MODEL_NAME
        print(f"加载 BGE-M3 模型: {path}")
        self.model = BGEM3FlagModel(path, use_fp16=use_fp16)
        print("BGE-M3 模型加载完成")

    def _encode(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        result = self.model.encode(
            [text],
            batch_size=1,
            max_length=8192,       # BGE-M3 最大支持 8192 tokens
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return result["dense_vecs"][0]

    def evaluate(self, original: str, compressed: str) -> dict:
        """
        计算语义相似度

        Args:
            original: 原始文本
            compressed: 压缩后的文本

        Returns:
            dict: {
                'semantic_similarity': float,  # 0~1，越高越好
                'original_length': int,
                'compressed_length': int
            }
        """
        if not original or not compressed:
            return {"semantic_similarity": 0.0, "error": "输入文本为空"}

        emb_orig = self._encode(original)
        emb_comp = self._encode(compressed)

        sim = float(
            cosine_similarity(
                emb_orig.reshape(1, -1),
                emb_comp.reshape(1, -1)
            )[0][0]
        )

        return {
            "semantic_similarity": round(sim, 4),
            "original_length": len(original),
            "compressed_length": len(compressed),
        }

    def batch_evaluate(self, pairs: list) -> list:
        """
        批量评估多个 (原文, 压缩文) 对

        Args:
            pairs: list of (original, compressed) tuples

        Returns:
            list of result dicts
        """
        results = []
        for original, compressed in pairs:
            results.append(self.evaluate(original, compressed))
        return results


# ──────────────────────────────────────────────
# 单独运行时的快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import json

    project_root = Path(__file__).resolve().parent.parent.parent

    # 读取原始文本
    original_file = project_root / "data" / "test_a.json"
    with open(original_file, "r", encoding="utf-8") as f:
        original_text = json.dumps(json.load(f), ensure_ascii=False, indent=2)

    # 读取各模型的压缩结果
    results_dir = project_root / "results"
    compressed_files = list(results_dir.glob("test_a_compress_*.txt"))

    if not compressed_files:
        print("未找到压缩结果文件")
        sys.exit(1)

    print("=" * 80)
    print("语义相似度评估 (BGE-M3)")
    print("=" * 80)

    evaluator = SemanticEvaluator()

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

        result = evaluator.evaluate(original_text, compressed_text)
        print(f"\n模型: {cf.stem}")
        print(f"  语义相似度:  {result['semantic_similarity']:.4f}")
        print(f"  原文长度:    {result['original_length']} 字符")
        print(f"  压缩文长度:  {result['compressed_length']} 字符")
