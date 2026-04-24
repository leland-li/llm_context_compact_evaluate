"""
recall_evaluator.py
使用 Qwen API（LLM-as-Judge 模式）评估压缩文本的关键信息召回率。

工作原理：
1. 让 Qwen 从原文中提取所有关键信息（数字、ID、日期、决策、实体）
2. 让 Qwen 判断每条关键信息是否在压缩文中被保留
3. 计算召回率 = 保留数 / 总数

配置说明：
    请在下方填写你的 QWEN_API_KEY 和 QWEN_BASE_URL
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import requests


# ──────────────────────────────────────────────
# ⚠️ 请在这里填写你的 Qwen API 配置
# ──────────────────────────────────────────────
QWEN_API_KEY = "sk-3802af2c3c5d49d0803bf7f56f5be945"        # 替换为你的 API Key
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 DashScope 兼容接口（国内版）
QWEN_MODEL = "qwen3-max"                        # 可选: qwen-turbo / qwen-plus / qwen-max


# ──────────────────────────────────────────────
# Prompt 模板
# ──────────────────────────────────────────────
EXTRACT_PROMPT = """\
You are an information extraction expert.

Given the following original text, extract ALL key information items that must be preserved in a compressed summary.

Key information includes:
- Numerical values (amounts, percentages, quantities)
- Identifiers (IDs, task IDs, user IDs, session IDs)
- Dates and timestamps
- Named entities (people, organizations, products, assets)
- Decisions and action results
- Tool call results

Original text:
{original}

Output ONLY a JSON array of strings. Example:
["TASK-8890", "2026-04-27", "1,250,000 CNY", "Gold", "+6.1%"]
"""

RECALL_PROMPT = """\
You are an information recall evaluator.

Given the key information items extracted from an original text, determine which items are retained in the compressed text.

Key information items:
{key_items}

Compressed text:
{compressed}

For each item, determine if the information (or its equivalent meaning) is present in the compressed text.
Output ONLY a JSON object. Example:
{{
  "evaluations": [
    {{"item": "TASK-8890", "retained": true, "reason": "explicitly mentioned"}},
    {{"item": "2026-04-28", "retained": false, "reason": "date not found"}}
  ],
  "recall_rate": 0.85
}}
"""


class RecallEvaluator:
    """
    基于 Qwen API (LLM-as-Judge) 的关键信息召回率评估器。
    """

    def __init__(
        self,
        api_key: str = QWEN_API_KEY,
        base_url: str = QWEN_BASE_URL,
        model: str = QWEN_MODEL,
        request_timeout: int = 60,
        retry_times: int = 3,
        retry_delay: float = 2.0
    ):
        """
        初始化 Qwen API 客户端

        Args:
            api_key:         Qwen API Key
            base_url:        API 基础 URL
            model:           使用的模型名称
            request_timeout: 请求超时秒数
            retry_times:     失败重试次数
            retry_delay:     重试间隔秒数
        """
        if api_key == "YOUR_QWEN_API_KEY_HERE":
            raise ValueError(
                "请先在 recall_evaluator.py 中配置 QWEN_API_KEY"
            )

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = request_timeout
        self.retry_times = retry_times
        self.retry_delay = retry_delay

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat(self, prompt: str) -> str:
        """
        调用 Qwen Chat API

        Args:
            prompt: 用户 prompt

        Returns:
            模型回复的文本
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,   # 低温度，保证输出稳定
            "max_tokens": 2048,
        }

        for attempt in range(self.retry_times):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < self.retry_times - 1:
                    print(f"  ⚠️  API 调用失败 (尝试 {attempt+1}/{self.retry_times}): {str(e)}")
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"API 调用失败: {str(e)}")

    def _extract_key_items(self, original: str) -> list:
        """
        步骤1：从原文中提取关键信息列表

        Args:
            original: 原始文本

        Returns:
            关键信息字符串列表
        """
        prompt = EXTRACT_PROMPT.format(original=original[:4000])  # 限制输入长度
        response = self._chat(prompt)

        # 从回复中解析 JSON 数组
        try:
            # 尝试直接解析
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # 尝试提取 [] 之间的内容
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            return []

    def _evaluate_recall(self, key_items: list, compressed: str) -> dict:
        """
        步骤2：评估关键信息在压缩文中的保留情况

        Args:
            key_items:  关键信息列表
            compressed: 压缩后的文本

        Returns:
            评估结果 dict
        """
        if not key_items:
            return {"evaluations": [], "recall_rate": 0.0, "error": "未提取到关键信息"}

        prompt = RECALL_PROMPT.format(
            key_items=json.dumps(key_items, ensure_ascii=False),
            compressed=compressed[:4000]
        )
        response = self._chat(prompt)

        # 从回复中解析 JSON 对象
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
            return {"evaluations": [], "recall_rate": 0.0, "error": "JSON 解析失败"}

    def evaluate(self, original: str, compressed: str) -> dict:
        """
        完整评估流程：提取关键信息 → 判断保留情况 → 计算召回率

        Args:
            original:   原始文本
            compressed: 压缩后的文本

        Returns:
            dict: {
                'recall_rate': float,       # 0~1，越高越好
                'key_items': list,          # 提取到的关键信息
                'evaluations': list,        # 每条信息的保留判断
                'total_items': int,
                'retained_items': int
            }
        """
        if not original or not compressed:
            return {"recall_rate": 0.0, "error": "输入文本为空"}

        # 步骤1：提取关键信息
        key_items = self._extract_key_items(original)
        if not key_items:
            return {"recall_rate": 0.0, "error": "未能提取关键信息"}

        # 步骤2：评估召回率
        eval_result = self._evaluate_recall(key_items, compressed)

        # 统计数据
        evaluations = eval_result.get("evaluations", [])
        retained = sum(1 for e in evaluations if e.get("retained", False))
        total = len(evaluations) or len(key_items)

        # 优先使用模型计算的 recall_rate，不然自己算
        recall_rate = eval_result.get("recall_rate", retained / total if total > 0 else 0.0)

        return {
            "recall_rate": round(float(recall_rate), 4),
            "key_items": key_items,
            "evaluations": evaluations,
            "total_items": total,
            "retained_items": retained,
        }


# ──────────────────────────────────────────────
# 单独运行时的快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
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
    print("关键信息召回率评估 (Qwen API / LLM-as-Judge)")
    print("=" * 80)

    evaluator = RecallEvaluator()

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

        print(f"\n模型: {cf.stem}")
        try:
            result = evaluator.evaluate(original_text, compressed_text)
            if "error" in result:
                print(f"  ❌ 错误: {result['error']}")
            else:
                print(f"  召回率:      {result['recall_rate']:.4f}  (越高越好)")
                print(f"  关键信息数:  {result['total_items']}")
                print(f"  保留数:      {result['retained_items']}")
                print(f"  提取的关键信息: {result['key_items']}")
        except Exception as e:
            print(f"  ❌ 评估失败: {str(e)}")
