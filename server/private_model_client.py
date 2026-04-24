"""
private_model_server.py
私有模型服务，暴露 generate 方法。
"""
import requests
import uuid
import json
import time
import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYSTEM_PROMPT

class PfmModelAdapter:
    def __init__(self, base_url="http://127.0.0.1:9090"):
        self.base_url = f"{base_url}/api/v1/interact"
        self.session = requests.Session()
        self.session.trust_env = False
        
        # 上下文窗口配置
        self.max_context_window = 6000        # 模型最大上下文
        self.system_prompt_size = 500         # 系统提示词大小估计
        self.safety_buffer = 1000             # 安全buffer（应对tokenizer差异）
        self.safe_chunk_size = (
            self.max_context_window - 
            self.system_prompt_size - 
            self.safety_buffer
        )  # = 4500
        self.max_recursion_depth = 8          # 最多递归3层

    def _read_file(self, file_path: str) -> str:
        """读取数据文件内容"""
        try:
            path = Path(file_path)
            if not path.exists():
                # 尝试从 data 目录读取
                path = Path(__file__).resolve().parent.parent / "data" / file_path
            
            if path.suffix == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    return json.dumps(content, ensure_ascii=False, indent=2)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def _get_new_session(self):
        url = f"{self.base_url}/session"
        payload = {"name": "benchmark_task", "sessionType": "LOCAL"}
        try:
            res = self.session.post(url, json=payload, timeout=10)
            data = res.json().get("data")
            if isinstance(data, dict):
                return data.get("id", data)
            return data
        except Exception:
            return None

    def _chunk_text(self, text: str, max_length: int = None) -> list:
        """
        智能分割文本，保留完整段落/句子
        
        Args:
            text: 要分割的文本
            max_length: 每块的最大长度（默认使用 safe_chunk_size）
        
        Returns:
            分割后的文本列表
        """
        if max_length is None:
            max_length = self.safe_chunk_size
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        # 先按段落分割（\n\n）
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_length:  # +2 for \n\n
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # 如果单个段落超长，按句子再分割
                if len(para) > max_length:
                    sentences = para.replace('。', '。\n').split('\n')
                    for sent in sentences:
                        if len(sent) > 0:
                            if len(current_chunk) + len(sent) <= max_length:
                                current_chunk = sent + "\n"
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sent + "\n"
                else:
                    current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _merge_compressions(self, results: list) -> str:
        """
        合并多个压缩结果
        
        Args:
            results: 压缩结果列表
        
        Returns:
            合并后的文本
        """
        return "\n".join([r for r in results if r and not r.startswith("Error")])
    
    def _recursive_compress(self, text: str, depth: int = 0) -> str:
        """
        递归压缩超长文本
        
        Args:
            text: 要压缩的文本
            depth: 当前递归深度
        
        Returns:
            压缩后的文本
        """
        # 防止过度递归
        if depth >= self.max_recursion_depth:
            print(f"⚠️  达到最大递归深度({self.max_recursion_depth}层)，停止递归")
            return text
        
        # 如果文本已经足够短，直接压缩
        if len(text) <= self.safe_chunk_size:
            response, _, _, _ = self.stream_inference(text)
            return response
        
        # 分块处理
        print(f"  [递归深度{depth}] 文本长度{len(text)}超过阈值({self.safe_chunk_size})，分块处理...")
        chunks = self._chunk_text(text, max_length=self.safe_chunk_size)
        print(f"  [递归深度{depth}] 分成{len(chunks)}块")
        
        results = []
        for i, chunk in enumerate(chunks):
            print(f"    处理第{i+1}/{len(chunks)}块...")
            response, _, _, _ = self.stream_inference(chunk)
            if response and not response.startswith("Error"):
                results.append(response)
        
        # 合并结果
        merged = self._merge_compressions(results)
        print(f"  [递归深度{depth}] 合并后长度: {len(merged)}")
        
        # 如果合并结果还是太长，继续递归
        if len(merged) > self.safe_chunk_size:
            print(f"  [递归深度{depth}] 合并结果仍超长，继续递归...")
            return self._recursive_compress(merged, depth + 1)
        else:
            return merged

    def stream_inference(self, prompt: str, use_system_prompt: bool = True) -> tuple:
        """
        流式推理接口，收集性能指标
        
        Returns:
            tuple: (response_text, first_token_time, total_time, token_count)
        """
        # 支持从文件读取内容
        if prompt.endswith((".json", ".txt")):
            file_content = self._read_file(prompt)
            if file_content.startswith("Error"):
                return file_content, 0, 0, 0
            prompt = file_content
        
        # 构造完整的提示词：系统提示词 + 文本
        if use_system_prompt:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        sid = self._get_new_session()
        if not sid:
            return "Error: Could not create session", 0, 0, 0
        
        url = f"{self.base_url}/session/{sid}"
        payload = {
            "question": full_prompt,
            "uuid": str(uuid.uuid4()),
            "kbqa": False
        }
        try:
            response = self.session.post(url, json=payload, timeout=120, stream=True)
            result = []
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data:"):
                        content = decoded_line.replace("data:", "").strip()
                        try:
                            inner_json = json.loads(content)
                            # 记录首 token 时间
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            
                            # 跳过开始和结束标记
                            if "message" in inner_json:
                                msg = inner_json["message"]
                                if msg not in ("start answering", "end answering"):
                                    result.append(msg)
                                    token_count += 1
                            # 提取最终答案
                            if "FINAL_ANSWER" in inner_json:
                                final_ans = inner_json["FINAL_ANSWER"]
                                result.append(final_ans)
                                token_count += 1
                        except Exception:
                            continue
            
            total_time = time.time() - start_time
            if first_token_time is None:
                first_token_time = total_time
            
            response_text = "".join(result) if result else "Error: No answer returned"
            return response_text, first_token_time, total_time, token_count
        except Exception as e:
            total_time = time.time() - start_time
            return f"Inference Failed: {str(e)}", 0, total_time, 0

def save_results(results: dict, test_file_name: str, results_dir: str = "results") -> None:
    """
    保存推理结果到文件
    
    Args:
        results: 推理结果字典
        test_file_name: 测试数据文件名（不含扩展名）
        results_dir: 结果保存目录
    """
    # 创建结果目录
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    # 生成结果文件名
    result_base_name = f"{test_file_name}_compress_private_model"
    
    # 1. 保存统计汇总到 JSON
    stats_summary = {}
    for model_key, result in results.items():
        if 'error' not in result:
            stats_summary[model_key] = {
                'coldstart_time_s': round(result['coldstart_time'], 2),
                'ttft_ms': round(result['ttft'] * 1000, 2),
                'total_time_s': round(result['total_time'], 2),
                'throughput_token_s': round(result['throughput'], 2),
                'token_count': result['token_count'],
                'input_length': result['input_length'],
                'output_length': result['output_length'],
                'compression_ratio': round(result['compression_ratio'], 2)
            }
    
    summary_file = results_path / f"{result_base_name}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 统计汇总已保存: {summary_file}")
    
    # 2. 为模型保存单独的压缩结果
    for model_key, result in results.items():
        if 'error' not in result:
            model_file = results_path / f"{result_base_name}.txt"
            with open(model_file, 'w', encoding='utf-8') as f:
                f.write(f"模型: Private Model Server\n")
                f.write(f"冷启动时间: {result['coldstart_time']:.2f}s\n")
                f.write(f"首Token响应时间(TTFT): {result['ttft']*1000:.2f}ms (热启)\n")
                f.write(f"总响应时间: {result['total_time']:.2f}s (热启)\n")
                f.write(f"吞吐量: {result['throughput']:.2f} token/s (热启)\n")
                f.write(f"Token数: {result['token_count']}\n")
                f.write(f"输入字符数: {result['input_length']}\n")
                f.write(f"输出字符数: {result['output_length']}\n")
                f.write(f"压缩比: {result['compression_ratio']:.2f}x\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"压缩结果:\n")
                f.write(f"{'='*80}\n")
                f.write(result['output'])
            print(f"✅ 模型结果已保存: {model_file}")

# 模块级单例
_adapter = PfmModelAdapter()

def infer(text: str) -> str:
    """
    统一API网关调用入口，兼容 transformers 风格（仅返回文本）。
    :param text: 输入文本
    :return: 推理结果字符串
    """
    response_text, _, _, _ = _adapter.stream_inference(text)
    return response_text

def infer_with_metrics(text: str) -> tuple:
    """
    带性能指标的推理接口
    :param text: 输入文本
    :return: tuple (response_text, first_token_time, total_time, token_count)
    """
    return _adapter.stream_inference(text)

def compress_with_chunking(text: str) -> tuple:
    """
    带分块处理的推理接口（自动处理超长文本）
    :param text: 输入文本或文件路径
    :return: tuple (response_text, first_token_time, total_time, token_count)
    """
    # 支持文件路径
    if text.endswith((".json", ".txt")):
        file_content = _adapter._read_file(text)
        if file_content.startswith("Error"):
            return file_content, 0, 0, 0
        text = file_content
    
    start_time = time.time()
    
    # 判断是否需要分块
    if len(text) <= _adapter.safe_chunk_size:
        print(f"✓ 文本长度{len(text)} < 安全阈值{_adapter.safe_chunk_size}，直接压缩")
        response, ttft, total_time, token_count = _adapter.stream_inference(text)
    else:
        print(f"✗ 文本长度{len(text)} > 安全阈值{_adapter.safe_chunk_size}，启动分块递归压缩")
        response = _adapter._recursive_compress(text, depth=0)
        total_time = time.time() - start_time
        ttft = total_time  # 近似值
        token_count = len(response) // 4  # 粗略估计
    
    return response, ttft, total_time, token_count

def load_test_file(filepath: str) -> str:
    """加载测试数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # print("="*80)
    # print("私有模型服务压缩测试")
    # print("="*80)
    
    # # 加载测试数据
    # test_files = ["test_a.json"]
    
    # for test_file in test_files:
    #     test_file_path = Path(__file__).resolve().parent.parent / "data" / test_file
    #     test_file_name = test_file.replace(".json", "")
        
    #     if not test_file_path.exists():
    #         print(f"\n⚠️  测试文件不存在: {test_file_path}")
    #         continue
        
    #     print(f"\n📂 测试文件: {test_file_name}")
        
    #     # 加载测试文本
    #     try:
    #         test_text = load_test_file(str(test_file_path))
    #         print(f"✓ 测试文本长度: {len(test_text)} 字符")
    #     except Exception as e:
    #         print(f"✗ 加载失败: {str(e)}")
    #         continue
        
    #     results = {}
        
    #     try:
    #         # 1. 执行预热（冷启动）
    #         print("   预热中...")
    #         warmup_response, warmup_ttft, warmup_time, warmup_tokens = compress_with_chunking(test_file)
            
    #         # 2. 执行实际推理（热启动）
    #         print("   推理中...")
    #         response_text, ttft, total_time, token_count = compress_with_chunking(test_file)
            
    #         # 计算吞吐量 (token/s)
    #         throughput = token_count / total_time if total_time > 0 else 0
            
    #         results['private_model'] = {
    #             'output': response_text,
    #             'coldstart_time': warmup_time,  # 冷启时间
    #             'ttft': ttft,  # 首 token 响应时间（热启）
    #             'total_time': total_time,  # 总响应时间（热启）
    #             'token_count': token_count,  # token 数量
    #             'throughput': throughput,  # 吞吐量 (token/s)
    #             'input_length': len(test_text),
    #             'output_length': len(response_text),
    #             'compression_ratio': len(test_text) / len(response_text) if len(response_text) > 0 else 0
    #         }
            
    #         print(f"  ✓ 完成 (冷启: {warmup_time:.2f}s, TTFT: {ttft:.3f}s, 总耗时: {total_time:.2f}s, 吞吐量: {throughput:.2f} token/s)")
        
    #     except Exception as e:
    #         print(f"  ✗ 失败: {str(e)}")
    #         results['private_model'] = {'error': str(e)}
        
    #     # 保存结果到文件
    #     save_results(results, test_file_name)
    
    # # 输出完成
    # print("\n" + "="*80)
    # print("✅ 所有测试已完成")
    # print("="*80)
     # 测试分块功能
    
    # 1. 读取 test_a.json
    test_file = Path(__file__).resolve().parent.parent / "data" / "test_b.json"
    
    # 2. 加载文件内容
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_text = json.dumps(test_data, ensure_ascii=False, indent=2)
    
    # 3. 添加系统提示词（因为实际压缩时也会加）
    full_text = f"{SYSTEM_PROMPT}\n\n{test_text}"
    
    # 4. 调用分块函数
    chunks = _adapter._chunk_text(full_text, max_length=_adapter.safe_chunk_size)
    
    # 5. 输出分块信息
    print(f"原始文本长度: {len(full_text)} 字符")
    print(f"安全阈值: {_adapter.safe_chunk_size} 字符")
    print(f"分块数量: {len(chunks)} 块\n")
    
    for i, chunk in enumerate(chunks):
        print(f"{'='*80}")
        print(f"第 {i+1} 块 - 长度: {len(chunk)} 字符")
        print(f"{'='*80}")
        print(chunk)  # 只显示前500字符
        # if len(chunk) > 500:
        #     print(f"... (省略 {len(chunk)-500} 字符)")
        print()