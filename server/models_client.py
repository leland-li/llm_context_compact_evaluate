import requests
import json
import os
import sys
import time
from typing import Optional
from pathlib import Path

# 添加父目录到 Python 路径，以便导入 config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class OllamaClient:
    """通用 Ollama 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = self.list_models()
    
    def infer(self, model: str, text: str, **kwargs) -> tuple:
        """
        流式推理接口（使用 Ollama Modelfile 中定义的参数和 system prompt）
        
        Args:
            model: 模型名称（Ollama 中定义的模型）
            text: 输入文本
            **kwargs: 其他参数（覆盖 Modelfile 参数，可选）
            
        Returns:
            tuple: (完整响应文本, 首token时间, 总响应时间, token计数)
        """
        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    'model': model,
                    'prompt': text,
                    'stream': True,
                    **kwargs
                },
                timeout=300,
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        response_text += chunk.get('response', '')
                        token_count += 1
                        
                        # 记录首 token 时间
                        if first_token_time is None and response_text.strip():
                            first_token_time = time.time() - start_time
                    except json.JSONDecodeError:
                        continue
            
            total_time = time.time() - start_time
            if first_token_time is None:
                first_token_time = total_time
            
            return response_text, first_token_time, total_time, token_count
        
        except Exception as e:
            raise Exception(f"流式推理失败: {str(e)}")
    
    def list_models(self) -> list:
        """列出所有模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return [m['name'] for m in response.json().get('models', [])]
        except:
            print("Warning: Could not fetch models from Ollama")
            return []

    def warmup(self, model: str, warmup_text: str = "This is a test prompt, don't worry about it. you can say hello if you want.", retries: int = 1) -> None:
        """
        模型预热，加载到显存
        
        Args:
            model: 模型名称
            warmup_text: 用于预热的短文本
            retries: 预热次数
        """
        for i in range(retries):
            try:
                # 直接调用 infer 但不记录结果
                response_text, ttft, total_time, token_count = self.infer(model, warmup_text)
                if i == 0:
                    print(f"   🔥 Warmup 完成 (首次加载耗时: {total_time:.2f}s)")
                else:
                    print(f"   🔥 Warmup#{i+1} 完成 (耗时: {total_time:.2f}s)")
            except Exception as e:
                print(f"   ⚠️  Warmup 失败: {str(e)}")
                continue

def load_test_data(filepath: str) -> str:
    """加载测试数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 将 JSON 转换为易读的文本格式
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return text


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
    result_base_name = f"{test_file_name}_compress"
    
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
    
    # 2. 为每个模型保存单独的压缩结果
    for model_key, result in results.items():
        if 'error' not in result:
            model_file = results_path / f"{result_base_name}_{model_key}.txt"
            with open(model_file, 'w', encoding='utf-8') as f:
                f.write(f"模型: {model_key}\n")
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


def format_result(model: str, result: str, max_length: int = 500) -> str:
    """格式化输出结果"""
    output = f"\n{'='*80}\n"
    output += f"模型: {model}\n"
    output += f"{'='*80}\n"
    output += f"{result[:max_length]}"
    if len(result) > max_length:
        output += f"\n... (省略 {len(result) - max_length} 字符)"
    output += f"\n字符数: {len(result)}\n"
    return output


# 使用示例
if __name__ == "__main__":
    print("="*80)
    print("Ollama 文本压缩模型对比测试")
    print("="*80)
    
    # 初始化客户端
    client = OllamaClient()
    
    # 检查 Ollama 连接
    if not client.available_models:
        print("ERROR: 无法连接到 Ollama 服务！")
        print("请确保 Ollama 已启动: ollama serve")
        sys.exit(1)
    
    print(f"\n✓ 检测到 {len(client.available_models)} 个已加载模型:")
    for model in client.available_models:
        print(f"  - {model}")
    
    # 加载测试数据
    test_file = Path(__file__).parent.parent / "data" / "test_d.json"
    print(f"\n📂 加载测试数据: {test_file}")
    
    if not test_file.exists():
        print(f"ERROR: 测试文件不存在 {test_file}")
        sys.exit(1)
    
    test_text = load_test_data(str(test_file))
    print(f"✓ 测试文本长度: {len(test_text)} 字符")
    
    # 要测试的压缩模型（已在 Ollama Modelfile 中定义了所有参数和 system prompt）
    models_to_test = ["qwen3.5-4b-compress", "gemma4-e4b-compress", "gemma4-e2b-compress", "glm-edge-4b-compress", "phi4-mini-compress"]
    
    # 获取可用模型并移除 :latest 标签
    available_models_normalized = [m.split(':')[0] for m in client.available_models]
    
    # 过滤出实际可用的模型
    available = [m for m in models_to_test if m in available_models_normalized]
    
    # 创建映射：规范名称 -> 实际名称（可能带标签）
    model_name_map = {}
    for actual_name in client.available_models:
        normalized = actual_name.split(':')[0]
        if normalized in models_to_test:
            model_name_map[normalized] = actual_name
    
    if not available:
        print(f"ERROR: 没有可用的模型！")
        print(f"期望模型: {models_to_test}")
        sys.exit(1)
    
    print(f"\n🧪 开始测试 {len(available)} 个压缩优化模型...\n")
    
    results = {}
    
    for i, model_key in enumerate(available, 1):
        # 获取实际的模型名称（可能带:latest标签）
        actual_model_name = model_name_map[model_key]
        
        print(f"[{i}/{len(available)}] 测试模型: {model_key} ({actual_model_name})...")
        try:
            # 获取该模型的请求级补丁参数（如 qwen3.5 的 think=False）
            request_patch = config.MODEL_REQUEST_PATCHES.get(model_key, {})

            # 1. 执行预热（冷启动）
            print("   预热中...")
            warmup_response, warmup_ttft, warmup_time, warmup_tokens = client.infer(actual_model_name, "This is a test.", **request_patch)
            
            # 2. 执行实际推理（热启动）
            print("   推理中...")
            response_text, ttft, total_time, token_count = client.infer(actual_model_name, test_text, **request_patch)
            
            # 计算吞吐量 (token/s)
            throughput = token_count / total_time if total_time > 0 else 0
            
            results[model_key] = {
                'output': response_text,
                'coldstart_time': warmup_time,  # 冷启时间
                'ttft': ttft,  # 首 token 响应时间（热启）
                'total_time': total_time,  # 总响应时间（热启）
                'token_count': token_count,  # token 数量
                'throughput': throughput,  # 吞吐量 (token/s)
                'input_length': len(test_text),
                'output_length': len(response_text),
                'compression_ratio': len(test_text) / len(response_text) if len(response_text) > 0 else 0
            }
            
            print(f"  ✓ 完成 (冷启: {warmup_time:.2f}s, TTFT: {ttft:.3f}s, 总耗时: {total_time:.2f}s, 吞吐量: {throughput:.2f} token/s)")
        
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            results[model_key] = {'error': str(e)}
    
    # 保存结果到文件
    test_file_name = test_file.stem  # 获取文件名（不含扩展名）
    save_results(results, test_file_name)
    
    # 输出结果
    print("\n" + "="*80)
    print("推理结果对比")
    print("="*80)
    
    for model in available:
        if 'error' in results[model]:
            print(f"\n❌ {model}: {results[model]['error']}")
        else:
            print(format_result(model, results[model]['output']))
    
    # 输出统计信息
    print("\n" + "="*80)
    print("性能统计 (冷启 vs 热启流式推理)")
    print("="*80)
    
    stats_table = f"\n{'模型名':<20} {'冷启(s)':<12} {'TTFT(ms)':<12} {'总耗时(s)':<12} {'吞吐量':<15} {'Token数':<10}\n"
    stats_table += "-" * 81 + "\n"
    
    for model in available:
        if 'error' not in results[model]:
            r = results[model]
            ttft_ms = r['ttft'] * 1000
            stats_table += f"{model:<20} {r['coldstart_time']:<12.2f} {ttft_ms:<12.2f} {r['total_time']:<12.2f} {r['throughput']:<15.2f} {r['token_count']:<10}\n"
    
    print(stats_table)
    
    # 添加压缩比统计
    print("\n" + "="*80)
    print("压缩效果统计")
    print("="*80)
    
    compress_table = f"\n{'模型名':<20} {'输入字符':<12} {'输出字符':<12} {'压缩比':<12}\n"
    compress_table += "-" * 56 + "\n"
    
    for model in available:
        if 'error' not in results[model]:
            r = results[model]
            compress_table += f"{model:<20} {r['input_length']:<12} {r['output_length']:<12} {r['compression_ratio']:<12.2f}x\n"
    
    print(compress_table)
    
    # 输出最佳模型
    print("\n📊 对比分析:")
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        fastest_coldstart = min(valid_results.items(), key=lambda x: x[1]['coldstart_time'])
        fastest_ttft = min(valid_results.items(), key=lambda x: x[1]['ttft'])
        highest_throughput = max(valid_results.items(), key=lambda x: x[1]['throughput'])
        fastest_total = min(valid_results.items(), key=lambda x: x[1]['total_time'])
        best_compression = max(valid_results.items(), key=lambda x: x[1]['compression_ratio'])
        shortest_output = min(valid_results.items(), key=lambda x: x[1]['output_length'])
        
        print(f"  ❄️  最快冷启: {fastest_coldstart[0]} ({fastest_coldstart[1]['coldstart_time']:.2f}s)")
        print(f"  ⚡ 最快首Token (TTFT): {fastest_ttft[0]} ({fastest_ttft[1]['ttft']*1000:.2f}ms)")
        print(f"  📈 最高吞吐量: {highest_throughput[0]} ({highest_throughput[1]['throughput']:.2f} token/s)")
        print(f"  🏃 最快总耗时: {fastest_total[0]} ({fastest_total[1]['total_time']:.2f}s)")
        print(f"  📉 最高压缩比: {best_compression[0]} ({best_compression[1]['compression_ratio']:.2f}x)")
        print(f"  📝 最短输出: {shortest_output[0]} ({shortest_output[1]['output_length']} 字符)")