# Context Compression Model Test Report

## 1. Test Background

To evaluate the feasibility and effectiveness of lightweight large models for context compression tasks in a local environment, this test conducted a unified comparison across multiple GGUF-quantized models, focusing on the following questions:

- Whether each model can be deployed and run stably under an 8GB VRAM, Windows 11 environment.
- The differences among models in compression efficiency, inference performance, and information fidelity under unified prompts and generation parameters.
- Which models are better suited for practical context compression scenarios across short-to-medium texts and long-context inputs.

This report is intended to support model selection and deployment decisions, with an emphasis on reproducible test results, clear analytical conclusions, and actionable follow-up recommendations.

## 2. Executive Summary

A total of 5 locally deployable models were tested in this evaluation, all using the Ollama + GGUF setup. The assessment covered two dimensions: performance and compression quality. The overall findings are as follows:

- Qwen3.5-4B-Q4_K_M delivered the best overall performance. It ranked highest in semantic similarity, key information recall, and overall stability, making it the current default recommendation for context compression.
- glm-edge-4b-compress performed best in first-token latency and throughput, making it suitable for latency-sensitive scenarios. However, its key information recall was significantly weaker on long-text inputs.
- gemma-4-E2B-it-Q8_0 demonstrated a certain degree of balance and can serve as an alternative candidate, but its overall quality and speed still lag behind Qwen3.5-4B.
- gemma-4-E4B-it-Q4_K_M incurred high inference cost under the current test hardware conditions. With 8GB GPU VRAM being insufficient, part of the inference was executed on CPU. Its cold-start time and total latency were significantly worse, making it unsuitable as the primary option in the current environment.
- Phi-4-mini-instruct-Q4_K_M achieved the most aggressive compression, but suffered substantial losses in semantic fidelity and key information retention. It is more suitable for extreme compression experiments than for formal business use.

Considering deployment cost, inference speed, and compression quality together, Qwen3.5-4B-Q4_K_M is recommended as the default solution, while glm-edge-4b-compress is recommended as a low-latency alternative.

## 3. Test Scope and Objects

### 3.1 Test Objectives

This test focuses on context compression and aims to compare the capabilities of different lightweight models in the following aspects:

- Whether the compressed text preserves key information from the original context.
- Whether the compressed output maintains good semantic consistency and linguistic fluency.
- The model's cold-start speed, first-token response time, and overall generation efficiency on local hardware.

### 3.2 Tested Models

| No. | Model Name | Quantization Format | File Size |
| --- | --- | --- | --- |
| 1 | gemma-4-E2B-it-Q8_0 | Q8_0 | 4.70GB |
| 2 | gemma-4-E4B-it-Q4_K_M | Q4_K_M | 4.63GB |
| 3 | ggml-model-Q4_K_M (GLM-Edge-4B) | Q4_K_M | 2.44GB |
| 4 | Phi-4-mini-instruct-Q4_K_M | Q4_K_M | 2.32GB |
| 5 | Qwen3.5-4B-Q4_K_M | Q4_K_M | 2.55GB |

Note: For readability, the models are referred to below as Gemma-4-E2B, Gemma-4-E4B, GLM-Edge-4B, Phi-4-mini, and Qwen3.5-4B.

## 4. Test Environment and Constraints

### 4.1 Hardware and System Environment

| Item | Specification |
| --- | --- |
| GPU | NVIDIA RTX 5060 Laptop GPU |
| VRAM | 8.0 GB |
| GPU Power | 115W |
| Power Adapter | 100W |
| System Memory | 32GB (31.4GB available) |
| Operating System | Windows 11 Pro |

### 4.2 Deployment Setup

| Item | Description |
| --- | --- |
| Final Deployment Method | Ollama local deployment |
| Model Format | GGUF |
| Parameter Configuration | Custom inference parameters |

### 4.3 Attempts with Alternative Solutions

| Deployment Solution | Status | Conclusion |
| --- | --- | --- |
| PyTorch + Transformers | Failed | Full-precision models required more performance than the test machine could provide and could not run stably |
| vLLM | Failed | Poor compatibility on Windows, unsuitable for this test machine |
| llama-cpp-python | Difficult to debug | Complex environment setup involving compatibility issues across CUDA, llama versions, and model versions |

### 4.4 Environment Constraints

- 8GB VRAM cannot support stable execution of full-precision large models, so the test objects were limited to quantized models.
- The Windows environment provides limited support for some inference frameworks that are primarily optimized for Linux, affecting deployment choices.
- CUDA, drivers, and inference framework versions must be strictly matched, increasing environment maintenance cost.
- Quantized models reduce resource consumption at the cost of some accuracy loss, so the conclusions of this test should be interpreted with that premise in mind.

## 5. Test Methodology

### 5.1 Unified Task Definition

The task in this evaluation was to compress conversation history, tool call results, and decision-making processes. The models were required to significantly shorten the output while preserving key information, decision chains, and tool execution results as much as possible.

### 5.2 Unified Prompts and Generation Parameters

None of the models were fine-tuned. All models were adapted to the text compression task solely through a unified prompt and generation parameters. The core prompt required the models to retain key numerical values, identifiers, dates, tool results, and causal logic, while removing chitchat, redundant logs, and irrelevant descriptions.

The unified generation parameters are as follows:

| Parameter | Value |
| --- | --- |
| num_ctx | 32768 (16384 for GLM; the model's native context window is 8192) |
| temperature | 0.1 |
| top_p | 0.85 |
| top_k | 40 |
| repeat_penalty | 1.15 |

### 5.3 Test Data

A total of 4 test sets were used in this evaluation:

| Test Set | Data Characteristics | Description |
| --- | --- | --- |
| test_a | Short-to-medium text, synthetic data | Constructed by a large model, mainly used to evaluate performance in mixed Chinese-English scenarios |
| test_b | Long text, real logs | Built from real logs from the QIRA backend |
| test_c | Long text, real logs | Built from real logs from the QIRA backend |
| test_d | Ultra-long text, real logs | Built from real logs from the QIRA backend |

Among them, test_a is more suitable for observing the model's fine-grained retention ability in structured information and bilingual scenarios, while test_b, test_c, and test_d are more suitable for observing long-context compression performance on real business logs.

### 5.4 Evaluation Metrics

This evaluation assessed both performance and quality.

#### Performance Metrics

| Metric | Meaning | Preferred Direction |
| --- | --- | --- |
| coldstart_time_s | Model cold-start latency | Lower is better |
| ttft_ms | Time to first token | Lower is better |
| total_time_s | End-to-end generation latency | Lower is better |
| throughput_token_s | Throughput | Higher is better |
| output_length | Length of compressed output | Must be interpreted together with quality metrics |
| compression_multiple | Original length / compressed output length | Higher indicates more aggressive compression |

#### Quality Metrics

| Metric | Meaning | Preferred Direction |
| --- | --- | --- |
| semantic_similarity | Semantic similarity between original and compressed text, evaluated with BGE-M3 | Higher is better |
| ppl | Perplexity of the compressed text, evaluated with llama-3.1-8B (GGUF) | Lower is better |
| recall_rate | Recall rate of key information in the compressed text, evaluated with Qwen3-Max in LLM-as-Judge mode | Higher is better |

Note: The original results used two different definitions for compression_ratio. To avoid ambiguity, this report adopts the following unified definitions:

- Compression multiple: original length / compressed output length.
- Compression rate: 1 - compressed output length / original length.

## 6. Test Results

### 6.1 Performance Results

#### test_a Performance Results

| Model | Cold Start (s) | TTFT (ms) | Total Time (s) | Throughput (token/s) | Output Length | Compression Multiple |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 2.87 | 2861.63 | 8.38 | 42.96 | 1031 | 2.70 |
| Gemma-4-E4B | 90.90 | 189521.13 | 282.41 | 1.48 | 1306 | 2.13 |
| Gemma-4-E2B | 16.05 | 10618.67 | 14.43 | 23.15 | 1017 | 2.74 |
| GLM-Edge-4B | 6.54 | 2384.71 | 5.27 | 50.31 | 918 | 3.03 |
| Phi-4-mini | 8.10 | 2508.61 | 5.70 | 18.76 | 302 | 9.23 |

#### test_b Performance Results

| Model | Cold Start (s) | TTFT (ms) | Total Time (s) | Throughput (token/s) | Output Length | Compression Multiple |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 2.92 | 6153.45 | 15.67 | 37.85 | 1832 | 16.09 |
| Gemma-4-E4B | 72.57 | 109448.83 | 185.78 | 1.94 | 1103 | 26.72 |
| Gemma-4-E2B | 15.62 | 11152.15 | 13.50 | 14.00 | 626 | 47.08 |
| GLM-Edge-4B | 5.28 | 3039.06 | 3.88 | 17.29 | 214 | 137.71 |
| Phi-4-mini | 8.64 | 5131.99 | 14.04 | 11.61 | 689 | 42.77 |

#### test_c Performance Results

| Model | Cold Start (s) | TTFT (ms) | Total Time (s) | Throughput (token/s) | Output Length | Compression Multiple |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 8.69 | 4705.64 | 12.41 | 39.18 | 1489 | 13.55 |
| Gemma-4-E4B | 74.41 | 122443.71 | 214.14 | 1.94 | 1361 | 14.83 |
| Gemma-4-E2B | 15.88 | 9479.60 | 13.04 | 23.32 | 1050 | 19.22 |
| GLM-Edge-4B | 8.00 | 2997.87 | 5.76 | 40.26 | 988 | 20.42 |
| Phi-4-mini | 8.72 | 3780.19 | 7.19 | 11.54 | 431 | 46.81 |

#### test_d Performance Results

| Model | Cold Start (s) | TTFT (ms) | Total Time (s) | Throughput (token/s) | Output Length | Compression Multiple |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 8.48 | 8383.93 | 17.56 | 31.21 | 1703 | 25.90 |
| Gemma-4-E4B | 63.55 | 187346.10 | 316.87 | 1.93 | 2069 | 21.32 |
| Gemma-4-E2B | 16.08 | 14257.76 | 21.16 | 25.75 | 1838 | 24.00 |
| GLM-Edge-4B | 5.03 | 3035.98 | 7.77 | 51.75 | 1957 | 22.54 |
| Phi-4-mini | 49.70 | 6872.35 | 22.18 | 9.15 | 1119 | 39.41 |

### 6.2 Compression Quality Results

#### test_a Quality Results

| Model | Compression Rate | Semantic Similarity | PPL | Recall Rate | Key Information Retained |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 63.00% | 0.8351 | 11.5629 | 0.95 | 20 / 21 |
| Gemma-4-E2B | 63.50% | 0.7961 | 9.2080 | 0.8571 | 17 / 22 |
| Gemma-4-E4B | 53.27% | 0.7870 | 7.6483 | 0.85 | 17 / 21 |
| GLM-Edge-4B | 67.23% | 0.8204 | 12.1093 | 0.60 | 13 / 21 |
| Phi-4-mini | 89.16% | 0.7141 | 17.7784 | 0.35 | 8 / 21 |

Note: test_a is a short-to-medium synthetic sample. In this setting, Qwen3.5-4B clearly outperformed the others in semantic consistency and key information recall. Although GLM-Edge-4B was faster, it retained fewer critical fields than Qwen3.5-4B. Phi-4-mini achieved the most aggressive compression, but with obvious quality loss.

#### test_b Quality Results

| Model | Compression Rate | Semantic Similarity | PPL | Recall Rate | Key Information Retained |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 93.80% | 0.6946 | 4.9279 | 0.33 | 2 / 6 |
| Gemma-4-E2B | 97.88% | 0.6594 | 7.9245 | 0.00 | 0 / 5 |
| Gemma-4-E4B | 96.26% | 0.6893 | 5.7911 | 0.00 | 0 / 5 |
| GLM-Edge-4B | 99.27% | 0.5643 | 25.6182 | 0.00 | 0 / 5 |
| Phi-4-mini | 97.66% | 0.6038 | 17.4503 | 0.17 | 1 / 6 |

Note: test_b is a long-text real log sample. Recall rates were generally low across all models, indicating that in long-context compression scenarios, models commonly fail to preserve key information. Qwen3.5-4B was still the best in this group, but its lead was limited.

#### test_c Quality Results

| Model | Compression Rate | Semantic Similarity | PPL | Recall Rate | Key Information Retained |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 92.64% | 0.7558 | 8.4086 | 0.33 | 2 / 6 |
| Gemma-4-E2B | 94.81% | 0.7178 | 7.5349 | 0.33 | 2 / 6 |
| Gemma-4-E4B | 93.26% | 0.7450 | 8.8862 | 0.33 | 2 / 6 |
| GLM-Edge-4B | 95.12% | 0.6672 | 6.7372 | 0.00 | 0 / 6 |
| Phi-4-mini | 97.86% | 0.6444 | 116.8923 | 0.17 | 1 / 6 |

Note: The results on test_c are broadly consistent with those on test_b. Qwen3.5-4B, Gemma-4-E2B, and Gemma-4-E4B were in the same tier in terms of recall rate, but Qwen3.5-4B achieved higher semantic similarity. Phi-4-mini showed a significantly elevated PPL, indicating issues with textual naturalness.

#### test_d Quality Results

| Model | Compression Rate | Semantic Similarity | PPL | Recall Rate | Key Information Retained |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B | 96.15% | 0.7368 | 7.2387 | 0.33 | 2 / 6 |
| Gemma-4-E2B | 95.84% | 0.7650 | 7.9162 | 0.14 | 1 / 7 |
| Gemma-4-E4B | 95.32% | 0.7294 | 10.3285 | 0.17 | 1 / 6 |
| GLM-Edge-4B | 95.57% | 0.5710 | 8.9130 | 0.00 | 0 / 6 |
| Phi-4-mini | 97.46% | 0.7298 | 65.9075 | 0.00 | 0 / 5 |

Note: In the ultra-long real log scenario, all models still exhibited high compression rates, but key information recall remained low. This suggests that the current prompt and model capabilities are still biased toward high-level summarization rather than explicit preservation of critical facts.

### 6.3 Overall Observations

| Dimension | Best Model | Observation |
| --- | --- | --- |
| Overall quality | Qwen3.5-4B | Most stable overall in semantic similarity and recall rate, especially suitable for formal business scenarios |
| Inference speed | GLM-Edge-4B | Clear advantage in TTFT and throughput, suitable for speed-sensitive applications |
| Balanced trade-off | Gemma-4-E2B | Relatively balanced in compression quality and inference performance, but still slightly behind Qwen3.5-4B overall |
| High-load risk | Gemma-4-E4B | Cold start and total latency are too high, making it less cost-effective under current hardware |
| Extreme compression | Phi-4-mini | Most aggressive compression, but with substantial quality degradation, unsuitable for high-fidelity scenarios |

## 7. Analysis

### 7.1 Performance Analysis

1. GLM-Edge-4B showed the lowest TTFT and the highest throughput on most test sets, making it the fastest model in this evaluation.
2. Qwen3.5-4B achieved a good balance between speed and quality, with cold-start time and total latency both significantly better than the Gemma series.
3. Gemma-4-E4B showed clear performance disadvantages on the current test device, especially in cold-start time and total latency, making it unsuitable for frequently invoked scenarios.
4. Phi-4-mini produced the shortest outputs, indicating the most aggressive compression, but this level of compression was achieved at the cost of obvious quality loss.

### 7.2 Quality Analysis

1. On test_a, Qwen3.5-4B achieved a recall rate of 0.95, significantly higher than the other models, indicating strong retention of structured key information, numerical values, and bilingual content.
2. In the long-text real log scenarios from test_b to test_d, recall rates were generally low across all models, indicating that as context length increases, models tend to generate high-level summaries rather than precisely preserving key fields.
3. Qwen3.5-4B remained relatively ahead in long-text scenarios, suggesting that it is the safer default option.
4. Although GLM-Edge-4B has a clear speed advantage, its recall rate dropped to 0 multiple times in long-text scenarios, indicating that it is more suitable for rapid overviews than for high-fidelity compression.
5. Phi-4-mini showed significantly elevated PPL in multiple long-text scenarios, indicating instability in output naturalness and readability.

### 7.3 Business Applicability Analysis

Combining the performance and quality results, the applicable scenarios for each model can be summarized as follows:

- Qwen3.5-4B: Suitable for formal business scenarios, especially for compression tasks that require strong key information retention and semantic consistency.
- GLM-Edge-4B: Suitable for fast-summary scenarios that prioritize first-response latency and can tolerate some information loss.
- Gemma-4-E2B: Suitable as an alternative option for supplementing cross-family model validation.
- Gemma-4-E4B: Not recommended as a primary model under current hardware conditions.
- Phi-4-mini: Suitable for exploring the limits of extreme compression, but not recommended for formal production use.

## 8. Conclusions and Recommendations

### 8.1 Final Recommendations

Based on the overall results of this evaluation, the following model selection strategy is recommended:

| Role | Recommended Model | Rationale |
| --- | --- | --- |
| Default solution | Qwen3.5-4B-Q4_K_M | Best overall quality, leading in semantic similarity and key information recall, with acceptable performance |
| Low-latency alternative | GLM-Edge-4B | Best first-token latency and throughput, suitable for speed-sensitive scenarios |
| Alternative validation option | Gemma-4-E2B-it-Q8_0 | Relatively balanced in performance and quality, suitable as a cross-check model |

### 8.2 Models Not Recommended

- Gemma-4-E4B-it-Q4_K_M: Inference latency is significantly too high, making the cost-benefit ratio unfavorable under the current test environment.
- Phi-4-mini-instruct-Q4_K_M: Compression is aggressive, but quality declines significantly, making it unsuitable for business scenarios that require high-fidelity compression.

## 9. Risks and Limitations

The results of this evaluation can support model selection, but the following limitations still apply:

1. The number of test samples is limited to only 4 test sets, and the coverage can still be expanded.
2. test_a is synthetic data, while test_b to test_d are real logs. Differences in data distribution should be considered when interpreting the conclusions.
3. Key information recall was evaluated using LLM-as-Judge. Although this approach provides strong interpretability, it still involves a certain degree of model subjectivity.
4. The test environment was limited to local deployment on Windows 11 with 8GB VRAM, so the conclusions do not directly represent performance on higher-end hardware or Linux server environments.
5. The current prompt is still more oriented toward summarization and has not yet been specifically optimized for strongly constrained retention of key fields, so long-text recall still has room for improvement.

## 10. Follow-up Optimization Suggestions

The next phase of optimization is recommended to focus on the following directions:

1. Optimize the prompt specifically to add stronger retention constraints for key fields, timestamps, numerical values, and tool results.
2. Introduce more real business log samples covering different lengths, structures, and business types of contextual data.
3. Beyond plain-text summarization, explore structured compression schemes, such as segmenting output by events, decisions, tool results, and pending actions.
4. Perform manual sampling-based review of the recall evaluation results to improve the credibility of the conclusions.
5. Re-run the evaluation under higher VRAM or Linux environments to assess the impact of deployment conditions on model selection outcomes.

## Appendix A. Summary of Compression Task Prompt Principles

The compression task in this evaluation uniformly required the models to follow these principles:

- Preserve tool execution results, key IDs, numerical values, and dates.
- Preserve decision chains and causal relationships.
- Remove chitchat, system logs, and redundant descriptions.
- Output a structured summary without self-introduction or extra commentary.

## Appendix B. Explanation of Key Information Recall Evaluation

The key information recall evaluation was performed using the Qwen API in LLM-as-Judge mode, focusing on whether the following information types were retained in the compressed text:

- Numerical values
- Identifiers
- Dates and times
- Named entities
- Decision results
- Tool call results

This metric is used to assess whether the compressed output retains sufficient fidelity on business-critical facts, and it is one of the core quality metrics in this model selection evaluation.

## Appendix C. Test Metadata

| Item | Content |
| --- | --- |
| Test files | test_a, test_b, test_c, test_d |
| Enabled evaluators | semantic similarity, PPL, recall rate |
