
import asyncio
import aiohttp
import time
import json
import random
import pandas as pd
import numpy as np
import signal
from dataclasses import dataclass
from typing import List, Dict
from transformers import AutoTokenizer

# 全局优雅退出标志，所有场景共享
GLOBAL_CANCELLED = False


# --- 配置区域 ---
@dataclass
class BenchmarkConfig:
    # API 配置
    api_url: str = "http://71.77.67.241:1995/v1/completions"
    model_name: str = "glm"

    # Excel 输入文件配置
    input_excel_file: str = "requests.xlsx"  # Excel 文件名
    input_prompt_column: str = "prompt"  # 读取 prompt 的列名
    use_excel_input: bool = False  # 是否从 Excel 读取请求（True=从Excel读取，False=自动生成）

    # 分词器路径 (可以是本地路径，也可以是 HuggingFace ID)
    tokenizer_path: str = "Qwen/Qwen2.5-32B-Instruct"

    # 核心测试参数
    concurrency: int = 4  # 并发数
    total_requests: int = 4  # 总请求数

    # Token 数量配置 (基准值，仅在 use_excel_input=False 时生效)
    input_tokens_target: int = 8000  # 期望输入的 Token 数
    output_tokens_target: int = 1000  # 期望输出的 Token 数 (即 max_tokens)

    # 浮动范围 (0.1 代表 +/- 10%，仅在 use_excel_input=False 时生效)
    input_jitter_ratio: float = 0.1
    output_jitter_ratio: float = 0.1

    temperature: float = 0.0
    timeout: int = 600

    # 直接指定 min/max token 范围 (优先于 target/jitter，仅 use_excel_input=False 时生效)
    # 对齐 evalscope 的 --min/max-prompt-length 和 --min/max-tokens
    input_min_tokens: int = None
    input_max_tokens: int = None
    output_min_tokens: int = None
    output_max_tokens: int = None

    # 是否在请求中携带 ignore_eos（强制生成到 max_tokens），对齐 evalscope --extra-args
    ignore_eos: bool = True

    # 随机种子，对齐 evalscope --seed
    seed: int = 42

    # 结果输出目录，对齐 evalscope --outputs-dir
    output_dir: str = "."


# --- 辅助类：Token 生成器 ---
class TokenController:
    def __init__(self, model_path):
        print(f"⏳ 正在加载分词器: {model_path} ...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("✅ 分词器加载完成")
        except Exception as e:
            print(f"❌ 分词器加载失败: {e}")
            print("⚠️ 请确保已安装 transformers 且网络通畅，或指定正确的本地路径")
            exit(1)

        # 准备一个足够长的基础文本用于切片 (重复无意义文本)
        # 使用中文和特殊符号混合，确保分词器能正常工作
        base_str = "测试人工智能性能Benchmark " * 100
        self.base_tokens = self.tokenizer.encode(base_str)

        # 如果基础文本不够长，自动倍增
        while len(self.base_tokens) < 10000:
            self.base_tokens *= 2

    def count_tokens(self, text: str) -> int:
        """统计文本的 Token 数量"""
        return len(self.tokenizer.encode(text))

    def generate_prompt(self, target_count: int) -> str:
        """生成指定 Token 数量的无语义文本"""
        # 截取指定数量的 token id
        # 注意：首尾可能会有特殊 token，这里直接截取中间部分模拟纯文本
        if target_count > len(self.base_tokens):
            # 如果还不够，临时扩展
            factor = (target_count // len(self.base_tokens)) + 2
            current_tokens = self.base_tokens * factor
        else:
            current_tokens = self.base_tokens

        selected_ids = current_tokens[:target_count]
        selected_ids[random.randint(0, len(selected_ids) - 1)] = random.randint(0, 10000)
        # 解码回字符串
        return self.tokenizer.decode(selected_ids, skip_special_tokens=True)


# --- 核心逻辑类 ---
class LLMBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.token_controller = TokenController(config.tokenizer_path)
        self.results = []
        self.headers = {"Content-Type": "application/json"}
        # 全局记录所有的 decode latency 用于最后的大盘统计
        self.global_decode_latencies = []
        # Ctrl+C 优雅退出标志
        self.cancelled = False

        # 如果从 Excel 读取请求，加载 Excel 数据
        if config.use_excel_input:
            try:
                df = pd.read_excel(config.input_excel_file)
                if config.input_prompt_column not in df.columns:
                    print(f"❌ Excel 文件中找不到列 '{config.input_prompt_column}'")
                    print(f"可用列: {list(df.columns)}")
                    exit(1)
                self.prompts = df[config.input_prompt_column].tolist()
                print(f"✅ 已从 {config.input_excel_file} 读取 {len(self.prompts)} 条请求")
            except FileNotFoundError:
                print(f"❌ 找不到文件: {config.input_excel_file}")
                exit(1)
            except Exception as e:
                print(f"❌ 读取 Excel 文件失败: {e}")
                exit(1)
        else:
            self.prompts = None

    def cancel(self):
        self.cancelled = True

    def _get_jittered_value(self, base: int, ratio: float) -> int:
        """计算带有随机抖动的值"""
        if ratio <= 0:
            return base
        delta = int(base * ratio)
        return random.randint(base - delta, base + delta)

    async def _send_request(self, session: aiohttp.ClientSession, req_id: int):
        # 1. 获取 prompt 文本
        if self.config.use_excel_input:
            # 从 Excel 中按顺序循环使用
            prompt_text = self.prompts[req_id % len(self.prompts)]
            actual_input_tokens = self.token_controller.count_tokens(prompt_text)
            current_input_tokens = actual_input_tokens  # 记录实际 Token 数
            current_max_tokens = self._get_jittered_value(
                self.config.output_tokens_target,
                self.config.output_jitter_ratio
            )
            ignore_eos = False
        else:
            # 自动生成随机长度的 prompt
            # 若指定了 min/max 范围（对齐 evalscope），则在区间内均匀采样，否则用 target/jitter
            if self.config.input_min_tokens is not None and self.config.input_max_tokens is not None:
                current_input_tokens = random.randint(
                    self.config.input_min_tokens, self.config.input_max_tokens
                )
            else:
                current_input_tokens = self._get_jittered_value(
                    self.config.input_tokens_target,
                    self.config.input_jitter_ratio
                )
            if self.config.output_min_tokens is not None and self.config.output_max_tokens is not None:
                current_max_tokens = random.randint(
                    self.config.output_min_tokens, self.config.output_max_tokens
                )
            else:
                current_max_tokens = self._get_jittered_value(
                    self.config.output_tokens_target,
                    self.config.output_jitter_ratio
                )
            prompt_text = self.token_controller.generate_prompt(current_input_tokens)
            actual_input_tokens = current_input_tokens
            ignore_eos = self.config.ignore_eos

        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": current_max_tokens,
            "temperature": self.config.temperature,
            "ignore_eos": ignore_eos,
            "stream":True
        }

        # 计时变量
        start_time = time.perf_counter()
        ttft = 0.0
        decode_latencies = []  # 存储每个 token 的生成耗时
        token_count = 0
        status = "Failed"
        error_msg = ""

        last_token_time = 0.0

        try:
            async with session.post(self.config.api_url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    status = f"HTTP {response.status}"
                    error_msg = await response.text()
                else:
                    async for line in response.content:
                        line = line.strip()
                        if not line or line == b"data: [DONE]":
                            continue

                        if line.startswith(b"data: "):
                            current_time = time.perf_counter()

                            # 1. 解析 JSON 内容
                            try:
                                json_str = line.decode('utf-8').replace("data: ", "")
                                data = json.loads(json_str)
                                # 兼容 /v1/completions (text) 和 /v1/chat/completions (delta content)
                                chunk_text = data['choices'][0].get('delta', {}).get('content', '')
                                if not chunk_text:
                                    chunk_text = data['choices'][0].get('text', '')
                            except:
                                continue  # 解析失败跳过

                            if not chunk_text:
                                continue  # 有时候会有空包（例如只含 usage 信息）

                            chunk_token_ids = self.token_controller.tokenizer.encode(chunk_text)
                            n_new_tokens = len(chunk_token_ids)

                            if token_count == 0:
                                # 第一个 token：计算 TTFT
                                ttft = current_time - start_time
                                last_token_time = current_time
                            else:
                                step_latency = current_time - last_token_time
                                # 如果这一包里有 3 个 token，这 3 个 token 共花费了 step_latency
                                # 记录时，可以记 3 次平均值，或者记一次总值，取决于你的统计口径
                                # 这里我们将整个包的耗时平均分摊给这 N 个 token
                                avg_step_latency = step_latency / n_new_tokens
                                decode_latencies.extend([avg_step_latency] * n_new_tokens)
                                last_token_time = current_time

                            token_count += n_new_tokens

                    status = "Success"

        except Exception as e:
            status = "Error"
            error_msg = str(e)

        total_time = time.perf_counter() - start_time

        # 计算单次请求的统计
        req_avg_decode = np.mean(decode_latencies) if decode_latencies else 0
        req_p90_decode = np.percentile(decode_latencies, 90) if decode_latencies else 0

        # 收集到全局
        if decode_latencies:
            self.global_decode_latencies.extend(decode_latencies)

        result = {
            "Request_ID": req_id,
            "Status": status,
            "Input_Tokens": actual_input_tokens,  # 实际输入 Token 数
            "Target_Max_Tokens": current_max_tokens,
            "Actual_Output_Tokens": token_count,
            "Total_Time_s": round(total_time, 4),
            "TTFT_s": round(ttft, 4),
            "Avg_Decode_s": round(req_avg_decode, 5),
            "P90_Decode_s": round(req_p90_decode, 5),
            "Decode_Detail": decode_latencies,
            "TPS": round(token_count / total_time, 2) if total_time > 0 else 0,
            "Error": error_msg,
            "Prompt_Snippet": prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
        }

        # 打印简略日志
        if status == "Success":
            print(
                f"[Req {req_id:02d}] TTFT: {result['TTFT_s']}s | InTokens: {actual_input_tokens} | OutTokens: {token_count} | AvgDecode: {result['Avg_Decode_s'] * 1000:.1f}ms | P90: {result['P90_Decode_s'] * 1000:.1f}ms")
        else:
            print(f"[Req {req_id:02d}] ❌ {status}")

        return result

    async def run(self):
        if self.config.use_excel_input:
            print(f"🚀 开始压测: 并发={self.config.concurrency}, 总请求={self.config.total_requests}")
            print(f"📄 从 Excel 读取请求: {self.config.input_excel_file} (列: {self.config.input_prompt_column})")
            print(f"📝 已加载 {len(self.prompts)} 条 prompt，将循环使用")
        else:
            print(f"🚀 开始压测: 并发={self.config.concurrency}, 总请求={self.config.total_requests}")
            print(f"📝 输入Token目标: {self.config.input_tokens_target} (±{self.config.input_jitter_ratio * 100}%)")
            print(f"📝 输出Token目标: {self.config.output_tokens_target} (±{self.config.output_jitter_ratio * 100}%)")

        semaphore = asyncio.Semaphore(self.config.concurrency)
        pending = set()

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async def worker(req_id):
                async with semaphore:
                    return await self._send_request(session, req_id)

            next_req_id = 1
            # 初始下发 concurrency 个任务
            for i in range(self.config.concurrency):
                if next_req_id <= self.config.total_requests:
                    t = asyncio.create_task(worker(next_req_id))
                    pending.add(t)
                    next_req_id += 1

            while pending:
                if self.cancelled:
                    # 取消所有 pending tasks
                    for t in pending:
                        t.cancel()
                    # 等待被取消的任务完成（会抛 CancelledError，忽略）
                    await asyncio.gather(*pending, return_exceptions=True)
                    break
                # 等待任意一个完成
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                    except asyncio.CancelledError:
                        result = {"Request_ID": 0, "Status": "Cancelled", "Error": "Cancelled"}
                    self.results.append(result)
                # 补发新任务，直到发完
                if not self.cancelled:
                    while len(pending) < self.config.concurrency and next_req_id <= self.config.total_requests:
                        t = asyncio.create_task(worker(next_req_id))
                        pending.add(t)
                        next_req_id += 1
                # 如果没有 pending 了且还有任务没发完，补发
                if not pending and next_req_id <= self.config.total_requests:
                    t = asyncio.create_task(worker(next_req_id))
                    pending.add(t)
                    next_req_id += 1

    def print_stats_and_save(self, filename="result.xlsx", summary_filename="summary.xlsx"):
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        success_df = df[df["Status"] == "Success"]

        print("\n" + "=" * 50)
        print("📊 全局性能统计报告")
        print("=" * 50)

        if success_df.empty:
            print("❌ 所有请求均失败，无法生成统计。")
            df.to_excel(filename, index=False)
            return

        # 1. 基础统计
        total_reqs = len(df)
        success_rate = (len(success_df) / total_reqs) * 100
        ttft_values = success_df["TTFT_s"].values
        avg_ttft = float(np.mean(ttft_values))
        p50_ttft = float(np.percentile(ttft_values, 50))
        p90_ttft = float(np.percentile(ttft_values, 90))
        p99_ttft = float(np.percentile(ttft_values, 99))
        avg_tps = success_df["TPS"].mean()
        avg_input_tokens = success_df["Input_Tokens"].mean()

        # 2. 端到端回答时长统计
        total_times = success_df["Total_Time_s"].values
        e2e_avg = float(np.mean(total_times))
        e2e_p90 = float(np.percentile(total_times, 90))
        e2e_p99 = float(np.percentile(total_times, 99))

        print(f"请求总数: {total_reqs}")
        print(f"成功率  : {success_rate:.2f}%")
        print(f"平均 TPS: {avg_tps:.2f}")
        print(f"平均 TTFT (首字): {avg_ttft:.4f} s")
        print(f"中位 TTFT (P50): {p50_ttft:.4f} s")
        print(f"TTFT P90: {p90_ttft:.4f} s")
        print(f"TTFT P99: {p99_ttft:.4f} s")
        print(f"平均输入 Token 数: {avg_input_tokens:.0f}")
        print(f"端到端回答时长 (Avg): {e2e_avg:.4f} s")
        print(f"端到端回答时长 (P90): {e2e_p90:.4f} s")
        print(f"端到端回答时长 (P99): {e2e_p99:.4f} s")

        # 3. Decode Latency 详细分布 (基于所有生成的 Token)
        latencies_ms = None
        if self.global_decode_latencies:
            latencies_ms = np.array(self.global_decode_latencies) * 1000
            print("-" * 30)
            print("⚡ Decode Latency 分布 (每 Token 生成耗时)")
            print("-" * 30)
            print(f"样本总数 (Tokens): {len(latencies_ms)}")
            print(f"Avg (平均): {np.mean(latencies_ms):.2f} ms")
            print(f"Min (最小): {np.min(latencies_ms):.2f} ms")
            print(f"Max (最大): {np.max(latencies_ms):.2f} ms")
            print(f"P50 (中位): {np.percentile(latencies_ms, 50):.2f} ms")
            print(f"P90 (90%):  {np.percentile(latencies_ms, 90):.2f} ms")
            print(f"P95 (95%):  {np.percentile(latencies_ms, 95):.2f} ms")
            print(f"P99 (99%):  {np.percentile(latencies_ms, 99):.2f} ms")

        # 保存详细数据到 Excel
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name="Raw Data", index=False)

        # 保存独立 Summary 文件
        summary_rows = [
            ("请求总数", total_reqs),
            ("成功数", len(success_df)),
            ("失败数", total_reqs - len(success_df)),
            ("成功率 (%)", round(success_rate, 2)),
            ("平均 TPS", round(avg_tps, 2)),
            ("平均 TTFT (s)", round(avg_ttft, 4)),
            ("中位 TTFT P50 (s)", round(p50_ttft, 4)),
            ("TTFT P90 (s)", round(p90_ttft, 4)),
            ("TTFT P99 (s)", round(p99_ttft, 4)),
            ("平均输入 Token 数", round(avg_input_tokens, 0)),
            ("端到端回答时长 - Avg (s)", round(e2e_avg, 4)),
            ("端到端回答时长 - P90 (s)", round(e2e_p90, 4)),
            ("端到端回答时长 - P99 (s)", round(e2e_p99, 4)),
        ]
        if latencies_ms is not None:
            summary_rows += [
                ("Decode Latency - Avg (ms)", round(float(np.mean(latencies_ms)), 2)),
                ("Decode Latency - P50 (ms)", round(float(np.percentile(latencies_ms, 50)), 2)),
                ("Decode Latency - P90 (ms)", round(float(np.percentile(latencies_ms, 90)), 2)),
                ("Decode Latency - P95 (ms)", round(float(np.percentile(latencies_ms, 95)), 2)),
                ("Decode Latency - P99 (ms)", round(float(np.percentile(latencies_ms, 99)), 2)),
            ]
        summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
        summary_df.to_excel(summary_filename, index=False)

        print("=" * 50)
        print(f"📂 详细结果已写入: {filename}")
        print(f"📂 统计摘要已写入: {summary_filename}")


def build_parser():
    import argparse

    # 参数命名对齐 evalscope (run_evalscope.py)，从而 env.sh 中可以像调用 evalscope 一样调用本脚本
    parser = argparse.ArgumentParser(
        description="MAE 性能压测 (CLI 对齐 evalscope)"
    )
    parser.add_argument("--number", type=int, default=8,
                        help="总请求数 (evalscope --number)")
    parser.add_argument("--parallel", type=int, default=8,
                        help="并发数 (evalscope --parallel)")
    parser.add_argument("--model", type=str, default="ds",
                        help="模型名 (evalscope --model)")
    parser.add_argument("--url", type=str, default="http://localhost:1995/v1/chat/completions",
                        help="API 地址 (evalscope --url)")
    parser.add_argument("--tokenizer-path", dest="tokenizer_path", type=str,
                        default="Qwen/Qwen2.5-32B-Instruct",
                        help="分词器路径 (evalscope --tokenizer-path)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (evalscope --seed)")
    parser.add_argument("--min-prompt-length", dest="min_prompt_length", type=int, default=2048,
                        help="最小输入 token 数 (evalscope --min-prompt-length)")
    parser.add_argument("--max-prompt-length", dest="max_prompt_length", type=int, default=2048,
                        help="最大输入 token 数 (evalscope --max-prompt-length)")
    parser.add_argument("--min-tokens", dest="min_tokens", type=int, default=128,
                        help="最小输出 token 数 (evalscope --min-tokens)")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=128,
                        help="最大输出 token 数 (evalscope --max-tokens)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--outputs-dir", dest="outputs_dir", type=str, default="logs",
                        help="结果输出目录 (evalscope --outputs-dir)")
    parser.add_argument("--extra-args", dest="extra_args", type=str, default='{"ignore_eos": true}',
                        help='额外请求参数 JSON，支持 ignore_eos (evalscope --extra-args)')

    # 从 Excel 读取请求（evalscope 无对应项，保留脚本自有能力）
    parser.add_argument("--input-excel-file", dest="input_excel_file", type=str, default="requests.xlsx")
    parser.add_argument("--input-prompt-column", dest="input_prompt_column", type=str, default="prompt")
    parser.add_argument("--use-excel-input", dest="use_excel_input", action="store_true")

    # 以下参数仅为兼容 evalscope 的调用形式而接受，脚本本身不使用
    parser.add_argument("--api", type=str, default="openai", help="(兼容 evalscope，忽略)")
    parser.add_argument("--dataset", type=str, default="random", help="(兼容 evalscope，忽略)")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None,
                        help="(兼容 evalscope，忽略)")
    parser.add_argument("--prefix-length", dest="prefix_length", type=int, default=0,
                        help="(兼容 evalscope，忽略)")
    parser.add_argument("--rate", type=float, default=-1, help="(兼容 evalscope，忽略)")

    return parser


def config_from_args(args):
    ignore_eos = True
    if args.extra_args:
        try:
            extra = json.loads(args.extra_args)
            ignore_eos = bool(extra.get("ignore_eos", True))
        except Exception:
            print(f"⚠️  无法解析 --extra-args: {args.extra_args}，将使用默认 ignore_eos=True")

    return BenchmarkConfig(
        api_url=args.url,
        model_name=args.model,
        tokenizer_path=args.tokenizer_path,
        use_excel_input=args.use_excel_input,
        input_excel_file=args.input_excel_file,
        input_prompt_column=args.input_prompt_column,
        concurrency=args.parallel,
        total_requests=args.number,
        input_min_tokens=args.min_prompt_length,
        input_max_tokens=args.max_prompt_length,
        output_min_tokens=args.min_tokens,
        output_max_tokens=args.max_tokens,
        # target/jitter 作为 min==max 时的等价基准
        input_tokens_target=(args.min_prompt_length + args.max_prompt_length) // 2,
        output_tokens_target=(args.min_tokens + args.max_tokens) // 2,
        temperature=args.temperature,
        timeout=args.timeout,
        ignore_eos=ignore_eos,
        seed=args.seed,
        output_dir=args.outputs_dir,
    )


if __name__ == "__main__":
    import signal, os

    args = build_parser().parse_args()

    # 固定随机种子，保证可复现（对齐 evalscope --seed）
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = config_from_args(args)

    print(f"\n{'='*50}")
    print(f"🚀 开始 MAE 性能压测")
    print(f"📌 模型      : {config.model_name}")
    print(f"📌 API       : {config.api_url}")
    print(f"📌 并发数    : {config.concurrency}")
    print(f"📌 总请求数  : {config.total_requests}")
    if not config.use_excel_input:
        print(f"📌 输入Token : [{config.input_min_tokens}, {config.input_max_tokens}]")
        print(f"📌 输出Token : [{config.output_min_tokens}, {config.output_max_tokens}]")
        print(f"📌 ignore_eos: {config.ignore_eos}")
    print(f"{'='*50}\n")

    bencher = LLMBenchmarker(config)

    def handle_sigint(signum, frame):
        global GLOBAL_CANCELLED
        print("\n⚠️  接收到 Ctrl+C，正在优雅退出...")
        GLOBAL_CANCELLED = True
        bencher.cancel()

    old_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bencher.run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n⚠️  正在取消中...")
        GLOBAL_CANCELLED = True
        bencher.cancel()
    finally:
        signal.signal(signal.SIGINT, old_handler)
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass

    # 保存结果到输出目录 (对齐 evalscope --outputs-dir)
    os.makedirs(config.output_dir, exist_ok=True)
    output_filename = os.path.join(config.output_dir, "results.xlsx")
    summary_filename = os.path.join(config.output_dir, "summary.xlsx")
    bencher.print_stats_and_save(output_filename, summary_filename)
