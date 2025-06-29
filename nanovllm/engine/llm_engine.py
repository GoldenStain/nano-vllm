# done
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM推理引擎核心类
    负责管理模型加载、序列调度、多进程协调和文本生成
    """

    def __init__(self, model, **kwargs):
        """
        初始化LLM引擎
        
        Args:
            model: 模型路径或HuggingFace模型ID
            **kwargs: 配置参数，会传递给Config类
        """
        # 提取Config类支持的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        
        # 初始化多进程相关变量
        self.ps = []  # 存储子进程列表
        self.events = []  # 存储进程间同步事件
        
        # 创建多进程上下文，使用spawn方式启动子进程
        ctx = mp.get_context("spawn")
        
        # 为张量并行创建子进程（从rank 1开始，rank 0是主进程）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建进程同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 创建主进程的模型运行器（rank 0）
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 初始化tokenizer，使用fast tokenizer提高性能
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        
        # 设置结束符token ID
        config.eos = self.tokenizer.eos_token_id
        
        # 初始化序列调度器
        self.scheduler = Scheduler(config)
        
        # 注册退出时的清理函数
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源并退出所有子进程
        """
        self.model_runner.call("exit")  # 通知模型运行器退出
        del self.model_runner  # 删除模型运行器
        # 等待所有子进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加新的生成请求到调度器
        
        Args:
            prompt: 输入提示，可以是字符串或token ID列表
            sampling_params: 采样参数配置
        """
        # 如果输入是字符串，则编码为token ID列表
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # 创建序列对象并添加到调度器
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理
        
        Returns:
            tuple: (完成的输出列表, token数量)
                - outputs: [(seq_id, completion_token_ids), ...]
                - num_tokens: 正数表示prefill阶段的token数，负数表示decode阶段的序列数
        """
        # 调度器选择要处理的序列
        seqs, is_prefill = self.scheduler.schedule()
        
        # 模型运行器执行推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 调度器后处理，更新序列状态
        self.scheduler.postprocess(seqs, token_ids)
        
        # 收集已完成的序列输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 计算token数量：prefill阶段为正数，decode阶段为负数
        # prefill任务显示的是所有sequence的token数量
        # decode任务显示的是sequence的数量
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有序列都已完成生成
        
        Returns:
            bool: 如果所有序列都完成则返回True
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表，可以是字符串或token ID列表
            sampling_params: 采样参数，可以是单个参数或参数列表
            use_tqdm: 是否显示进度条
            
        Returns:
            list: 生成结果列表，每个元素包含'text'和'token_ids'字段
        """
        # 初始化进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 如果采样参数是单个对象，则复制给所有提示
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 添加所有请求到调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        outputs = {}  # 存储生成结果
        prefill_throughput = decode_throughput = 0.  # 吞吐量统计
        
        # 持续执行推理步骤直到所有序列完成
        while not self.is_finished():
            t = perf_counter()  # 记录开始时间
            output, num_tokens = self.step()  # 执行一步推理
            
            # 更新进度条显示的吞吐量信息
            if use_tqdm:
                if num_tokens > 0:
                    # prefill阶段：计算token/秒
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # decode阶段：计算序列/秒
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集完成的序列输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条
        
        # 按序列ID排序输出结果
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        # 将token ID解码为文本，返回包含文本和token ID的字典
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        
        if use_tqdm:
            pbar.close()  # 关闭进度条
        
        return outputs
