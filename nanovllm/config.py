# done
import os
from dataclasses import dataclass

from transformers import AutoConfig


@dataclass
class Config:
    model: str  # 模型路径或模型ID，可以是本地路径或HuggingFace模型名称
    max_num_batched_tokens: int = 32768  # 批处理中最大token数量，控制内存使用
    max_num_seqs: int = 512  # 同时处理的最大序列数量
    max_model_len: int = 4096  # 模型支持的最大序列长度
    gpu_memory_utilization: float = 0.9  # GPU内存使用率，0.9表示使用90%的GPU内存
    tensor_parallel_size: int = 1  # 张量并行大小，用于多GPU推理加速
    enforce_eager: bool = False  # 是否强制使用eager模式，禁用图编译优化
    hf_config: AutoConfig | None = (
        None  # HuggingFace模型配置，会在初始化时自动加载    eos: int = -1  # 结束符token ID，会在初始化时从tokenizer获取
    )
    kvcache_block_size: int = 256  # KV缓存块大小（单位：token数量），必须是256的倍数
    num_kvcache_blocks: int = -1  # KV缓存块数量，-1表示自动计算

    def __post_init__(self):
        """
        初始化后的配置验证和设置
        - 验证模型路径是否存在
        - 验证配置参数的有效性
        - 加载HuggingFace模型配置
        - 根据模型配置调整最大序列长度
        """
        assert os.path.isdir(self.model)  # 确保模型路径存在且是目录
        assert self.kvcache_block_size % 256 == 0  # KV缓存块大小必须是256的倍数
        assert 1 <= self.tensor_parallel_size <= 8  # 张量并行大小必须在1-8之间
        self.hf_config = AutoConfig.from_pretrained(
            self.model
        )  # 加载HuggingFace模型配置
        # 使用模型配置中的最大位置嵌入长度，取较小值避免超出模型限制
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
