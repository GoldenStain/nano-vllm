# done
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    """
    序列调度器
    负责管理推理请求的调度、内存分配和序列状态转换
    实现了prefill和decode两阶段的调度策略
    """

    def __init__(self, config: Config):
        """
        初始化调度器
        
        Args:
            config: 配置对象，包含调度相关参数
        """
        self.max_num_seqs = config.max_num_seqs  # 同时处理的最大序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 批处理最大token数
        self.eos = config.eos  # 结束符token ID
        
        # 初始化KV缓存块管理器
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        
        # 序列队列：等待处理的序列
        self.waiting: deque[Sequence] = deque()
        # 序列队列：正在运行的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        检查是否所有序列都已完成
        
        Returns:
            bool: 如果没有等待和运行中的序列则返回True
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新序列到等待队列
        
        Args:
            seq: 要添加的序列对象
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度序列进行推理
        实现两阶段调度：prefill（预填充）和decode（解码）
        
        Returns:
            tuple: (调度的序列列表, 是否为prefill阶段)
        """
        # Phase 1: Prefill阶段 - 处理新的等待序列
        scheduled_seqs = []  # 本次调度的序列列表
        num_seqs = 0  # 已调度的序列数量
        num_batched_tokens = 0  # 已调度的token总数
        
        # 尝试调度等待队列中的序列进行prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 获取队首序列
            
            # 检查是否超过token限制或无法分配内存块
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 无法调度更多序列，退出循环
            
            # 调度该序列
            num_seqs += 1
            self.block_manager.allocate(seq)  # 为序列分配KV缓存块
            num_batched_tokens += len(seq) - seq.num_cached_tokens  # 累计token数
            seq.status = SequenceStatus.RUNNING  # 更新序列状态为运行中
            
            # 从等待队列移动到运行队列
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        # 如果有prefill序列被调度，返回prefill批次
        if scheduled_seqs:
            return scheduled_seqs, True

        # Phase 2: Decode阶段 - 处理正在运行的序列
        # 尝试调度运行队列中的序列进行decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 从运行队列取出序列
            
            # 检查是否可以为序列分配新的token空间
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 如果有其他运行序列，抢占最后一个序列
                    self.preempt(self.running.pop())
                else:
                    # 如果没有其他序列可抢占，抢占当前序列
                    self.preempt(seq)
                    break
            else:
                # 可以分配空间，调度该序列
                num_seqs += 1
                self.block_manager.may_append(seq)  # 预留空间
                scheduled_seqs.append(seq)
        
        # decode阶段必须有序列被调度
        assert scheduled_seqs
        
        # 将调度的序列重新放回运行队列的前部（保持顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占序列，释放其资源并移回等待队列
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING  # 状态改为等待
        self.block_manager.deallocate(seq)  # 释放KV缓存块
        self.waiting.appendleft(seq)  # 放回等待队列前部（优先处理）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        处理推理结果，更新序列状态
        
        Args:
            seqs: 推理的序列列表
            token_ids: 生成的token ID列表
            
        Returns:
            list[bool]: 处理结果列表（当前实现未返回值）
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  # 将生成的token添加到序列
            
            # 检查序列是否应该结束
            if (
                not seq.ignore_eos and token_id == self.eos  # 遇到结束符
            ) or seq.num_completion_tokens == seq.max_tokens:  # 达到最大长度
                seq.status = SequenceStatus.FINISHED  # 标记为完成
                self.block_manager.deallocate(seq)  # 释放KV缓存块
                self.running.remove(seq)  # 从运行队列移除
