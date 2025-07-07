# done
from collections import deque

import numpy as np
import xxhash

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV缓存块
    每个块存储固定数量token的键值对缓存
    支持引用计数和内容哈希，用于缓存共享和去重
    """

    def __init__(self, block_id):
        """
        初始化块对象
        
        Args:
            block_id: 块的唯一标识符
        """
        self.block_id = block_id  # 块ID
        self.ref_count = 0  # 引用计数，表示有多少序列在使用此块
        self.hash = -1  # 块内容的哈希值，-1表示未计算或无效
        self.token_ids = []  # 存储的token ID列表

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的哈希值和token内容
        
        Args:
            hash: 新的哈希值
            token_ids: 新的token ID列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态，准备重新分配
        """
        self.ref_count = 1  # 重置引用计数为1
        self.hash = -1  # 清空哈希值
        self.token_ids = []  # 清空token列表


class BlockManager:
    """
    KV缓存块管理器
    负责管理GPU内存中的KV缓存块，实现内存分配、释放和共享
    支持基于哈希的缓存去重，提高内存利用率
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器
        
        Args:
            num_blocks: 总块数量
            block_size: 每个块的大小（token数量）
        """
        assert num_blocks > 0
        self.block_size = block_size  # 每个块的token容量
        
        # 创建所有块对象
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        
        # 哈希值到块ID的映射，用于快速查找相同内容的块
        self.hash_to_block_id: dict[int, int] = dict()
        
        # 空闲块ID队列，用于快速分配
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        
        # 已使用块ID集合，用于快速查询
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token序列的哈希值
        使用xxhash算法，支持前缀哈希用于增量计算
        
        Args:
            token_ids: 要计算哈希的token ID列表
            prefix: 前缀哈希值，用于链式哈希计算
            
        Returns:
            int: 计算出的哈希值
        """
        h = xxhash.xxh64()  # 创建64位哈希对象
        if prefix != -1:
            # 如果有前缀，先加入前缀哈希
            h.update(prefix.to_bytes(8, "little"))
        # 将token数组转换为字节并计算哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配指定的块
        
        Args:
            block_id: 要分配的块ID
            
        Returns:
            Block: 分配的块对象
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保块未被使用
        
        block.reset()  # 重置块状态
        self.free_block_ids.remove(block_id)  # 从空闲队列移除
        self.used_block_ids.add(block_id)  # 添加到使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放指定的块
        
        Args:
            block_id: 要释放的块ID
            
        Returns:
            Block: 释放的块对象
        """
        assert self.blocks[block_id].ref_count == 0  # 确保没有引用
        
        self.used_block_ids.remove(block_id)  # 从使用集合移除
        self.free_block_ids.append(block_id)  # 添加到空闲队列

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列分配足够的块
        
        Args:
            seq: 要检查的序列
            
        Returns:
            bool: 如果有足够的空闲块则返回True
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配KV缓存块
        实现智能缓存共享：相同内容的块会被多个序列共享
        
        Args:
            seq: 要分配缓存的序列
        """
        assert not seq.block_table  # 确保序列还没有分配块
        
        h = -1  # 当前块的哈希值
        cache_miss = False  # 是否发生缓存未命中
        
        # 为序列的每个块分配缓存
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取第i个块的token
            
            # 只有满块才计算哈希（用于缓存共享）
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            
            # 查找是否存在相同哈希的块
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 检查是否真正匹配（哈希冲突检查）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中
            
            if cache_miss:
                # 缓存未命中，分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中，共享现有块
                seq.num_cached_tokens += self.block_size  # 增加缓存token数
                if block_id in self.used_block_ids:
                    # 块已在使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块未在使用，重新分配
                    block = self._allocate_block(block_id)
            
            # 更新块内容和哈希映射
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            
            seq.block_table.append(block_id)  # 将块添加到序列的块表

    def deallocate(self, seq: Sequence):
        """
        释放序列的所有KV缓存块
        
        Args:
            seq: 要释放缓存的序列
        """
        # 逆序释放块（后进先出）
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数
            
            # 如果没有其他引用，释放块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        # 清空序列的缓存状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列追加新token
        
        Args:
            seq: 要检查的序列
            
        Returns:
            bool: 如果可以追加则返回True
        """
        # 只有当序列长度刚好是块大小的倍数+1时才需要新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        为序列追加token预留空间
        根据序列长度决定是否需要分配新块或更新现有块
        
        Args:
            seq: 要追加token的序列
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]  # 最后一个块
        
        if len(seq) % self.block_size == 1:
            # 序列长度刚好超出块边界，需要新块
            assert last_block.hash != -1  # 上一个块应该已经固化
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            
        elif len(seq) % self.block_size == 0:
            # 序列长度刚好填满块，更新块哈希
            assert last_block.hash == -1  # 当前块应该还未固化
            token_ids = seq.block(seq.num_blocks - 1)  # 获取最后一个块的token
            
            # 计算前缀哈希（用于链式哈希）
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            
            # 更新块内容和哈希映射
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 序列在块内部，不需要特殊处理
            assert last_block.hash == -1  # 当前块应该还未固化
