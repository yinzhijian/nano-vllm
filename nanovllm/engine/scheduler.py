from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []  # 保存本轮被选中执行的序列
        num_batched_tokens = 0  # 统计本轮已安排的 token 数，用于限制批次大小

        # Prefill：优先安排等待队列中的序列处理尚未计算的 prompt token。
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]  # 只查看等待队列队首，按先进先出顺序调度
            remaining = self.max_num_batched_tokens - num_batched_tokens  # 本批次还可容纳的 token 数
            if remaining == 0:
                break  # 批次 token 预算已用完，结束 prefill 调度
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)  # 查询可复用的前缀块并检查显存块是否足够
                if num_cached_blocks == -1:
                    break  # 当前没有足够的 KV cache 块，暂时无法调度该序列
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size  # 扣除缓存前缀后仍需计算的 token 数
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens  # 已分配过块时，只计算尚未缓存的 token
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break  # 仅当本轮还没有其他序列时才允许分块 prefill
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)  # 建立块表：复用缓存块并分配所需的新块
            seq.num_scheduled_tokens = min(num_tokens, remaining)  # 本轮为该序列安排的 token 数，不超过剩余预算
            num_batched_tokens += seq.num_scheduled_tokens  # 将该序列占用的 token 数计入批次预算
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING  # prompt 已全部安排计算，将序列状态切换为运行中
                self.waiting.popleft()  # 从等待队列移除已开始运行的序列
                self.running.append(seq)  # 放入运行队列，等待后续 decode
            scheduled_seqs.append(seq)  # 将该序列加入本轮执行列表

        if scheduled_seqs:
            return scheduled_seqs, True  # 本轮安排了 prefill，返回序列列表及 prefill 标记

        # Decode：没有安排 prefill 时，为运行中的序列各安排一个生成 token。
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()  # 取出运行队列队首，准备安排一次 decode
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())  # 空间不足时先抢占运行队列末尾的序列，释放其 KV cache
                else:
                    self.preempt(seq)  # 没有其他序列可抢占时，改为抢占当前序列
                    break  # 当前序列已退回等待队列，不再安排它进行 decode
            else:
                seq.num_scheduled_tokens = 1  # decode 每轮只为每条序列安排一个新 token
                seq.is_prefill = False  # 标记该序列当前执行的是生成阶段，而非 prompt 阶段
                self.block_manager.may_append(seq)  # 如果 token 将进入新块，则为序列追加一个 KV cache 块
                scheduled_seqs.append(seq)  # 将成功安排的序列加入本轮执行列表
        assert scheduled_seqs  # 确保 decode 至少安排到一个序列，否则表示调度状态异常
        self.running.extendleft(reversed(scheduled_seqs))  # 按原调度顺序放回运行队列队首
        return scheduled_seqs, False  # 返回本轮序列列表及非 prefill 标记（即 decode）

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
