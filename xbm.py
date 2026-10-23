import torch

class XBM:
    def __init__(self, size, emb_dim):
        self.K = size  # memory bank 容量
        self.emb_dim = emb_dim

        self.feats = torch.zeros(size, emb_dim).cuda()
        self.targets = torch.zeros(size, dtype=torch.long).cuda()

        self.ptr = 0  # 指针位置
        self.is_full = False

    def enqueue_dequeue(self, feats, targets):
        """入队当前 batch 特征到 memory bank（自动覆盖旧数据）"""
        q_size = len(targets)

        if self.ptr + q_size > self.K:
            # 超出最大容量，覆盖尾部
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
            self.is_full = True
        else:
            # 正常插入
            self.feats[self.ptr:self.ptr + q_size] = feats
            self.targets[self.ptr:self.ptr + q_size] = targets
            self.ptr += q_size
            if self.ptr == self.K:
                self.is_full = True

    def get(self):
        """获取当前 memory 中所有特征（含刚入队 batch 特征）"""
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]
