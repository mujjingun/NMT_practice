import torch
import torch.nn.functional as F
import math

torch.autograd.set_detect_anomaly(True)


# Append SOS and EOS to data
def augment(batch_seq, sos, eos, pad):
    x = F.pad(batch_seq, [1, 1, 0, 0], value=pad)
    x[:, 0] = sos
    mask = (x != pad)
    end = F.pad((x == pad)[:, 1:] != (x == pad)[:, :-1], [1, 0, 0, 0])
    x[end] = eos
    return x, mask


# make positional encoding
def make_pos_embed(max_length, embed_size):
    t = torch.arange(1, max_length + 1, dtype=torch.float32)
    omega = torch.arange(1, embed_size // 2 + 1, dtype=torch.float32) / embed_size
    wt = t.unsqueeze(1) * torch.pow(10000, -omega).unsqueeze(0)
    pos_embed = torch.zeros((1, max_length, embed_size))
    pos_embed[0, :, 0::2] = torch.sin(wt)
    pos_embed[0, :, 1::2] = torch.cos(wt)
    return pos_embed


# Input Embedding Layer
class Embedding(torch.nn.Module):
    def __init__(self, max_length, vocab_size, pad, embed_size=512):
        super(Embedding, self).__init__()
        self.embed_size = embed_size
        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=pad)
        self.pos_embed = make_pos_embed(max_length, embed_size)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        word_embed = self.embed(x) * math.sqrt(self.embed_size)
        pos_embed = self.pos_embed[:, :word_embed.shape[1]]
        result = word_embed + pos_embed
        result = self.dropout(result)
        return result

    def unembed(self, x):
        result = torch.matmul(x, self.embed.weight.transpose(0, 1))
        return result


# Multi Head Self-Attention
class SelfAttention(torch.nn.Module):
    def __init__(self, in_features=512, num_heads=8):
        super(SelfAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_features = in_features // num_heads
        self.wq = torch.nn.Linear(in_features, in_features)
        self.wk = torch.nn.Linear(in_features, in_features)
        self.wv = torch.nn.Linear(in_features, in_features)
        self.wo = torch.nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor):
        # split heads
        shape = [x.shape[0], x.shape[1], self.num_heads, self.head_features]
        q = self.wq(x).reshape(shape)
        k = self.wk(x).reshape(shape)
        v = self.wv(x).reshape(shape)
        # batched matrix multiply
        qk = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_features)
        # casual mask
        qk[torch.triu(torch.ones_like(qk), diagonal=1).bool()] = -math.inf
        # softmax
        a = torch.softmax(qk, dim=3)
        # apply attention
        result = torch.einsum('bhqk,bkhd->bqhd', a, v)
        # projection
        result = self.wo(result.reshape([*result.shape[:2], self.in_features]))
        return result


# Multi Head Memory Attention
class MemoryAttention(torch.nn.Module):
    def __init__(self, in_features=512, num_heads=8):
        super(MemoryAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_features = in_features // num_heads
        self.wq = torch.nn.Linear(in_features, in_features)
        self.wk = torch.nn.Linear(in_features, in_features)
        self.wv = torch.nn.Linear(in_features, in_features)
        self.wo = torch.nn.Linear(in_features, in_features)

    def forward(self, x: torch.Tensor, mem: torch.Tensor):
        # split heads
        q_shape = [x.shape[0], x.shape[1], self.num_heads, self.head_features]
        q = self.wq(x).reshape(q_shape)
        m_shape = [mem.shape[0], mem.shape[1], self.num_heads, self.head_features]
        k = self.wk(mem).reshape(m_shape)
        v = self.wv(mem).reshape(m_shape)
        # batched matrix multiply
        qk = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_features)
        # softmax
        a = torch.softmax(qk, dim=3)
        # apply attention
        result = torch.einsum('bhqk,bkhd->bqhd', a, v)
        # projection
        result = self.wo(result.reshape([*result.shape[:2], self.in_features]))
        return result


# Positionwise FF Network
class PositionwiseFF(torch.nn.Module):
    def __init__(self, inout_features=512, hidden_features=2048):
        super(PositionwiseFF, self).__init__()
        self.l1 = torch.nn.Linear(inout_features, hidden_features)
        self.l2 = torch.nn.Linear(hidden_features, inout_features)

    def forward(self, x):
        # L1
        x = self.l1(x)
        x = torch.relu(x)
        # L2
        x = self.l2(x)
        x = torch.relu(x)
        return x


# Encoder
class Encoder(torch.nn.Module):
    def __init__(self, embedding, num_layers=6):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.dropout = torch.nn.Dropout(0.1)
        self.ln = torch.nn.LayerNorm(embedding.embed_size)
        self.layers = []
        for _ in range(num_layers):
            attn = SelfAttention()
            ff = PositionwiseFF()
            self.layers.append((attn, ff))

    def forward(self, source):
        x = self.embedding(source)
        for attn, ff in self.layers:
            x = x + self.dropout(attn(x))
            x = self.ln(x)
            x = x + self.dropout(ff(x))
            x = self.ln(x)
        return x


# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, embedding, num_layers=6):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.dropout = torch.nn.Dropout(0.1)
        self.ln = torch.nn.LayerNorm(embedding.embed_size)
        self.layers = []
        for _ in range(num_layers):
            mem_attn = MemoryAttention()
            self_attn = SelfAttention()
            ff = PositionwiseFF()
            self.layers.append((mem_attn, self_attn, ff))

    def forward(self, encoded, target):
        x = self.embedding(target)
        for mem_attn, self_attn, ff in self.layers:
            x = x + self.dropout(mem_attn(x, encoded))
            x = self.ln(x)
            x = x + self.dropout(self_attn(x))
            x = self.ln(x)
            x = x + self.dropout(ff(x))
            x = self.ln(x)
        x = self.embedding.unembed(x)
        return x


# Transformer Model
class Transformer(torch.nn.Module):
    def __init__(self, max_length, src_vocab_size, tgt_vocab_size, sos, eos, pad, d_model=512):
        super(Transformer, self).__init__()
        src_embedding = Embedding(max_length + 1, src_vocab_size, pad=pad, embed_size=d_model)
        self.encoder = Encoder(src_embedding)
        tgt_embedding = Embedding(max_length + 1, tgt_vocab_size, pad=pad, embed_size=d_model)
        self.decoder = Decoder(tgt_embedding)
        self.sos, self.eos, self.pad = sos, eos, pad
        self.d_model = d_model
        self.warmup_steps = 4000
        self.step = 1
        self.optim = torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=10e-9)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad)

    def forward(self, source, target):
        encoded = self.encoder(source)
        result = self.decoder(encoded, target)
        return result

    def train_step(self, source, target):
        source, src_mask = augment(torch.LongTensor(source), sos=self.sos, eos=self.eos, pad=self.pad)
        target, tgt_mask = augment(torch.LongTensor(target), sos=self.sos, eos=self.eos, pad=self.pad)
        log_pr = self.forward(source, target[:, :-1])
        loss = self.loss_func(log_pr.reshape(-1, log_pr.shape[2]), target[:, 1:].reshape(-1))

        lr = self.d_model ** -0.5 * min(self.step ** -0.5, self.step * self.warmup_steps ** -1.5)
        print("lr = ", lr)
        for group in self.optim.param_groups:
            group['lr'] = lr
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.step += 1

        return loss.item()
