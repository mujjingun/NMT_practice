import torch
import torch.nn.functional as F
import math
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# make positional encoding
def make_pos_embed(max_length, embed_size):
    t = torch.arange(1, max_length + 1, dtype=torch.float32)
    omega = torch.arange(1, embed_size // 2 + 1, dtype=torch.float32) / embed_size
    wt = t.unsqueeze(1) * torch.pow(10000, -omega).unsqueeze(0)
    pos_embed = torch.zeros((1, max_length, embed_size))
    pos_embed[0, :, 0::2] = torch.sin(wt)
    pos_embed[0, :, 1::2] = torch.cos(wt)
    return pos_embed.to(device)


# Input Embedding Layer
class Embedding(torch.nn.Module):
    def __init__(self, max_length, vocab_size, pad, embed_size=512):
        super(Embedding, self).__init__()
        self.embed_size = embed_size
        self.embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=pad).to(device)
        self.pos_embed = make_pos_embed(max_length, embed_size)
        self.dropout = torch.nn.Dropout(0.1)
        self.unembedding = torch.nn.Linear(embed_size, vocab_size).to(device)

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
    def __init__(self, masking, in_features=512, num_heads=8):
        super(SelfAttention, self).__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_features = in_features // num_heads
        self.masking = masking
        self.wq = torch.nn.Linear(in_features, in_features).to(device)
        self.wk = torch.nn.Linear(in_features, in_features).to(device)
        self.wv = torch.nn.Linear(in_features, in_features).to(device)
        self.wo = torch.nn.Linear(in_features, in_features).to(device)

    def forward(self, x: torch.Tensor):
        # split heads
        shape = [x.shape[0], x.shape[1], self.num_heads, self.head_features]
        q = self.wq(x).reshape(shape)
        k = self.wk(x).reshape(shape)
        v = self.wv(x).reshape(shape)
        # batched matrix multiply
        qk = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_features)
        # casual mask
        if self.masking:
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
        self.wq = torch.nn.Linear(in_features, in_features).to(device)
        self.wk = torch.nn.Linear(in_features, in_features).to(device)
        self.wv = torch.nn.Linear(in_features, in_features).to(device)
        self.wo = torch.nn.Linear(in_features, in_features).to(device)

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
        self.l1 = torch.nn.Linear(inout_features, hidden_features).to(device)
        self.l2 = torch.nn.Linear(hidden_features, inout_features).to(device)

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
        self.ln = torch.nn.LayerNorm(embedding.embed_size).to(device)
        self.attns = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.attns.append(SelfAttention(masking=False))
            self.ffs.append(PositionwiseFF())

    def forward(self, source):
        x = self.embedding(source)
        for attn, ff in zip(self.attns, self.ffs):
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
        self.ln = torch.nn.LayerNorm(embedding.embed_size).to(device)
        self.mem_attns = torch.nn.ModuleList()
        self.self_attns = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mem_attns.append(MemoryAttention())
            self.self_attns.append(SelfAttention(masking=True))
            self.ffs.append(PositionwiseFF())

    def forward(self, encoded, target):
        x = self.embedding(target)
        for mem_attn, self_attn, ff in zip(self.mem_attns, self.self_attns, self.ffs):
            x = x + self.dropout(mem_attn(x, encoded))
            x = self.ln(x)
            x = x + self.dropout(self_attn(x))
            x = self.ln(x)
            x = x + self.dropout(ff(x))
            x = self.ln(x)
        x = self.embedding.unembed(x)
        return x


def augment(batch_seq):
    x = torch.LongTensor(batch_seq).to(device)
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
        self.max_length = max_length
        self.warmup_steps = 4000
        self.step = 1
        self.optim = torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=10e-9)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad)

    def forward(self, source, target):
        encoded = self.encoder(source)
        result = self.decoder(encoded, target)
        return result

    def loss(self, source, target):
        source = augment(source)
        target = augment(target)
        log_pr = self.forward(source, target[:, :-1])
        loss = self.loss_func(log_pr.reshape(-1, log_pr.shape[2]), target[:, 1:].reshape(-1))
        return loss

    def train_step(self, source, target):
        loss = self.loss(source, target)
        lr = self.d_model ** -0.5 * min(self.step ** -0.5, self.step * self.warmup_steps ** -1.5)
        for group in self.optim.param_groups:
            group['lr'] = lr
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.step += 1

        return loss.item()

    def predict(self, source):
        source = augment(source)
        batch_size = source.shape[0]
        encoded = self.encoder(source)

        # start with sos
        target = torch.zeros([batch_size, 1], dtype=torch.long)
        target.fill_(self.sos)
        target = target.to(device)
        for _ in tqdm.tqdm(range(self.max_length)):
            pr = self.decoder(encoded, target)[:, -1]
            pr = torch.softmax(pr, dim=1)
            new_col = torch.multinomial(pr, 1)
            target = torch.cat([target, new_col], dim=1)

        # add eos
        target = F.pad(target, [0, 1])
        target[:, -1] = self.eos

        # set pad after eos
        pad_mask = F.pad((target == 1)[:, :-1], [1, 0])
        target[pad_mask] = self.pad

        return target.cpu().numpy().tolist()

    def save(self, file_name):
        save_state = {
            'step': self.step,
            'weights': self.state_dict(),
            'optim': self.optim.state_dict(),
        }
        torch.save(save_state, file_name)

    def load(self, file_name):
        load_state = torch.load(file_name, map_location=device)
        self.load_state_dict(load_state['weights'])
        self.optim.load_state_dict(load_state['optim'])
        self.step = load_state['step']
