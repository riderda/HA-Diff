import torch
import math
from torch import nn
from torch.nn import functional as F


try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:

    MAMBA_AVAILABLE = False


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.timembedding(t)


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, c):
        return self.condEmbedding(c)


class DownSample1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.c1 = nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv1d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb=None, cemb=None):
        return self.c1(x) + self.c2(x)


class UpSample1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)

        self.conv = nn.Conv1d(in_ch, in_ch, 3, padding=1)

    def forward(self, x, temb=None, cemb=None):
        x = self.up(x)
        x = self.conv(x)
        return x


class MambaBlock(nn.Module):


    def __init__(self, in_ch, out_ch, tdim, dropout=0.1, attn=True):
        super().__init__()


        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()


        self.norm = nn.GroupNorm(32, in_ch)
        self.act = Swish()


        self.cond_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))
        self.temb_proj = nn.Sequential(Swish(), nn.Linear(tdim, out_ch))


        self.attn = AttnBlock1D(out_ch) if attn else nn.Identity()


        self.out_norm = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)

        if MAMBA_AVAILABLE:

            self.mamba = Mamba(
                d_model=out_ch,
                d_state=32,
                d_conv=4,
                expand=2,
            )
            self.use_mamba = True
        else:

            self.mamba = nn.Sequential(
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(32, out_ch),
                Swish(),
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
            )
            self.use_mamba = False

    def forward(self, x, temb, cemb):

        identity = x


        x = self.norm(x)
        x = self.act(x)
        x = self.shortcut(x)


        cond = self.temb_proj(temb)[..., None] + self.cond_proj(cemb)[..., None]
        x = x + cond


        x = x.permute(0, 2, 1)
        x = self.mamba(x)

        x = x.permute(0, 2, 1)


        x = self.out_norm(x)
        x = self.act(x)
        x = self.dropout(x)


        x = x + identity


        x = self.attn(x)

        return x


class AttnBlock1D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.q = nn.Conv1d(in_ch, in_ch, 1)
        self.k = nn.Conv1d(in_ch, in_ch, 1)
        self.v = nn.Conv1d(in_ch, in_ch, 1)
        self.out = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)
        k = self.k(h).view(B, C, -1)
        v = self.v(h).view(B, C, -1)

        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        h = torch.bmm(attn, v.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, L)
        return x + self.out(h)


class ConditionalUNet1D(nn.Module):
    def __init__(
            self,
            T=1000,
            num_labels=10,
            seq_len=200,
            feature_dim=5,
            ch=128,
            ch_mult=[1, 2, 2, 2],
            num_res_blocks=2,
            dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.tdim = ch * 4


        self.time_embedding = TimeEmbedding(T, ch, self.tdim)
        self.cond_embedding = ConditionalEmbedding(num_labels, ch, self.tdim)


        self.head = nn.Conv1d(feature_dim, ch, 3, padding=1)


        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):

                self.downblocks.append(MambaBlock(now_ch, out_ch, self.tdim, dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample1D(now_ch))
                chs.append(now_ch)


        self.middleblocks = nn.ModuleList([
            MambaBlock(now_ch, now_ch, self.tdim, dropout, attn=True),
            MambaBlock(now_ch, now_ch, self.tdim, dropout),
        ])


        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):

                self.upblocks.append(MambaBlock(chs.pop() + now_ch, out_ch, self.tdim, dropout))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample1D(now_ch))


        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv1d(now_ch, feature_dim, 3, padding=1)
        )

    def forward(self, x, t, labels):

        x = x.permute(0, 2, 1)

        temb = self.time_embedding(t)
        cemb = self.cond_embedding(labels)


        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)


        for layer in self.middleblocks:
            h = layer(h, temb, cemb)


        for layer in self.upblocks:
            if isinstance(layer, MambaBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)

        h = self.tail(h)

        h = h.permute(0, 2, 1)

        return h


if __name__ == '__main__':
    B = 2
    seq_len = 16000
    feature_dim = 5
    model = ConditionalUNet1D(T=1000, num_labels=10, seq_len=seq_len, feature_dim=5)

    x = torch.randn(B, seq_len, 5)
    t = torch.randint(0, 1000, (B,))
    labels = torch.randint(0, 11, (B,))

    out = model(x, t, labels)
