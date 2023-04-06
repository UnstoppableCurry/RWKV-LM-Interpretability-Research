"""Convert RWKV PyTorch savepoint to TorchScript model.
"""
from typing import NamedTuple, List, Optional, Final

import torch
import torch.nn as nn
import torch.nn.functional as F

import click


class LayerNorm(NamedTuple):
    weight: nn.Parameter
    bias: nn.Parameter


class ATT(NamedTuple):
    time_mix_k: nn.Parameter
    time_mix_v: nn.Parameter
    time_mix_r: nn.Parameter
    time_first: nn.Parameter
    time_decay: nn.Parameter
    key: nn.Parameter
    value: nn.Parameter
    receptance: nn.Parameter
    output: nn.Parameter


class FFN(NamedTuple):
    time_mix_k: nn.Parameter
    time_mix_r: nn.Parameter
    key: nn.Parameter
    value: nn.Parameter
    receptance: nn.Parameter


class Block(NamedTuple):
    att: ATT
    ffn: FFN
    ln1: LayerNorm
    ln2: LayerNorm


class Weight(NamedTuple):
    emb: nn.Parameter
    blocks: List[Block]
    ln0: LayerNorm
    ln_out: LayerNorm
    head: nn.Parameter


class RWKV_RNN_JIT(nn.Module):
    float_mode: Final [torch.dtype]
    n_layer: Final[int]
    n_embd: Final[int]
    device: Final[torch.device]

    RWKV_RESCALE_LAYER: final[int] = 6

    weight: Weight

    def __init__(
        self,
        *,
        model_path: str,
        float_mode: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.float_mode = float_mode
        self.device = device

        with torch.no_grad():
            w = torch.load(model_path, map_location="cpu")
            n_embd = w["emb.weight"].shape[1]
            n_layer = 0

            keys = list(w.keys())
            print_need_newline = False
            print(keys)

            for x in keys:
                w[x].requires_grad = False
                if x == "emb.weight" or "ln0" in x:
                    continue

                block_id = int(x.split(".")[1]) if ("blocks." in x) else 0
                n_layer = max(n_layer, block_id + 1)

                if ".time_" in x:
                    w[x] = w[x].squeeze()
                if (
                    "key.weight" in x
                    or "value.weight" in x
                    or "receptance.weight" in x
                    or "output.weight" in x
                ):
                    w[x] = w[x].t()

                if ".time_decay" in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif ".time_first" in x:
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].to(dtype=self.float_mode)

                if float_mode == torch.float16:
                    if "att.output.weight" in x:
                        w[x] = w[x] / (2 ** int(block_id // self.RWKV_RESCALE_LAYER))
                    if "ffn.value.weight" in x:
                        w[x] = w[x] / (2 ** int(block_id // self.RWKV_RESCALE_LAYER))

                w[x] = w[x].to(device)

                shape = w[x].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 1:
                    shape = f"  {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f"  {str(shape[0]).rjust(5)}      "
                if block_id == 0:
                    if print_need_newline:
                        print("\n", end="")
                        print_need_newline = False
                    print(
                        x.ljust(32),
                        str(w[x].dtype).replace("torch.", "").ljust(10),
                        w[x].device,
                        shape,
                    )
                else:
                    print_need_newline = True
                    print(".", end="", flush=True)
        print()
        print('n_layer ',n_layer,'n_embd',n_embd)
        self.n_layer = n_layer
        self.n_embd = n_embd

        emb = w["emb.weight"]
        ln_out = LayerNorm(w["ln_out.weight"], w["ln_out.bias"])
        ln0 = LayerNorm(w["blocks.0.ln0.weight"], w["blocks.0.ln0.bias"])
        head = w["head.weight"]
        blocks = [
            Block(
                att=ATT(
                    time_mix_k=w[f"blocks.{i}.att.time_mix_k"],
                    time_mix_v=w[f"blocks.{i}.att.time_mix_v"],
                    time_mix_r=w[f"blocks.{i}.att.time_mix_r"],
                    time_first=w[f"blocks.{i}.att.time_first"],
                    time_decay=w[f"blocks.{i}.att.time_decay"],
                    key=w[f"blocks.{i}.att.key.weight"],
                    value=w[f"blocks.{i}.att.value.weight"],
                    receptance=w[f"blocks.{i}.att.receptance.weight"],
                    output=w[f"blocks.{i}.att.output.weight"],
                ),
                ffn=FFN(
                    time_mix_k=w[f"blocks.{i}.ffn.time_mix_k"],
                    time_mix_r=w[f"blocks.{i}.ffn.time_mix_r"],
                    key=w[f"blocks.{i}.ffn.key.weight"],
                    value=w[f"blocks.{i}.ffn.value.weight"],
                    receptance=w[f"blocks.{i}.ffn.receptance.weight"],
                ),
                ln1=LayerNorm(w[f"blocks.{i}.ln1.weight"], w[f"blocks.{i}.ln1.bias"]),
                ln2=LayerNorm(w[f"blocks.{i}.ln2.weight"], w[f"blocks.{i}.ln2.bias"]),
            )
            for i in range(self.n_layer)
        ]

        with torch.no_grad():  # precompute embedding
            x = self.LN(emb, ln0)
            emb = x.to(dtype=self.float_mode)

        self.weight = Weight(emb, blocks, ln0, ln_out, head)

    def LN(self, x, w: LayerNorm):
        return F.layer_norm(x, (self.n_embd,), weight=w.weight, bias=w.bias)

    def FF_one(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = state[5 * i + 0].to(dtype=self.float_mode)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5 * i + 0] = x.float()

        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    def FF_seq(self, x, state, i: int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = torch.cat(
            (state[5 * i + 0].to(dtype=self.float_mode).unsqueeze(0), x[:-1, :])
        )
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5 * i + 0] = x[-1, :].float()

        r = torch.sigmoid(xr @ rw)
        k = torch.square(torch.relu(xk @ kw))
        kv = k @ vw
        return r * kv

    def SA_one(
        self,
        x,
        state,
        i: int,
        time_mix_k,
        time_mix_v,
        time_mix_r,
        time_first,
        time_decay,
        kw,
        vw,
        rw,
        ow,
    ):
        xx = state[5 * i + 1].to(dtype=self.float_mode)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5 * i + 1] = x.float()

        r = torch.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5 * i + 2]
        bb = state[5 * i + 3]
        pp = state[5 * i + 4]
        ww = time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        state[5 * i + 2] = e1 * aa + e2 * v
        state[5 * i + 3] = e1 * bb + e2
        state[5 * i + 4] = p
        wkv = (a / b).to(dtype=self.float_mode)
        return (r * wkv) @ ow

    def SA_seq(
        self,
        x,
        state,
        i: int,
        time_mix_k,
        time_mix_v,
        time_mix_r,
        time_first,
        time_decay,
        kw,
        vw,
        rw,
        ow,
    ):
        xx = torch.cat(
            (state[5 * i + 1].to(dtype=self.float_mode).unsqueeze(0), x[:-1, :])
        )
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5 * i + 1] = x[-1, :].float()

        r = torch.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5 * i + 2]
        bb = state[5 * i + 3]
        pp = state[5 * i + 4]
        T = x.shape[0]
        for t in range(T):
            ww = time_first + k[t]
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = torch.maximum(ww, k[t])
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k[t] - p)
            if t != T - 1:
                aa = e1 * aa + e2 * v[t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[5 * i + 2] = e1 * aa + e2 * v[t]
                state[5 * i + 3] = e1 * bb + e2
                state[5 * i + 4] = p
            xx[t] = (a / b).to(dtype=self.float_mode)
        return (r * xx) @ ow

    def FF(
        self,
        x,
        state,
        i: int,
        time_mix_k,
        time_mix_r,
        kw,
        vw,
        rw,
        *,
        seq_mode: bool,
    ):
        if seq_mode:
            return self.FF_seq(x, state, i, time_mix_k, time_mix_r, kw, vw, rw)
        else:
            return self.FF_one(x, state, i, time_mix_k, time_mix_r, kw, vw, rw)

    def SA(
        self,
        x,
        state,
        i: int,
        time_mix_k,
        time_mix_v,
        time_mix_r,
        time_first,
        time_decay,
        kw,
        vw,
        rw,
        ow,
        *,
        seq_mode: bool,
    ):
        if seq_mode:
            return self.SA_seq(
                x,
                state,
                i,
                time_mix_k,
                time_mix_v,
                time_mix_r,
                time_first,
                time_decay,
                kw,
                vw,
                rw,
                ow,
            )
        else:
            return self.SA_one(
                x,
                state,
                i,
                time_mix_k,
                time_mix_v,
                time_mix_r,
                time_first,
                time_decay,
                kw,
                vw,
                rw,
                ow,
            )

    def forward(
        self,
        tokens: List[int],
        state: Optional[torch.Tensor],
        # state_is_none: bool,
        preprocess_only: bool = False,
    ):
        with torch.no_grad():
            w = self.weight

            seq_mode = len(tokens) > 1

            x = w.emb[tokens] if seq_mode else w.emb[tokens[-1]]
            x = x.to(self.device)

            if state is None:
                state = torch.zeros(self.n_layer * 5, self.n_embd, device=self.device)
                for i in range(self.n_layer):
                    state[5 * i + 4] -= 1e30

            for i in range(self.n_layer):
                ww = w.blocks[i].att
                x = x + self.SA(
                    self.LN(x, w.blocks[i].ln1),
                    state,
                    i,
                    ww.time_mix_k,
                    ww.time_mix_v,
                    ww.time_mix_r,
                    ww.time_first,
                    ww.time_decay,
                    ww.key,
                    ww.value,
                    ww.receptance,
                    ww.output,
                    seq_mode=seq_mode,
                )

                ww = w.blocks[i].ffn
                x = x + self.FF(
                    self.LN(x, w.blocks[i].ln2),
                    state,
                    i,
                    ww.time_mix_k,
                    ww.time_mix_r,
                    ww.key,
                    ww.value,
                    ww.receptance,
                    seq_mode=seq_mode,
                )

                if (
                    self.float_mode == torch.float16
                    and (i + 1) % self.RWKV_RESCALE_LAYER == 0
                ):
                    x = x / 2

            if preprocess_only:
                return torch.empty(1), state

            x = self.LN(x[-1, :], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
            x = w.head @ x

            return x.float(), state


# @click.command()
# @click.option(
#     "--float-mode",
#     type=click.Choice(["fp32", "fp16", "bf16"]),
# )
# @click.option("--device", type=click.Choice(["cpu", "cuda"]))
# @click.argument("model_path", type=click.Path(exists=True))
# @click.argument("output_path", type=click.Path())
def convert(float_mode, device, model_path, output_path):
    float_modes = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf32": torch.bfloat16,
    }

    model = RWKV_RNN_JIT(
        model_path=model_path,
        float_mode=float_modes[float_mode],
        device=torch.device(device),
    )
    model = torch.jit.script(model)

    if float_mode == "bf16":
        model = model.bfloat16()
    elif float_mode == "fp16":
        model = model.half()
    else:
        model = model.float()

    if device == "cuda":
        model = model.cuda()

    model.save(output_path)

    print("脚本编写时的 ChatRWKV 版本为：git+3286707，其他版本可能不兼容。")


if __name__ == "__main__":
    MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-1B5-EngChn-test4-20230115.pth'

    convert('fp16','cuda:0',MODEL_NAME,'./')