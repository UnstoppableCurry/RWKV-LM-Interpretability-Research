########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
import numpy as np

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

########################################################################################################

args.RUN_DEVICE = "cuda"  # cuda // cpu
# fp16 (good for GPU, does NOT support CPU) // fp32 (good for CPU) // bf16 (worse accuracy, supports CPU)
args.FLOAT_MODE = "fp16"

os.environ[
    "RWKV_JIT_ON"] = '0'  # '1' or '0'. very useful for fp32, but might be harmful for GPU fp16. please benchmark !!!

CHAT_LANG = 'Chinese'  # English // Chinese // more to come

QA_PROMPT = False  # True: Q & A prompt // False: User & Bot prompt
# 中文问答设置QA_PROMPT=True（只能问答，问答效果更好，但不能闲聊） 中文聊天设置QA_PROMPT=False（可以闲聊，但需要大模型才适合闲聊）

# Download RWKV-4 models from https://huggingface.co/BlinkDL

if CHAT_LANG == 'English':
    # args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-14B-20230204-7324'
    args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-7B-20220911-79'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230204-7324'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20221115-8047'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-Instruct-test1-20230124'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-340'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/14b-run1/rwkv-6210'

elif CHAT_LANG == 'Chinese':
    args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-1B5-EngChn-test4-20230115'
    # args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-7B-EngChn-test4-20230116'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-test4-20230115'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-test4-20230115'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-490'
    # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/1.5-run1z/rwkv-415'

if '-169M-' in args.MODEL_NAME:
    args.n_layer = 12
    args.n_embd = 768
if '-430M-' in args.MODEL_NAME:
    args.n_layer = 24
    args.n_embd = 1024
if '-1B5-' in args.MODEL_NAME or '/1.5-' in args.MODEL_NAME:
    args.n_layer = 24
    args.n_embd = 2048
elif '-3B-' in args.MODEL_NAME or '/3-' in args.MODEL_NAME:
    args.n_layer = 32
    args.n_embd = 2560
elif '-7B-' in args.MODEL_NAME or '/7-' in args.MODEL_NAME:
    args.n_layer = 32
    args.n_embd = 4096
elif '-14B-' in args.MODEL_NAME or '/14-' in args.MODEL_NAME or '/14b-' in args.MODEL_NAME:
    args.n_layer = 40
    args.n_embd = 5120

args.ctx_len = 1024

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 200

GEN_TEMP = 1.0
GEN_TOP_P = 0.85

AVOID_REPEAT = '，。：？！'

########################################################################################################

print(f'\nLoading ChatRWKV - {CHAT_LANG} - {args.RUN_DEVICE} - {args.FLOAT_MODE} - QA_PROMPT {QA_PROMPT}')
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from src.model_run import RWKV_RNN
from src.utils import TOKENIZER

tokenizer = TOKENIZER("20B_tokenizer.json")

args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0
os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
MODEL_NAME = args.MODEL_NAME

# Load Model

print(f'Loading model - {MODEL_NAME}')
model = RWKV_RNN(args)

out_onnx = 'RWKV-4-Pile-7B-20220911-79.onnx'
x = torch.tensor([float(c) for c in range(1000)]).type(torch.long).to('cuda')
# x1 = torch.randn((160, 1096)).type(torch.long).to('cuda')
x1=None
# x3 = [0]

# input_names = [ 'tokens', 'state', 'preprocess_only' ]
input_names = ['tokens', 'state']
output_names = ["output", 'model_state']
# torch_out = torch.onnx.export(model, (x, x1), out_onnx, input_names=input_names,
#                               output_names=output_names)

# traced_script_module = torch.jit.trace(model, x)
traced_script_module = torch.jit.script(model, x)

# traced_script_module.save('./')
