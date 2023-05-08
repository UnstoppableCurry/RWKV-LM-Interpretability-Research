########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import matplotlib.pyplot as plt
import os, copy, types, gc, sys
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
import options
import argparse

opt = options.Options().init(argparse.ArgumentParser()).parse_args()


def plt2ndarr(plt):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import PIL.Image as Image

    # 将plt转化为numpy数据
    canvas = FigureCanvasAgg(plt.gcf())
    # print(type(canvas))
    # 绘制图像
    canvas.draw()
    # 获取图像尺寸
    w, h = canvas.get_width_height()
    # 解码string 得到argb图像
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)

    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象 (需要Image对象的同学到此为止就可以了)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    # print(rgb_image.shape)
    videoWrite.write(rgb_image)
    return rgb_image


def draw_box_string(img, x, y, string):
    """
    img: imread读取的图片;
    x,y:字符起始绘制的位置;
    string: 显示的文字;
    return: img
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    # simhei.ttf 是字体，你如果没有字体，需要下载
    font = ImageFont.truetype("/usr/share/fonts/zh/simsun.ttc", 10, encoding="utf-8")
    # font = ImageFont.truetype("SourceHanSansCN-Regular.ttf", 50, encoding="utf-8")
    draw.text((x, y - 50), string, (0, 0, 0), font=font)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img

def print_interpretability(result):
    lower, upper = result['99%_range']  # 99%范围

    print(' 分布检查结果 -->',result['distribution']+' 标准差 -->',result['std']+' 方差 -->',result['var']+' KL散度 -->',result['kl_div']+' 99%数值范围 -->',lower,upper)

def interpretability(ffn):
    '''
选择一个合适的THRESHOLD值是比较主观的,取值范围较广。一般来说,一个好的THRESHOLD值应满足:

• 大于0。THRESHOLD永远应大于0,表示两者差值实际上有一定的差距,分布不完全相同。

• 小于1。THRESHOLD取值过大,例如大于1,就失去了判断的意义。任何非0差值都会被看作接近正态分布的了,没有实际作用。

• 根据具体问题和数据选择。不同问题和数据集的THRESHOLD最佳值会有所不同。 genral 一个范围而言,0.1到0.5是较常见而合理的取值,但需要通过试验获得最佳THRESHOLD。

• 兼顾严谨性和准确性。THRESHOLD越小,判断越严谨谨慎,仅看作最接近正态分布的分布为正态分布。THRESHOLD越大,判断越宽容,较为偏离的分布也会被看作为正态分布。需要平衡这两个方面,既要准确又要谨慎。

• 考虑后续操作。THRESHOLD的选择应考虑后续任务的要求。高严谨性要求的下游任务通常需要较小的THRESHOLD;简单分类等操作THRESHOLD可以稍大。

一些示例THRESHOLD取值及其含义:

• 0.1:较严谨的判断,只视最接近正态分布的分布为正态分布。
• 0.3:中等严谨度,较多分布会被看作正态分布,宽容度适中。这是较常见取值的上限。
• 0.5:较宽松的判断,许多偏离正态分布的分布也会被认定为正态分布。严谨度相对较低,准确性相对较高。
• 大于1:判断失去严谨性和实际意义,不建议采用。

所以在实际实现中,选取THRESHOLD=0.1 to 0.5是较为合理的取值范围。但最终选择还是需要根据具体问题、数据特征和后续操作要求来确定的。同时,也可以完全舍弃THRESHOLD,转而使用KS检验或观察視覺化等更为严谨的方法判断分布。

总之,THRESHOLD是一个用于判断分布是否为正态分布的人工阈值参数。其选取应兼顾严谨性和准确性,多考虑具体环境因素,通过评估实践获得最佳值,或用更严谨方法替换THRESHOLD判断。

如有任何不明白或问题的地方,尽管继续提问。我很乐意继续为你提供帮助!
    '''
    THRESHOLD = 0.5
    # 1. 检查分布
    dist = torch.normal(0, 1, size=ffn.size()).to(args.RUN_DEVICE)  # 正态分布Tensor
    dist = (ffn - dist).pow(2).sum().to('cpu')/ffn.size()[0] # 计算差的L2范数
    dist=dist.to('cpu')
    # print(f'是否正态分布: {dist < THRESHOLD}')

    # 2. 离散程度相关
    std = torch.std(ffn)  # 标准差
    variance = torch.var(ffn)  # 方差
    kl_div = torch.kl_div(ffn.float(), torch.normal(0, 1, size=ffn.size()).to(args.RUN_DEVICE) )  # KL散度

    # 3. 找到99%分位数
    upper_bound = torch.quantile(ffn.float(), 0.99)
    lower_bound = -torch.quantile(ffn.float(), 0.99)

    # 4. 根据范围去除异常值
    final_tensor = torch.clamp(ffn, min=lower_bound, max=upper_bound)

    return {
        'distribution': dist < THRESHOLD,
        'std': std,
        'var': variance,
        'kl_div': kl_div,
        '99%_range': (lower_bound, upper_bound),
        'processed_tensor': final_tensor
    }


try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

########################################################################################################

args.RUN_DEVICE =opt.RUN_DEVICE #"cuda"  # cuda // cpu
# fp16 (good for GPU, does NOT support CPU) // fp32 (good for CPU) // bf16 (worse accuracy, supports CPU)
args.FLOAT_MODE =opt.FLOAT_MODE #"fp16"

os.environ[
    "RWKV_JIT_ON"] = '0'  # '1' or '0'. very useful for fp32, but might be harmful for GPU fp16. plea se benchmark !!!

CHAT_LANG = opt.CHAT_LANG#'Chinese'  # English // Chinese // more to come

QA_PROMPT = False  # True: Q & A prompt // False: User & Bot prompt
# 中文问答设置QA_PROMPT=True（只能问答，问答效果更好，但不能闲聊） 中文聊天设置QA_PROMPT=False（可以闲聊，但需要大模型才适合闲聊）

# Download RWKV-4 models from https://huggingface.co/BlinkDL

if CHAT_LANG == 'English':
    # args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-14B-20230213-8019'
    args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-7B-20220911-79'
    # args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-1B5-20220903-8040'
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
    args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-7B-EngChn-test4-20230116'
    # args.MODEL_NAME = '/www/model/rwkv/RWKV-4-Pile-1B5-EngChn-test4-20230115'
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

if CHAT_LANG == 'English':
    interface = ":"

    if QA_PROMPT:
        user = "Q"
        bot = "A"
        intro = f'The following is a verbose and detailed Q & A conversation of factual information.'
    else:
        user = "User"
        bot = "Bot"
        intro = f'The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.'

    init_prompt = f'''
{intro}

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+++ --> continue last free generation (only for +gen / +qa)
++ --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
'''
elif CHAT_LANG == 'Chinese':
    interface = ":"
    if QA_PROMPT:
        user = "Q"
        bot = "A"
        init_prompt = f'''
Expert Questions & Helpful Answers

Ask Research Experts

'''
    else:
        user = "User"
        bot = "Bot"
        init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

{user}{interface} 企鹅会飞吗

{bot}{interface} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

'''
    HELP_MSG = '''指令:

直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行
+ --> 让机器人换个回答
+reset --> 重置对话

+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+qq 某某问题 --> 问独立的问题（忽略上下文），且敞开想象力，用\\n代表换行
+++ --> 继续 +gen / +qa / +qq 的回答
++ --> 换个 +gen / +qa / +qq 的回答

现在可以输入内容和机器人聊天（注意它不大懂中文，它可能更懂英文）。请经常使用 +reset 重置机器人记忆。
目前没有“重复惩罚”，所以机器人有时会重复，此时必须使用 + 换成正常回答，以免污染电脑记忆。
注意：和上下文无关的独立问题，必须用 +qa 或 +qq 问，以免污染电脑记忆。
'''

# Load Model

print(f'Loading model - {MODEL_NAME}')
model = RWKV_RNN(args)

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = tokenizer.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd


########################################################################################################

def run_rnn(tokens, newline_adj=0,outlier=[-2]):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(tokens,type(tokens ))
    out, all_ffn_out, model_state = model.forward(tokens, model_state,outlier=outlier)
    # assss  = all_ffn_out[0].to('cpu').numpy()
    # data=np.array([x.to('cpu').numpy() for x in all_ffn_out])

    # plt.plot(assss)
    # ax.plot_surface(ax, rstride=1, cstride=1, cmap='rainbow')
    # plt2ndarr(plt)
    # plt.show()
    # print(f'### model ###\n{tokens}\n[{tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj  # adjust \n probability
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out, all_ffn_out


all_state = {}


def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


########################################################################################################

# Run inference
print(f'\nRun prompt...')

out, all_ffn_out = run_rnn(tokenizer.encode(init_prompt))
save_all_stat('', 'chat_init', out)
gc.collect()
torch.cuda.empty_cache()

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{tokenizer.decode(model_tokens)}]\n')


def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')


def draw_ffn(send_msg):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    # ax = Axes3D(fig)
    ax.set_zlim(-0, 500)
    ax.set_zlabel('value')
    ax.set_xlabel('channel')
    ax.set_ylabel('layer')

    # ax.set_title(send_msg, fontsize=10)
    print('send_msg->', send_msg)
    if len(all_ffn_out[1].shape) == 1:
        for index in range(len(all_ffn_out)):
            ffn_out = all_ffn_out[index]
            x = [x for x in range(ffn_out.shape[0])]
            y = ffn_out.to('cpu').numpy()
            # ax.bar(ffn_out.to('cpu').numpy(),[x for x in range(2048)])
            # ax.bar(ffn_out.to('cpu').numpy(),[x for x in range(2048)])
            # ax.plot([x for x in range(2048)],ffn_out.to('cpu').numpy(),zs=index)
            ax.plot(x, y, zs=index, zdir='y')
        # ax.text(4, 6, s=send_msg, fontsize=5, color='green')
        # 将Matplotlib图像转换为OpenCV图像格式
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, send_msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # 将图像写入视频
        videoWrite.write(img)

        # 清除轴对象并准备下一帧
        ax.clear()
        plt.show()
        plt.close()


def on_message(message):
    global model_tokens, model_state

    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()
    # if len(msg) > 1000:
    #     reply_msg('your message is too long (max 1000 tokens)')
    #     return

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0

    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out, all_ffn_out = run_rnn(tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = (i - CHAT_LEN_LONG) * 0.25  # MUST END THE GENERATION
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p=x_top_p,
            )
            out, all_ffn_out = run_rnn([token], newline_adj=newline_adj,outlier=opt.outlier)

            xxx = tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:  # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1

            send_msg = tokenizer.decode(model_tokens[begin:])

            # draw_ffn(send_msg)
            # plt2ndarr(plt)
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            # ax = Axes3D(fig)
            ax.set_zlim(-500, 500)
            ax.set_zlabel('value')
            ax.set_xlabel('channel')
            ax.set_ylabel('layer')

            # ax.set_title(send_msg, fontsize=10)
            # print('send_msg->', send_msg)

            if len(all_ffn_out[1].shape) == 1:
                for index in range(len(all_ffn_out)):
                    ffn_out = all_ffn_out[index]
                    ffn_result=interpretability(ffn_out)
                    if opt.print_interpretability:
                        print_interpretability(ffn_result)
                    x = [x for x in range(ffn_out.shape[0])]
                    y = ffn_out.to('cpu').numpy()
                    # ax.bar(ffn_out.to('cpu').numpy(),[x for x in range(2048)])
                    # ax.bar(ffn_out.to('cpu').numpy(),[x for x in range(2048)])
                    # ax.plot([x for x in range(2048)],ffn_out.to('cpu').numpy(),zs=index)
                    ax.plot(x, y, zs=index, zdir='y')
                # ax.text(4, 6, s=send_msg, fontsize=5, color='green')
                # 将Matplotlib图像转换为OpenCV图像格式
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                send_msg_y = 100
                send_first_lens = 80
                if len(send_msg) // send_first_lens > 1:
                    for send_msg_index in range(len(send_msg) // send_first_lens):
                        if send_msg_index == 0:
                            # cv2.putText(img, send_msg[0:20], (10, send_msg_y), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 0, 0), 2)
                            img=draw_box_string(img,10,send_msg_y, send_msg[0:send_first_lens])
                            # img = draw_box_string(img, 10, send_msg_y, u'你好啊')
                        else:
                            if send_msg_index == len(send_msg) // 20:
                                # cv2.putText(img, send_msg[send_msg_index * 20 + 20:],
                                #             (10, send_msg_y+35), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 0, 0), 2)
                                img = draw_box_string(img, 10, send_msg_y + 35,
                                                      send_msg[send_msg_index * send_first_lens + send_first_lens:])

                            # cv2.putText(img, send_msg[send_msg_index*20:send_msg_index*20+20], (10, send_msg_y), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 0, 0), 2)
                            img = draw_box_string(img, 10, send_msg_y, send_msg[
                                                                       send_msg_index * send_first_lens:send_msg_index * send_first_lens + send_first_lens])

                        send_msg_y += 35

                else:
                    # cv2.putText(img, send_msg, (10, send_msg_y), cv2.FONT_HERSHEY_DUPLEX , 1, (0, 0, 0), 2)
                    img = draw_box_string(img, 10, send_msg_y, send_msg)
                if opt.show:
                    cv2.imshow('outleir_runtime',img)
                # 将图像写入视频
                videoWrite.write(img)

                # 清除轴对象并准备下一帧
                ax.clear()
                # plt.show()
                plt.close()
            if '\n\n' in send_msg:
                videoWrite.release()
                cv2.destroyAllWindows()
                send_msg = send_msg.strip()
                break

            # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)


index = 0
while True:
    index += 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite = cv2.VideoWriter(opt.video_name + str(index) + '.mp4', fourcc, opt.video_FPS, opt.video_size)
    msg = input(f'{user}{interface} ')
    # msg='hello'
    # msg='+gen Tell me what Python is and what its characteristics are. Please demonstrate your ability to write code and explain what it can do.'
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Error: please say something')
