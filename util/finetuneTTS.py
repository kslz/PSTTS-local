import os
import shutil
import subprocess
import yaml
from yacs.config import CfgNode
import json
from pathlib import Path
import soundfile as sf
from typing import List
import numpy as np
import paddle
import re

from paddlespeech.t2s.exps.syn_utils import am_to_static
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import run_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.modules.nets_utils import make_pad_mask
from paddlespeech.t2s.modules.normalizer import ZScore


# 配置路径信息
# 这里的路径是提前写好在 Aistudio 上的路径信息，如果需要迁移到其它的环境，请自行配置
cwd_path = "./"
pretrained_model_dir = "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0"
config_path = "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/conf/finetune.yaml"
inference_dir = "./inference"
exp_base = "./work/exp_"

os.makedirs(inference_dir, exist_ok=True)

pwg_inference_dir = os.path.join(cwd_path, "inference/pwgan_aishell3_static_1.1.0")
hifigan_inference_dir = os.path.join(cwd_path, "inference/hifigan_aishell3_static_1.1.0")
wavernn_inference_dir = os.path.join(cwd_path, "inference/wavernn_csmsc_static_0.2.0")


# 命令行执行函数，可以进入指定路径下执行
def run_cmd(cmd, cwd_path, message, check_result_path=None):
    p = subprocess.Popen(cmd, shell=True, cwd=cwd_path)
    res = p.wait()
    print(cmd)
    print("运行结果：", res)

    # 检查是否有对应文件生成
    if check_result_path:
        file_num = [filename for filename in os.listdir(check_result_path)]
        if file_num == 0:
            res = 1

    if res == 0:
        # 运行成功
        return True
    else:
        # 运行失败
        print(message)
        print(f"你可以新建终端，打开命令行，进入：{cwd_path}")
        print(f"执行：{cmd}")
        print("查看命令行中详细的报错信息")
        return False

def status_change(status_path, status):
    with open(status_path, "w", encoding="utf8") as f:
        status = {
            'on_status': status
        }
        json.dump(status, f, indent=4)

# 找到最新生成的模型
def find_max_ckpt(model_path):
    max_ckpt = 0
    for filename in os.listdir(model_path):
        if filename.endswith('.pdz'):
            files = filename[:-4]
            a1, a2, it = files.split("_")
            if int(it) > max_ckpt:
                max_ckpt = int(it)
    return max_ckpt

# Step1 : 生成标准数据集
def step1_generate_standard_dataset(label_path, exp_path, data_dir):
    error_code = 0

    print("Step1 开始执行: 生成标注数据集")
    # 生成 data 数据路径
    exp_data_path = data_dir
    os.makedirs(exp_data_path, exist_ok=True)
    exp_data_label_file = os.path.join(exp_data_path, "labels.txt")

    if not os.path.exists(label_path):
        print(f"标注文件: {label_path} 不存在，请检查数据预标注是否成功！")
        error_code = 1
        return None, error_code
    
    with open(label_path, "r", encoding='utf8') as f:
        ann_result = json.load(f)

    if len(ann_result) == 0:
        print(f"标注文件: {label_path} 内容为空，请检查数据预标注是否成功！")
        error_code = 2
        return None, error_code
    
    if len(ann_result) < 5:
        print(f"标注文件: {label_path} 中，音频数据小于5条，请增加上传的音频数量！")
        error_code = 3
        return None, error_code

    # 复制音频文件
    for label in ann_result:
        if os.path.exists(label['filepath']):
            shutil.copy(label['filepath'], exp_data_path)
        else:
            print(f"音频文件：{label['filepath']} 不存在，请重新检查数据是否正确")
            error_code = 4
            return None, error_code

    # 生成标准标注文件
    with open(exp_data_label_file, "w", encoding="utf8") as f:
        for ann in ann_result:
            f.write(f"{ann['filename'].split('.')[0]}|{ann['pinyin']}\n")
    
    # 检查 data 里面是否为空，为空则显示异常
    wav_file = [filename for filename in os.listdir(exp_data_path) if filename.endswith(".wav")]
    if len(wav_file) == 0:
        # 为空则是异常
        shutil.rmtree(exp_data_path)
        error_code = 5
        return None, error_code
    return exp_data_path, error_code
    

# Step2: 检查非法数据
def step2_check_oov(data_dir, new_dir, lang="zh"):
    print("Step2 开始执行: 检查数据集是否合法")
    cmd = f"""
            python3 /home/aistudio/util/check_oov.py \
                --input_dir={data_dir} \
                --pretrained_model_dir={pretrained_model_dir} \
                --newdir_name={new_dir} \
                --lang={lang}
            """
    if not run_cmd(cmd, cwd_path, message="Step2 检查非法数据 执行失败，请检查数据集是否配置正确"):
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        return False
    else:
        return True

# Step3: 生成 MFA 对齐结果
def step3_get_mfa(new_dir, mfa_dir, lang="zh"):
    print("Step3 开始执行: 使用 MFA 对数据进行对齐")
    cmd = f"""
        python3 /home/aistudio/util/get_mfa_result.py \
            --input_dir={new_dir} \
            --mfa_dir={mfa_dir} \
            --lang={lang}
        """
    if not run_cmd(cmd, cwd_path, message="Step3 MFA对齐 执行失败，请执行命令行，查看 MFA 详细报错", check_result_path=mfa_dir):
        if os.path.exists(mfa_dir):
            shutil.rmtree(mfa_dir)
        return False
    else:
        return True

# Step4: 生成 Duration 
def step4_duration(mfa_dir):
    if not os.path.exists(mfa_dir):
        print("Step3 执行失败，未生成 MFA 对齐文件夹")
        return False
    print("Step4 开始执行: 生成 Duration 文件")
    cmd = f"""
        python3 /home/aistudio/util/generate_duration.py \
            --mfa_dir={mfa_dir}
        """
    if not run_cmd(cmd, cwd_path, message="Step4 生成 Duration 执行失败，请执行命令行"):
        if os.path.exists("./durations.txt"):
            os.remove("./durations.txt")
        return False
    else:
        return True

# Step5: 数据预处理
def step5_extract_feature(data_dir, dump_dir):
    print("Step5 开始执行: 开始数据预处理")
    cmd = f"""
        python3 /home/aistudio/util/extract_feature.py \
            --duration_file="./durations.txt" \
            --input_dir={data_dir} \
            --dump_dir={dump_dir}\
            --pretrained_model_dir={pretrained_model_dir}\
        """
    if not run_cmd(cmd, cwd_path, message="Step5 执行失败，请执行命令行"):
        if os.path.exists(dump_dir):
            shutil.rmtree(dump_dir)
        return False
    else:
        return True

# Step6: 准备微调环境
def step6_prepare_env(output_dir):
    print("Step6 开始执行: 准备微调环境")
    cmd = f"""
        python3 /home/aistudio/util/prepare_env.py \
            --pretrained_model_dir={pretrained_model_dir} \
            --output_dir={output_dir}
    """
    if not run_cmd(cmd, cwd_path, message="Step6 准备训练环境 执行失败，请执行命令行"):
        return False
    else:
        return True

# Step7: 微调训练
def step7_finetune(dump_dir, output_dir, max_finetune_step, batch_size=None, learning_rate=None):
    # # 读取默认的yaml文件
    # with open(config_path) as f:
    #     finetune_config = yaml.safe_load(f)
    # # 多少个 step 保存一次
    # finetune_config['num_snapshots'] = num_snapshots
    # # 1. 自动调整 batch ，通过 dump/train里面文件数量判断
    train_data_dir = os.path.join(dump_dir, "train/norm/data_speech")
    file_num = len([filename for filename in os.listdir(train_data_dir) if filename.endswith(".npy")])
    if not batch_size:    
        if file_num <= 32:
            batch_size = file_num
        else:
            batch_size = 32
    else:
        # 指定 batch size 的情况下还是需要看音频文件的数量
        if file_num <= batch_size:
            batch_size = file_num

    # # 2. 支持调整 learning_rate
    if not learning_rate:
        learning_rate = 0.001

    print("Step7 开始执行: 微调试验开始")
    
    cmd = f"""
        python3 /home/aistudio/util/finetuneTrain.py \
            --dump_dir={dump_dir} \
            --output_dir={output_dir} \
            --max_step={max_finetune_step} \
            --batch_size={batch_size} \
            --learning_rate={learning_rate}
        """
    if not run_cmd(cmd, cwd_path, message="Step7 微调试验失败 执行失败，请执行命令行"):
        return False
    else:
        return True

# 导出成静态图
def step8_export_static_model(exp_name):
    exp_path = os.path.join(exp_base + exp_name)
    model_path = os.path.join(exp_path, "output/checkpoints")
    dump_dir = os.path.join(exp_path, "dump")
    output_dir = os.path.join(exp_path, "output")
    ckpt = find_max_ckpt(model_path)
    
    am_config = f"{cwd_path}/models/fastspeech2_mix_ckpt_1.2.0/default.yaml"

    with open(am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    finetune_model_path = f"{output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz"
    am_inference = get_am_inference(
        am="fastspeech2_mix",
        am_config=am_config,
        am_ckpt=finetune_model_path,
        am_stat=f"{cwd_path}/models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy",
        phones_dict=f"{dump_dir}/phone_id_map.txt",
        tones_dict=None,
        speaker_dict=f"{dump_dir}/speaker_id_map.txt")

    out_inference_dir = os.path.join(inference_dir, exp_name)
    # 静态图导出
    am_inference = am_to_static(
            am_inference=am_inference,
            am="fastspeech2_mix",
            inference_dir=out_inference_dir,
            speaker_dict=f"{dump_dir}/speaker_id_map.txt")
    
    # 把 phone_dict 也复制过去
    shutil.copy(f"{dump_dir}/phone_id_map.txt", out_inference_dir)

    # 把预训练模型中 npy 文件，yaml 文件，txt文件都放进去
    for filename in os.listdir(pretrained_model_dir):
        fileend = os.path.splitext(filename)[1]
        if fileend in [".npy", ".yaml", ".txt"]:
            shutil.copy(f"{pretrained_model_dir}/{filename}", out_inference_dir)
    
    # 动态图模型也放进去
    if os.path.exists(finetune_model_path):
        shutil.copy(finetune_model_path, out_inference_dir)


# 使用微调后的模型生成音频
def step9_generateTTS(text_dict, wav_output_dir, exp_name, voc="PWGan"):
    # 配置一下参数信息
    exp_path = os.path.join(exp_base + exp_name)
    model_path = os.path.join(exp_path, "output/checkpoints")
    output_dir = os.path.join(exp_path, "output")
    dump_dir = os.path.join(exp_path, "dump")
    
    text_file = os.path.join(exp_path, "sentence.txt")
    status_json = os.path.join(exp_path, "generate_status.json")
    # 写一个物理锁
    status_change(status_json, True)
    
    with open(text_file, "w", encoding="utf8") as f:
        for k,v in sorted(text_dict.items(), key=lambda x:x[0]):
            f.write(f"{k} {v}\n")

    ckpt = find_max_ckpt(model_path)

    if voc == "PWGan":
        cmd = f"""
        python3 /home/aistudio/PaddleSpeech/paddlespeech/t2s/exps/fastspeech2/../synthesize_e2e.py \
                        --am=fastspeech2_mix \
                        --am_config=models/fastspeech2_mix_ckpt_1.2.0/default.yaml \
                        --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                        --am_stat=models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
                        --voc="pwgan_aishell3" \
                        --voc_config=models/pwg_aishell3_ckpt_0.5/default.yaml \
                        --voc_ckpt=models/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
                        --voc_stat=models/pwg_aishell3_ckpt_0.5/feats_stats.npy \
                        --lang=mix \
                        --text={text_file} \
                        --output_dir={wav_output_dir} \
                        --phones_dict={dump_dir}/phone_id_map.txt \
                        --speaker_dict={dump_dir}/speaker_id_map.txt \
                        --spk_id=0 \
                        --inference_dir
                        --ngpu=1
        """
    elif voc == "WaveRnn":
        cmd = f"""
        python3 /home/aistudio/PaddleSpeech/paddlespeech/t2s/exps/fastspeech2/../synthesize_e2e.py \
                        --am=fastspeech2_mix \
                        --am_config=models/fastspeech2_mix_ckpt_1.2.0/default.yaml \
                        --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                        --am_stat=models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
                        --voc="wavernn_csmsc" \
                        --voc_config=models/wavernn_csmsc_ckpt_0.2.0/default.yaml \
                        --voc_ckpt=models/wavernn_csmsc_ckpt_0.2.0/snapshot_iter_400000.pdz \
                        --voc_stat=models/wavernn_csmsc_ckpt_0.2.0/feats_stats.npy \
                        --lang=mix \
                        --text={text_file} \
                        --output_dir={wav_output_dir} \
                        --phones_dict={dump_dir}/phone_id_map.txt \
                        --speaker_dict={dump_dir}/speaker_id_map.txt \
                        --spk_id=0 \
                        --ngpu=1
        """
    elif voc == "HifiGan":
        cmd = f"""
        python3 /home/aistudio/PaddleSpeech/paddlespeech/t2s/exps/fastspeech2/../synthesize_e2e.py \
                        --am=fastspeech2_mix \
                        --am_config=models/fastspeech2_mix_ckpt_1.2.0/default.yaml \
                        --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                        --am_stat=models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
                        --voc="hifigan_aishell3" \
                        --voc_config=models/hifigan_aishell3_ckpt_0.2.0/default.yaml \
                        --voc_ckpt=models/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz \
                        --voc_stat=models/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy \
                        --lang=mix \
                        --text={text_file} \
                        --output_dir={wav_output_dir} \
                        --phones_dict={dump_dir}/phone_id_map.txt \
                        --speaker_dict={dump_dir}/speaker_id_map.txt \
                        --spk_id=0 \
                        --ngpu=1
        """
    else:
        print("声码器不符合要求，请重新选择")
        status_change(status_json, False)
        return False
    
    if not run_cmd(cmd, cwd_path, message="Step9 生成音频 执行失败，请执行命令行"):
        status_change(status_json, False)
        return False
    else:
        status_change(status_json, False)
        return True


# 使用静态图生成
def step9_generateTTS_inference(text_dict, exp_name, voc="PWGan"):
    exp_path = exp_base + exp_name
    wav_output_dir = os.path.join(exp_path, "wav_out")
    am_inference_dir = os.path.join(inference_dir, exp_name)
    
    device = "gpu"
    
    status_json = os.path.join(exp_path, "generate_status.json")
    # 写一个物理锁
    status_change(status_json, True)

    # frontend
    frontend = get_frontend(
        lang="mix",
        phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
        tones_dict=None
    )

    # am_predictor
    am_predictor = get_predictor(
        model_dir=am_inference_dir,
        model_file="fastspeech2_mix" + ".pdmodel",
        params_file="fastspeech2_mix" + ".pdiparams",
        device=device)
    
    # voc_predictor
    if voc == "PWGan":
        voc_predictor = get_predictor(
            model_dir=pwg_inference_dir,
            model_file="pwgan_aishell3" + ".pdmodel",
            params_file="pwgan_aishell3" + ".pdiparams",
            device=device)
    elif voc == "WaveRnn":
        voc_predictor = get_predictor(
            model_dir=wavernn_inference_dir,
            model_file="wavernn_csmsc" + ".pdmodel",
            params_file="wavernn_csmsc" + ".pdiparams",
            device=device)
    elif voc == "HifiGan":
        voc_predictor = get_predictor(
            model_dir=hifigan_inference_dir,
            model_file="hifigan_aishell3" + ".pdmodel",
            params_file="hifigan_aishell3" + ".pdiparams",
            device=device)

    output_dir = Path(wav_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = list(text_dict.items())

    merge_sentences = True
    fs = 24000
    for utt_id, sentence in sentences:
        am_output_data = get_am_output(
            input=sentence,
            am_predictor=am_predictor,
            am="fastspeech2_mix",
            frontend=frontend,
            lang="mix",
            merge_sentences=merge_sentences,
            speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
            spk_id=0, )
        wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)
        # 保存文件
        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)
    
    status_change(status_json, False)
    return True

def freeze_layer(model, layers: List[str]):
    """freeze layers

    Args:
        layers (List[str]): frozen layers
    """
    for layer in layers:
        for param in eval("model." + layer + ".parameters()"):
            param.trainable = False
# 前端
def get_tts_phone_ids(sentence, exp_name, exp_fun=False):
    am_inference_dir = os.path.join(inference_dir, exp_name)
    phones_dict = f"{am_inference_dir}/phone_id_map.txt"
    # 检测有没有SSML标记
    if sentence.strip() != "" and re.match(r".*?<speak>.*?</speak>.*", sentence,
                                           re.DOTALL):
        # 有SSML标记，只支持中文修改多音字
        lang = "zh"
        merge_sentences = False
        frontend = get_frontend(lang=lang, phones_dict=phones_dict, tones_dict=None)
    else:
        # 没有 SSML 标记，按照中英双语处理
        lang = "mix"
        frontend = get_frontend(lang=lang, phones_dict=phones_dict, tones_dict=None)
        if exp_fun:
            merge_sentences = True
        else:
            merge_sentences = False
    print(f"merge_sentences: {merge_sentences}")
    frontend_dict = run_frontend(
                    frontend=frontend,
                    text=sentence,
                    merge_sentences=merge_sentences,
                    get_tone_ids=False,
                    lang=lang)
    phone_ids = frontend_dict['phone_ids']
    return phone_ids

def get_idx2ph_dict(exp_name):
    am_inference_dir = os.path.join(inference_dir, exp_name)
    phones_dict = f"{am_inference_dir}/phone_id_map.txt"
    id2phs_dict = {}

    with open(phones_dict, "r", encoding="utf8") as f:
        for line in f.readlines():
            if line.strip():
                ph, idx = line.strip().split()
                id2phs_dict[idx] = ph
    return id2phs_dict

def fs2_inference(model, text, spk_id, alpha=1.0, duration=None, return_duration=False):
    x = paddle.cast(text, 'int64')
    xs = x.unsqueeze(0)
    ilens = paddle.shape(x)[0]
    # forward encoder
    x_masks = model._source_mask(ilens)
    # (B, Tmax, adim)
    hs, _ = model.encoder(xs, x_masks)
    # spk_id 添加进去了
    spk_emb = model.spk_embedding_table(spk_id)
    hs = model._integrate_with_spk_embed(hs, spk_emb)
    # forward duration predictor and variance predictors
    d_masks = make_pad_mask(ilens)

    # True
    if model.stop_gradient_from_pitch_predictor:
        p_outs = model.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
    else:
        p_outs = model.pitch_predictor(hs, d_masks.unsqueeze(-1))
    
    # False
    if model.stop_gradient_from_energy_predictor:
        e_outs = model.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
    else:
        e_outs = model.energy_predictor(hs, d_masks.unsqueeze(-1))

    # duration
    # 输出已经是整数(float32)了
    if duration is None:
        d_outs = model.duration_predictor.inference(hs, d_masks)
    else:
        d_outs = duration
    
    if return_duration:
        return d_outs

    # print(d_outs)
    
    # 往下添加 pitch, energy 信息
    p_embs = model.pitch_embed(p_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
    e_embs = model.energy_embed(e_outs.transpose((0, 2, 1))).transpose(
        (0, 2, 1))
    hs = hs + e_embs + p_embs

    # (B, Lmax, adim)
    hs = model.length_regulator(hs, d_outs, alpha, is_inference=True)

    # decoder
    h_masks = None
    zs, _ = model.decoder(hs, h_masks)
    # (B, Lmax, odim)
    before_outs = model.feat_out(zs).reshape(
        (paddle.shape(zs)[0], -1, model.odim))
    after_outs = before_outs + model.postnet(
                before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
    return before_outs, after_outs, d_outs, p_outs, e_outs

def duration_phones_to_list(duration, phone_ids, id2phs_dict):
    phs_lists = []
    phs_lens = phone_ids[0].shape[0]
    for i in range(phs_lens):
        ph = id2phs_dict[str(phone_ids[0].numpy()[i])]
        dur = int(duration[0].numpy()[i])
        phs_lists.append({
            "ph": ph,
            "dur": dur
        })
        # print(f"{i}\t{ph}\t{dur}")
    return phs_lists

def list_to_durations(phs_lists):
    duration = []
    for k in phs_lists:
        dur = k['dur']
        duration.append(int(dur))
    duration = paddle.to_tensor([duration], dtype=paddle.float32)
    return duration


def load_fs2_model(exp_name):
    am_inference_dir = os.path.join(inference_dir, exp_name)

    ckpt = find_max_ckpt(am_inference_dir)
    if ckpt == 0 or ckpt is None:
        print("ckpt 模型不存在，请将模型放入inference路径")
        return None
    model_path = f"{am_inference_dir}/snapshot_iter_{ckpt}.pdz"
    speaker_dict = f"{am_inference_dir}/speaker_id_map.txt"
    phones_dict = f"{am_inference_dir}/phone_id_map.txt"

    # 加载fs2模型
    default_config_file = f"{am_inference_dir}/default.yaml"
    with open(default_config_file) as f:
        config = CfgNode(yaml.safe_load(f))

    with open(phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    odim = config.n_mels

    with open(speaker_dict, 'rt') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    spk_num = len(spk_id)
    model = FastSpeech2(
        idim=vocab_size,
        odim=odim, 
        spk_num=spk_num, 
        **config["model"])

    archive = paddle.load(model_path)
    model.set_state_dict(archive['main_params'])
    model.eval()
    return model

def load_voc_model(voc):
    device = "gpu"
    # voc_predictor
    if voc == "PWGan":
        voc_predictor = get_predictor(
            model_dir=pwg_inference_dir,
            model_file="pwgan_aishell3" + ".pdmodel",
            params_file="pwgan_aishell3" + ".pdiparams",
            device=device)
    elif voc == "WaveRnn":
        voc_predictor = get_predictor(
            model_dir=wavernn_inference_dir,
            model_file="wavernn_csmsc" + ".pdmodel",
            params_file="wavernn_csmsc" + ".pdiparams",
            device=device)
    elif voc == "HifiGan":
        voc_predictor = get_predictor(
            model_dir=hifigan_inference_dir,
            model_file="hifigan_aishell3" + ".pdmodel",
            params_file="hifigan_aishell3" + ".pdiparams",
            device=device)
    else:
        return None
    
    return voc_predictor

def generateTTS_inference_adjust_duration(text_dict, exp_name, voc="PWGan", alpha=1.0, duration=None, merge_sentences=False, exp_fun=False):
    exp_path = exp_base + exp_name
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)

    wav_output_dir = os.path.join(exp_path, "wav_out")
    if not os.path.exists(wav_output_dir):
        os.makedirs(wav_output_dir, exist_ok=True)
    am_inference_dir = os.path.join(inference_dir, exp_name)
    phones_dict = f"{am_inference_dir}/phone_id_map.txt"
    status_json = os.path.join(exp_path, "generate_status.json")
    
    # 写一个物理锁
    status_change(status_json, True)
    
    # 前端
    # frontend = get_frontend(
    #             lang="mix", phones_dict=phones_dict, tones_dict=None)
    
    # 加载fs2模型
    model = load_fs2_model(exp_name)
    if not model:
        status_change(status_json, False)
        print("微调模型加载失败！")
        return False

    fs = 24000

    # 加载声码器
    # voc_predictor
    voc_predictor = load_voc_model(voc)
    if not voc:
        status_change(status_json, False)
        print("声码器加载失败！")
        return False

    am_stat = os.path.join(am_inference_dir, "speech_stats.npy")
    am_mu, am_std = np.load(am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    
    
    for utt_id, sentence in text_dict.items():
        phone_ids = get_tts_phone_ids(sentence, exp_name, exp_fun=exp_fun)
        wavs = []
        for phone_id in phone_ids:
            spk_id = paddle.to_tensor(0)
            before_outs, after_outs, d_outs, p_outs, e_outs = fs2_inference(model, text=phone_id, alpha=alpha, spk_id=spk_id,duration=duration)
            normalized_mel = after_outs[0]
            logmel = am_normalizer.inverse(normalized_mel)
            am_output_data = logmel.numpy()
            # 给到 vocoder 进行推理
            wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)
            wavs.append(wav)
        wavs = np.concatenate(wavs)
        # 保存文件
        sf.write(f"{wav_output_dir}/{utt_id}.wav", wavs, samplerate=fs)
    
    status_change(status_json, False)
    return True
    

# 全流程微调训练
def finetuneTTS(label_path, exp_name, start_step=1, end_step=100, max_finetune_step=100, batch_size=None, learning_rate=None):
    exp_path = os.path.join(exp_base + exp_name)
    os.makedirs(exp_path, exist_ok=True)
    data_dir = os.path.join(exp_path, "data")
    finetune_status_json = os.path.join(exp_path, "finetune_status.json")
    status_change(finetune_status_json, True)
    
    if start_step <= 1 and end_step >= 1:
        # Step1 : 生成标准数据集
        if not os.path.exists(data_dir):
            exp_data_path, error_code = step1_generate_standard_dataset(label_path, exp_path, data_dir)
            if exp_data_path is None:
                status_change(finetune_status_json, False)
                # 步骤一出错
                return 
        else:
            print(f"{data_dir} 已存在，跳过此步骤！")
    
    # Step2: 检查非法数据
    new_dir = os.path.join(exp_path, "new_dir")
    if start_step <= 2 and end_step >= 2:  
        if not os.path.exists(new_dir):
            if not step2_check_oov(data_dir, new_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{new_dir} 已存在，跳过此步骤")
    
    mfa_dir = os.path.join(exp_path, "mfa")
    if start_step <= 3 and end_step >= 3:
        # Step3: MFA 对齐
        if not os.path.exists(mfa_dir):
            if not step3_get_mfa(new_dir, mfa_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{mfa_dir} 已存在，跳过此步骤")
    
    if start_step <= 4 and end_step >= 4:
        # Step4: 生成时长信息文件
        if not step4_duration(mfa_dir):
            status_change(finetune_status_json, False)
            return
    
    dump_dir = os.path.join(exp_path, "dump")
    if start_step <= 5 and end_step >= 5:
        # Step5: 数据预处理
        if not os.path.exists(dump_dir):
            if not step5_extract_feature(data_dir, dump_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{dump_dir} 已存在，跳过此步骤")
    
    output_dir = os.path.join(exp_path, "output")
    if start_step <= 6 and end_step >= 6:
        # Step6: 生成训练环境
        if not os.path.exists(output_dir):
            if not step6_prepare_env(output_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{output_dir} 已存在，跳过此步骤")
    
    if start_step <= 7 and end_step >= 7:
        # Step7: 微调训练
        if not step7_finetune(dump_dir, output_dir, max_finetune_step, batch_size=None, learning_rate=None):
            status_change(finetune_status_json, False)
            return
    
    if start_step <= 8 and end_step >= 8:
        # 这一步不需要运行中
        status_change(finetune_status_json, False)
        # Step8: 导出静态图模型
        if not step8_export_static_model(exp_name):
            return
    
    status_change(finetune_status_json, False)
    return True

# 生成音频
def generateTTS(text_dict, exp_name, voc="PWGan"):
    if not step9_generateTTS_inference(text_dict, exp_name, voc):
        print("音频生成失败，请微调模型是否成功!")
        return None
    else:
        return True
