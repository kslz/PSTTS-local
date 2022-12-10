# -*- coding: utf-8 -*-
"""
    @Time : 2022/12/10 12:39
    @Author : 李子
    @Url : https://github.com/kslz
"""

import os
import random
import time

import soundfile as sf
import numpy as np
import paddle

from paddlespeech.t2s.exps.syn_utils import get_voc_output

from util.finetuneTTS import load_fs2_model, load_voc_model, get_tts_phone_ids, fs2_inference
from paddlespeech.t2s.modules.normalizer import ZScore

exp_base = "./work/exp_"
exp_name = "xthnr2"
exp_path = exp_base + exp_name
inference_dir = "./inference"
voc = "PWGan"
wav_output_dir = os.path.join(exp_path, "wav_out")

# 进程池大小，越大则并发越高，希望你有个好配置
pnum = 2

def init_dir():
    if not os.path.exists(exp_path):
        os.makedirs(exp_path, exist_ok=True)
    if not os.path.exists(wav_output_dir):
        os.makedirs(wav_output_dir, exist_ok=True)
    print("初始化路径完毕")


def init_models():
    am_inference_dir = os.path.join(inference_dir, exp_name)
    phones_dict = f"{am_inference_dir}/phone_id_map.txt"
    # status_json = os.path.join(exp_path, "generate_status.json")

    # 写一个物理锁
    # status_change(status_json, True)

    # 前端
    # frontend = get_frontend(
    #             lang="mix", phones_dict=phones_dict, tones_dict=None)

    # 加载fs2模型
    model = load_fs2_model(exp_name)
    if not model:
        # status_change(status_json, False)
        print("微调模型加载失败！")
        return False

    # 加载声码器
    # voc_predictor
    voc_predictor = load_voc_model(voc)
    if not voc:
        # status_change(status_json, False)
        print("声码器加载失败！")
        return False

    am_stat = os.path.join(am_inference_dir, "speech_stats.npy")
    am_mu, am_std = np.load(am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    return model, voc_predictor, am_normalizer


def tts_by_fs2(model, am_normalizer, voc_predictor, sentence, alpha=1.0):
    phone_ids = get_tts_phone_ids(sentence, exp_name, exp_fun=False)
    wavs = []
    for phone_id in phone_ids:
        spk_id = paddle.to_tensor(0)
        before_outs, after_outs, d_outs, p_outs, e_outs = fs2_inference(model, text=phone_id, alpha=alpha,
                                                                        spk_id=spk_id, duration=None)
        normalized_mel = after_outs[0]
        logmel = am_normalizer.inverse(normalized_mel)
        am_output_data = logmel.numpy()
        # 给到 vocoder 进行推理
        wav = get_voc_output(
            voc_predictor=voc_predictor, input=am_output_data)
        wavs.append(wav)
    wavs = np.concatenate(wavs)
    # 保存文件
    fs = 24000
    filename = make_filename()
    sf.write(f"{wav_output_dir}/{filename}.wav", wavs, samplerate=fs)
    return os.path.join(wav_output_dir, filename + ".wav")


def make_filename():
    random_str = list("abcdefgh")
    random.shuffle(random_str)
    filename = f"{time.time_ns()}_{''.join(random_str)}"
    return filename