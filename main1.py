from util.finetuneTTS import generateTTS_inference_adjust_duration

exp_name = "test"

# 格式
# "文件名": "需要合成的文本"
# 新增 SSML 多音字处理（句子只支持中文)
# SSML使用说明：https://github.com/PaddlePaddle/PaddleSpeech/discussions/2538
text_dict = {
    "3": "你好世界。",
}

# 生成想要合成的声音
# 声码器可选 【PWGan】【WaveRnn】【HifiGan】
# 动态图推理，可调节速度， alpha 为 float, 表示语速系数，可以按需求精细调整
# alpha 越小，速度越快，alpha 越大，速度越慢, eg, alpha = 0.5, 速度变为原来的一倍， alpha = 2.0， 速度变为一半
generateTTS_inference_adjust_duration(text_dict, exp_name, voc="PWGan", alpha=1.1)
