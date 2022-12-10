# PSTTS-local
部署本地的paddlespeech-tts，支持设置微调后的模型以及并发数

首先将静态模型和你微调后的模型放入 ./inference 下

然后安装环境

首先你需要安装paddle https://www.paddlepaddle.org.cn/

然后运行
```
pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装其它环境

如图为文件路径
![image](https://user-images.githubusercontent.com/54951765/206846134-3c639e7c-0277-4e4c-ba00-a69e576978ca.png)

## 本地生成文件
修改 `main1.py` 中的内容 `exp_name` 为你的微调后导出的文件夹名， `text_dict` 为你要生成的文本内容，修改完成后在终端中输入 `python main1.py` 运行即可

## 通过web请求生成并下载音频文件

修改 `get_models.py` 文件中的内容 `exp_name` 为你的微调后导出的文件夹名，`voc` 为要使用的声码器（声码器可选 【PWGan】【WaveRnn】【HifiGan】），`num` 为进程池大小，每个进程要消耗1.5g左右的内存。

修改完成在终端中输入以下命令启动服务器
```
uvicorn main:app --reload --port 25566
```

然后你可以通过访问本机的127.0.0.1:25566来生成并下载音频，示例如下：
```
http://127.0.0.1:25566/tts/?sentence=你好世界&alpha=1.2
```

## 其他

默认音频文件生成位置：`work/exp_test/wav_out`

## todo
声学模型不需要并发，之后会优化掉，内存占用也会小很多
