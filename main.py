# -*- coding: utf-8 -*-
"""
    @Time : 2022/12/10 8:37
    @Author : 李子
    @Url : https://github.com/kslz
"""
import os

from typing import Union

from fastapi import FastAPI
from starlette.responses import FileResponse

from util.get_models import init_dir, pnum
from util.process_test import TTSPoll

app = FastAPI()

tts_poll = None



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/tts/")
def tts(sentence: str, alpha: float = 1.0):
    global tts_poll
    file_path = tts_poll.tts(sentence, alpha)
    return FileResponse(
        file_path,  # 这里的文件名是你要发送的文件路径
        filename=os.path.basename(file_path),  # 这里的文件名是你要给用户展示的下载的文件名，比如我这里叫lol.exe
    )


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


def main():
    init_dir()
    global tts_poll
    tts_poll = TTSPoll(pnum)

    pass



main()