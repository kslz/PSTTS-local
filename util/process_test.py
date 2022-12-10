# -*- coding: utf-8 -*-
"""
    @Time : 2022/12/10 12:32
    @Author : 李子
    @Url : https://github.com/kslz
"""
import threading
import time
from multiprocessing import Process, Pipe

from util.get_models import init_models, tts_by_fs2


class MyProcess(Process):
    def __init__(self, name, subconn):
        Process.__init__(self, name=name)
        self.am_normalizerinit_models = None
        self.voc_predictor = None
        self.model = None
        self.subconn = subconn
        self.is_ok = False

    def run(self):
        self.model, self.voc_predictor, self.am_normalizerinit_models = init_models()
        print(f"进程 {self.name} 初始化完成。")
        self.subconn.send((True,))
        while True:
            rec = self.subconn.recv()
            print(f"进程 {self.name} 开始合成：{rec[0]} 语速为：{rec[1]}")
            self.subconn.send((False,))
            file_path = tts_by_fs2(self.model, self.am_normalizerinit_models, self.voc_predictor, rec[0], rec[1])
            self.subconn.send((True, file_path))


    #
    # def tts_by_fs2_process(self, sentence, alpha):
    #     tts_by_fs2(self.model, self.am_normalizerinit_models, self.voc_predictor, sentence, alpha)


# pipe_list_g = []
# process_list_g = []


class TTSPoll():
    def __init__(self, pnum):
        self.pnum = pnum
        self.pipe_list = []
        for i in range(pnum):
            self.pipe_list.append(Pipe())
        # global pipe_list_g
        # pipe_list_g = self.pipe_list

        self.process_list = []
        for i in range(pnum):
            p = MyProcess(str(i), self.pipe_list[i][1])
            p.daemon = True
            p.start()
            self.process_list.append([p, False])

        # global process_list_g
        # process_list_g = self.pipe_list

    def update_list(self):
        for i in range(self.pnum):
            pipe = self.pipe_list[i]
            while pipe[0].poll():
                state = pipe[0].recv()[0]
                self.process_list[i][1] = state


    def tts(self, sentence, alpha):
        while True:
            self.update_list()
            index = 0
            print(self.process_list, sentence, alpha)
            for i in self.process_list:
                if i[1]:
                    con = self.pipe_list[index]
                    con[0].send((sentence, alpha))
                    self.process_list[index][1] = con[0].recv()[0]  # 合成中
                    result = con[0].recv()
                    self.process_list[index][1] = result[0]
                    return result[1]
                index += 1
            print("并发已满，排队中")
            time.sleep(1)

# class UpdateThread(threading.Thread):
#     def __init__(self,i,pipe):
#         threading.Thread.__init__(self)
#         self.name = i
#         self.pipe = pipe
#
#     def run(self):
#         pipe
