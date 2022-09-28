import logging
import os
import re
import time
from logging.handlers import TimedRotatingFileHandler


class Logger():
    def __init__(self):
        self.LOG_PATH = os.getcwd() + '/log/asr'
        self.log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
        self.formatter = logging.Formatter(self.log_fmt)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.log.suffix = "%Y-%m-%d_%H-%M.log"
        self.log.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
        log_file_handler = TimedRotatingFileHandler(filename=self.LOG_PATH, when="D", interval=1, backupCount=7)
        log_file_handler.setFormatter(self.formatter)
        self.log.addHandler(log_file_handler)

    def get_logger(self):
        return self.log


def clean_sound_cache():
    path = 'data/demo_cache/'
    for i in os.listdir(path):
        # print(i)
        if len(i) < 10:
            continue
        file_path = path + i  # 生成日志文件的路径
        timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        # 获取日志的年月，和今天的年月
        today_m = int(timestamp[4:6])  # 今天的月份
        m = int(i[4:6])  # 日志的月份
        today_y = int(timestamp[0:4])  # 今天的年份
        y = int(i[0:4])  # 日志的年份

        # 对上个月的日志进行清理，即删除。
        if m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)


def clean_log(log):
    path = 'log/'
    path2 = 'exp/log/'
    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    today_m = int(timestamp[4:6])  # 今天的月份
    today_y = int(timestamp[0:4])  # 今天的年份
    log.info('clean web log')
    for i in os.listdir(path):
        if len(i) < 4:
            continue
        file_path = path + i  # 生成日志文件的路径
        file_m = int(i[9:11])  # 日志的月份
        file_y = int(i[4:8])  # 日志的年份
        # 对上个月的日志进行清理，即删除。
        # print(file_path)
        if file_m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif file_y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)
    log.info('clean speech log')
    for i in os.listdir(path2):
        file_path = path2 + i  # 生成日志文件的路径
        file_m = int(i[32:34])  # 日志的月份
        file_y = int(i[27:31])  # 日志的年份
        if file_m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif file_y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)
