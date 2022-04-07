import logging
import os
import random
import time
from logging.handlers import TimedRotatingFileHandler

import numpy as np
from paddle.io import DataLoader

from paddlespeech.s2t.exps.deepspeech2.config import get_cfg_defaults
from paddlespeech.s2t.frontend.utility import read_manifest
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model

speech_save_dir = 'data/demo_cache'  # Directory to save demo audios.
warmup_manifest = 'examples/aishell/s0/data/manifest.dev'  # Filepath of manifest to warm up.
config_file = 'conf/deepspeech2.yaml'
checkpoint_path = 'exp/deepspeech2/checkpoints/avg_1'
config = get_cfg_defaults()
config.merge_from_file(config_file)
LOG_PATH = os.getcwd() + '/log/asr'
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log = logging.getLogger()
log.setLevel(logging.INFO)
log.suffix = "%Y-%m-%d_%H-%M.log"
log.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
log_file_handler = TimedRotatingFileHandler(filename=LOG_PATH, when="D", interval=1, backupCount=7)
log_file_handler.setFormatter(formatter)

log.addHandler(log_file_handler)
import paddle

config.data.manifest = warmup_manifest
config.freeze()
# print(config)
config.defrost()
dataset = ManifestDataset.from_config(config)
config.collator.augmentation_config = ""
config.collator.keep_transcription_text = True
config.collator.batch_size = 1
config.collator.num_workers = 0
collate_fn = SpeechCollator.from_config(config)
test_loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)
model = DeepSpeech2Model.from_pretrained(test_loader, config, checkpoint_path)
model.eval()


# prepare ASR inference handler
def file_to_transcript(filename):
    feature = test_loader.collate_fn.process_utterance(filename, "")
    audio = np.array([feature[0]]).astype('float32')  # [1, T, D]
    # audio = audio.swapaxes(1,2)
    log.info('---file_to_transcript feature----')
    # print(audio.shape)
    audio_len = feature[0].shape[0]
    # print(audio_len)
    audio_len = np.array([audio_len]).astype('int64')  # [1]

    result_transcript = model.decode(
        paddle.to_tensor(audio),
        paddle.to_tensor(audio_len),
        vocab_list=test_loader.collate_fn.vocab_list,
        decoding_method=config.decoding.decoding_method,
        lang_model_path=config.decoding.lang_model_path,
        beam_alpha=config.decoding.alpha,
        beam_beta=config.decoding.beta,
        beam_size=config.decoding.beam_size,
        cutoff_prob=config.decoding.cutoff_prob,
        cutoff_top_n=config.decoding.cutoff_top_n,
        num_processes=config.decoding.num_proc_bsearch)
    return result_transcript[0]


# warm up model
def warm_up_test(audio_process_handler,
                 manifest_path,
                 num_test_cases,
                 random_seed=0):
    """Warming-up test."""
    manifest = read_manifest(manifest_path)
    rng = random.Random(random_seed)
    samples = rng.sample(manifest, num_test_cases)
    for idx, sample in enumerate(samples):
        log.info("Warm-up Test Case %d: %s" % (idx, sample['feat']))
        start_time = time.time()
        transcript = audio_process_handler(sample['feat'])
        finish_time = time.time()
        log.info("Response Time: %f, Transcript: %s" %
                 (finish_time - start_time, transcript))
        # print("Response Time: %f, Transcript: %s" %
        # (finish_time - start_time, transcript))


# # warming up with utterrances sampled from Librispeech
log.info('Warming up ...')
warm_up_test(
    audio_process_handler=file_to_transcript,
    manifest_path=warmup_manifest,
    num_test_cases=3)
log.info("warm up finish. waiting for message")


def handle_uploaded_file(file, model_config, speech_save_dir, test_loader, checkpoint_path, audio_process_handler):
    # model = DeepSpeech2Model.from_pretrained(test_loader, model_config, checkpoint_path)
    # model.eval()
    if not os.path.exists(speech_save_dir):
        os.mkdir(speech_save_dir)

    start_time = time.time()
    transcript = audio_process_handler(file)
    finish_time = time.time()
    log.info("Response Time: %f, Transcript: %s" %
             (finish_time - start_time, transcript))
    return transcript.encode('utf-8')


def clean_sound_cache():
    path = 'data/demo_cache/'
    for i in os.listdir(path):
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


def clean_log():
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
