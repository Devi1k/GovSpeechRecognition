import os
import random
import re
import struct
import time
import wave
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler
from time import gmtime
from time import strftime

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms

import numpy as np
import paddle

from paddle.io import DataLoader

from paddlespeech.s2t.exps.deepspeech2.config import get_cfg_defaults
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.frontend.utility import read_manifest

LOG_PATH = '../log'
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler = TimedRotatingFileHandler(filename=LOG_PATH, when="D", interval=1, backupCount=7)
log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
log_file_handler.setFormatter(formatter)
log_file_handler.setLevel(logging.DEBUG)
log = logging.getLogger()
log.addHandler(log_file_handler)

speech_save_dir = 'data/demo_cache'  # Directory to save demo audios.
warmup_manifest = 'data/manifest.dev'  # Filepath of manifest to warm up.
config_file = 'conf/deepspeech2.yaml'
checkpoint_path = 'exp/deepspeech2/checkpoints/avg_1'
config = get_cfg_defaults()
config.merge_from_file(config_file)
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
    log.debug('---file_to_transcript feature----')
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
        log.debug("Warm-up Test Case %d: %s" % (idx, sample['feat']))
        start_time = time.time()
        transcript = audio_process_handler(sample['feat'])
        finish_time = time.time()
        log.debug("Response Time: %f, Transcript: %s" %
                  (finish_time - start_time, transcript))


# # warming up with utterrances sampled from Librispeech
log.debug('Warming up ...')
warm_up_test(
    audio_process_handler=file_to_transcript,
    manifest_path=warmup_manifest,
    num_test_cases=3)


def handle_uploaded_file(file, model_config, speech_save_dir, test_loader, checkpoint_path, audio_process_handler):
    # model = DeepSpeech2Model.from_pretrained(test_loader, model_config, checkpoint_path)
    # model.eval()
    if not os.path.exists(speech_save_dir):
        os.mkdir(speech_save_dir)

    start_time = time.time()
    transcript = audio_process_handler(file)
    finish_time = time.time()
    log.debug("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))
    return transcript.encode('utf-8')


def clean_cache():
    path = '../data/demo_cache/'
    for i in os.listdir(path):
        file_path = path + i  # 生成日志文件的路径
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        # 获取日志的年月，和今天的年月
        today_m = int(timestamp[4:5])  # 今天的月份
        m = int(i[4:5])  # 日志的月份
        today_y = int(timestamp[0:4])  # 今天的年份
        y = int(i[0:4])  # 日志的年份

        # 对上个月的日志进行清理，即删除。
        if m < today_m:
            if os.path.exists(file_path):  # 判断生成的路径对不对，防止报错
                os.remove(file_path)  # 删除文件
        elif y < today_y:
            if os.path.exists(file_path):
                os.remove(file_path)


@csrf_exempt
def upload_file(request):
    clean_cache()
    log.debug('-----------------------------------------------------------')
    log.debug('-----------------------------------------------------------')

    body_content = request.body
    # print(body_content)

    if request.method == 'POST':
        # print(request.FILES)
        # print(wav)
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        out_filename = os.path.join(
            speech_save_dir,
            timestamp + ".wav")
        # blob = request.FILES['audio_data']
        # audio = wave.open(out_filename, 'wb')
        # audio.setnchannels(1)
        # audio.setsampwidth(1)
        # audio.setframerate(8000)
        # audio.setnframes(1)
        # audio.writeframes(blob.read())
        # audio.close()

        # print(body_content[:138])
        with open(out_filename, 'wb+') as destination:
            destination.write(body_content[44:])
        # 138 : 182
        # target_len = struct.unpack('<i', body_content[154:158])[0]
        wav_file = body_content[137:]
        # print(wav_file[0:45])

        # print(struct.unpack('h', wav_file[0:2])[0])
        # print(struct.unpack('<i', wav_file[40:44])[0])
        file = wave.open(out_filename, 'wb')
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(48000)
        file.writeframes(wav_file)
        file.close()

        log.debug("Saved to %s." % out_filename)

        # print('file ' + f)
        if out_filename:
            start_time = time.time()
            context = handle_uploaded_file(file=out_filename, model_config=config, speech_save_dir=speech_save_dir,
                                           test_loader=test_loader,
                                           checkpoint_path=checkpoint_path,
                                           audio_process_handler=file_to_transcript)
            end_time = time.time()
            # log.debug(end_time - start_time)
            context = str(context, encoding="utf-8")
            return JsonResponse({'message': 'success', 'data': context, 'code': 0})
        return JsonResponse({'message': 'unknown file', 'code': 50013})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})


class UploadFileForm(forms.Form):
    # title = forms.CharField(max_length=50)
    file = forms.FileField()
