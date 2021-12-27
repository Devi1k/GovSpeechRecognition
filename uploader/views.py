import array
import json
import os
import random
import struct
import time
import wave
from time import gmtime
from time import strftime

import numpy
import scipy
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django import forms
import functools

import numpy as np
import paddle

from scipy.io import wavfile
from paddle.io import DataLoader

from paddlespeech.s2t.exps.deepspeech2.config import get_cfg_defaults
from paddlespeech.s2t.io.collator import SpeechCollator
from paddlespeech.s2t.io.dataset import ManifestDataset
from paddlespeech.s2t.models.ds2 import DeepSpeech2Model
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.utility import add_arguments
from paddlespeech.s2t.utils.utility import print_arguments
from paddlespeech.s2t.frontend.utility import read_manifest

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
    print('---file_to_transcript feature----')
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
        print("Warm-up Test Case %d: %s" % (idx, sample['feat']))
        start_time = time.time()
        transcript = audio_process_handler(sample['feat'])
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))


# # warming up with utterrances sampled from Librispeech
print('Warming up ...')
warm_up_test(
    audio_process_handler=file_to_transcript,
    manifest_path=warmup_manifest,
    num_test_cases=3)


def handle_uploaded_file(file, model_config, speech_save_dir, test_loader, checkpoint_path, audio_process_handler):
    # model = DeepSpeech2Model.from_pretrained(test_loader, model_config, checkpoint_path)
    # model.eval()
    if not os.path.exists(speech_save_dir):
        os.mkdir(speech_save_dir)
    # timestamp = strftime("%Y%m%d%H%M%S", gmtime())
    # out_filename = os.path.join(
    #     speech_save_dir,
    #     timestamp + "_" + ".wav")
    # write to wav file
    # print('write successfully')
    # with open(out_filename, 'wb+') as destination:
    #     for chunk in file.chunks():
    #         destination.write(chunk)
    start_time = time.time()
    transcript = audio_process_handler(file)
    finish_time = time.time()
    print("Response Time: %f, Transcript: %s" %
          (finish_time - start_time, transcript))
    return transcript.encode('utf-8')


@csrf_exempt
def upload_file(request):
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
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

        print('-------------body-----------------')
        # print(body_content[:138])
        with open(out_filename, 'wb+') as destination:
            destination.write(body_content[44:])
        # 138 : 182
        # target_len = struct.unpack('<i', body_content[154:158])[0]
        wav_file = body_content[137:]
        # print(wav_file[0:45])

        # print(struct.unpack('h', wav_file[0:2])[0])
        print(struct.unpack('h', wav_file[22:24])[0])
        print(struct.unpack('<i', wav_file[24:28])[0])
        # print(struct.unpack('<i', wav_file[40:44])[0])
        file = wave.open(out_filename, 'wb')
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(48000)
        file.writeframes(wav_file)
        file.close()
        print('write successfully')

        print("Saved to %s." % out_filename)

        print('----------------file-----------------------')
        # print('file ' + f)
        if out_filename:
            print('--------------success get file---------------')
            start_time = time.time()
            context = handle_uploaded_file(file=out_filename, model_config=config, speech_save_dir=speech_save_dir,
                                           test_loader=test_loader,
                                           checkpoint_path=checkpoint_path,
                                           audio_process_handler=file_to_transcript)
            end_time = time.time()
            print(end_time - start_time)
            context = str(context, encoding="utf-8")
            return JsonResponse({'message': 'success', 'data': context, 'code': 0})
        return JsonResponse({'message': 'unknown file', 'code': 50013})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})


class UploadFileForm(forms.Form):
    # title = forms.CharField(max_length=50)
    file = forms.FileField()
