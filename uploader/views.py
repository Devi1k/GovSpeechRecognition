import wave
from time import gmtime
from time import strftime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from paddlespeech.server.bin.paddlespeech_client import ASRClientExecutor
from .log import *
from .utils import *

asrclient_executor = ASRClientExecutor()
log = Logger().get_logger()
res = asrclient_executor(
    input='data/demo_cache/demo.wav',
    server_ip="127.0.0.1",
    port=8090,
    sample_rate=48000,
    lang="zh_cn",
    audio_format="wav").json()['result']['transcription']
log.info('warm up:' + res)
log.info('wait for message')


@csrf_exempt
def upload_file(request):
    clean_log(log)
    clean_sound_cache()
    log.info('-----------------------------------------------------------')

    body_content = request.body
    # print(body_content)

    if request.method == 'POST':
        # print(body_content)
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        out_filename = os.path.join(
            speech_save_dir,
            timestamp + ".wav")
        # export file
        with open(out_filename, 'wb+') as destination:
            destination.write(body_content[44:])
        # 137 : 182
        wav_file = body_content[137:]
        file = wave.open(out_filename, 'wb')
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(48000)
        file.writeframes(wav_file)
        file.close()

        log.info("Saved to %s." % out_filename)

        # start_time = time.time()
        # context = handle_uploaded_file(file=out_filename, model_config=config, speech_save_dir=speech_save_dir,
        #                                test_loader=test_loader,
        #                                checkpoint_path=checkpoint_path,
        #                                audio_process_handler=file_to_transcript)
        # end_time = time.time()
        # log.info(end_time - start_time)
        start = time.time()
        res = asrclient_executor(
            input=out_filename,
            server_ip="127.0.0.1",
            port=8090,
            sample_rate=48000,
            lang="zh_cn",
            audio_format="wav").json()['result']['transcription']
        end = time.time()
        log.info('trans:{},cost {} seconds'.format(res, str(end - start)))
        print(type(res))
        # context = str(res, encoding="utf-8")
        return JsonResponse({'message': 'success', 'data': res, 'code': 0})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
