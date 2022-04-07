import wave
from time import gmtime
from time import strftime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .utils import *


@csrf_exempt
def upload_file(request):
    clean_log()
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
        context = handle_uploaded_file(file=out_filename, model_config=config, speech_save_dir=speech_save_dir,
                                       test_loader=test_loader,
                                       checkpoint_path=checkpoint_path,
                                       audio_process_handler=file_to_transcript)
        # end_time = time.time()
        # log.info(end_time - start_time)
        context = str(context, encoding="utf-8")
        return JsonResponse({'message': 'success', 'data': context, 'code': 0})
    return JsonResponse({'message': 'unknown methods',
                         'code': 50012})
