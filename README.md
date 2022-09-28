# GovSpeechRecognition

## Quick Start

model_file:链接: https://pan.baidu.com/s/1KZxoXDHhbmVh2Is0NXUv2A 提取码: n1tv  
put checkpoints dir into exp/deepspeech2  
klm_file into data/lm  
data_aishell.tgz into examples/dataset/data_shell

```shell
# start paddlespeech server
conda activate CrossWOZ
cd ../PaddleSpeech/demos/speech_server/
nohup python start_server.py > /dev/null 2>&1 &
```

```
cd GovSpeechRecognition
# source the environment

# run program
nohup python3 manage.py runserver 0.0.0.0:5555 > /dev/null 2>&1 &
```