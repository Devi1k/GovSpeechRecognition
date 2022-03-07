# GovSpeechRecognition

## Quick Start

model_file:链接: https://pan.baidu.com/s/1KZxoXDHhbmVh2Is0NXUv2A 提取码: n1tv  
put checkpoints dir into exp/deepspeech2  
klm_file into data/lm  
data_aishell.tgz into examples/dataset/data_shell

```shell
# source the environment
cd examples/aishell/s0
source path.sh
source parse_options.sh`

# prepare data
bash ./local/data.sh
```

```
cd GovSpeechRecognition
# source the environment
conda activate env
# run program
make run
```