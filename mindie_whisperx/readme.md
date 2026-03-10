# WhisperX推理指导

- [WhisperX推理指导](#whisperx推理指导)
- [概述](#概述)
- [推理环境准备](#推理环境准备)
- [快速上手](#快速上手)
  - [获取源码](#获取源码)
  - [模型编译](#模型编译)
  - [模型推理](#模型推理)

# 概述
使用mindtorch部署高性能版本的whisper-large-v3模型。其中，将开源的whisperX中的语音切分和自动组batch的能力迁移过来，达到提升性能的目的。


# 推理环境准备

- 该模型需要以下插件与驱动

  **表 1**  版本配套表

  | 配套   | 版本        | 环境准备指导 |
  |-----------| ------- | ------------ |
  | Python | 3.10.13   | -            |
  | torch  | 2.1.0+cpu | -            |
  | torch_audio  | 2.1.0+cpu | -            |
  | CANN   | 8.0.B023  | -            |
  | MindIE | 1.0.B030  | -       |

# 快速上手
## 获取源码

1. 安装mindie包

   ```bash
   # 安装mindie
   chmod +x ./Ascend-mindie_xxx.run
   ./Ascend-mindie_xxx.run --install
   source /usr/local/Ascend/mindie/set_env.sh
   ```


2. Whisper large V3模型权重下载路径:
    ```bash
   mkdir whisper_pretrained
    https://huggingface.co/openai/whisper-large-v3/tree/main
    ```
    将权重文件存放至当前目录下的whisper_pretrained文件夹，请先创建改文件夹。仅以whisper_pretrained为例，用户可根据实际情况创建目录。
    

3. WhisperX中VAD模型权重下载路径:

    ```bash
   mkdir vad_pretrained
   wget https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin --no-check-certificate
    mv   pytorch_model.bin ./vad_pretrained/whisperx-vad-segmentation.bin
   ```
    将权重文件存放至当前目录下的vad_pretrained文件夹，请先创建改文件夹。仅以vad_pretrained为例，用户可根据实际情况创建目录。


4. 安装依赖
    ```
    pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    pip3 install nltk
    pip3 install librosa
    pip3 install transformers==4.36.0
    pip3 install numpy==1.24.0
    pip3 install ml-dtypes
    pip3 install cloudpickle
    pip3 install pyannote.audio==3.1.1
    ``` 
   如果环境上安装了torch_npu, pyannote.audio会默认引入, 需要卸载torch_npu.
   同时需要保证环境安装了libsndfile1, ffmpeg库.

## 模型编译
模型编译分为两部分内容，需要分别编译whisper模型和vad模型，并将编译好的模型保存到同一个路径，大概耗时两小时左右。

### 编译whisper模型
```
    python3 compile_whisper.py \
    -model_path ./whisper_pretrained \
    -bs 16 \
    -save_path ./compiled_models \
    -soc_version *
  ```
    参数说明：
      - -model_path: 预训练模型路径,必选。
      - -bs: batch_size, 默认值为16， 可选。
      - -save_path: 编译好的模型的保存文件，必选。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -soc_version: 芯片类型,必选。
      - -hardware: 机器型号，默认值800IA2，可选["300IPro", "800IA2"]。
    约束说明：
        1. 当前暂不支持动态batch，batch_size改变后，需要重新编图。
        2. 支持的hardware类型为"300IPro"或"800IA2"。
        3. 芯片类型需要用户在环境上查询得到。
        如果无法确定当前设备的soc_version，则在安装NPU驱动包的服务器执行npu-smi info命令进行查询，
        在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。


### 编译vad模型
在编译VAD模型前需要先打补丁，使用如下命令
```
cd pipeline
python3 patch_apply.py
python3 remove_script.py
cd ..
```
打完补丁后，开始编译vad模型，注意入参vad_model_path的路径需要与前面预训练权重保存的路径一致。
```
python3 compile_vad.py \
-vad_model_path ./vad_pretrained \
-soc_version *
```

参数说明：
  - -vad_model_path: VAD预训练模型路径,必选。
  - -save_path: 编译好的模型的保存文件，可选，默认值"./compiled_models"。
  - -device_id: 选在模型运行的卡编号，默认值0，可选。
  - -soc_version: 芯片类型,必选。

注：1.VAD模型编译的保存路径需要和Whisper-large-V3模型编译保存路径一致 
   2.芯片类型需要用户在环境上查询得到。如果无法确定当前设备的soc_version，则在安装NPU驱动包的服务器执行npu-smi info命令进行查询。
   在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的soc_version值为Ascendxxxyy。


## 模型推理
1. 开启cpu高性能模式进一步提升性能，开启失败不影响功能。

```
echo performance |tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
```
2. 安装绑核工具
```
apt-get update
apt-get  install numactl
```
查询卡的NUMA node
```
lspci -vs bus-id
```
注：通过npu-smi info查询推理卡id对应的bus-id，然后使用上面的命令得到推理卡id对应的NUMA node。
使用lscpu指令查询NUMA node对应亲和的CPU核
```
lscpu
```
能够看到对应NUMA node和对应哪些cpu，例如
```
NUMA node0 CPU(s): 0-31
NUMA node1 CPU(s): 32-63
```
3.绑定CPU并模型推理，假设查询到的核数为0-31，具体
   ```
    taskset -c 0-31 python3 example.py \
    -whisper_model_path ./whisper_pretrained \
    -vad_model_path ./vad_pretrained \
    -compiled_models ./compiled_models
    -audio_path /audio_path
    -bs *
   ```

    参数说明：
      - -whisper_model_path : whisper的预训练模型路径,必选。
      - -vad_model_path : vad的预训练模型路径，必选。
      - -bs: batch_size大小，需要与编图时传入的大小保持一致。
      - -compiled_models: 编译好的模型的保存文件，可选，默认值"./compiled_models"。
      - -device_id: 选在模型运行的卡编号，默认值0，可选。
      - -open_warm_up: 是否开启预热,建议测试性能时开启该开关。