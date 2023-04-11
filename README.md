# 一个支持ChatLLM模型训练的公开框架项目
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-390/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

这个项目旨在使用**Stanford Alpaca**构建统一的ChatLLM模型训练框架，支持的模型有：
- ChatGLM 
- Bloom （开发中）
- OPT （开发中）
- 其他

## 作者
- 赵健博 @ 梧桐车联/长安汽车, 王路宝 @ 梧桐车联/长安汽车, 郭苏州 @ 梧桐车联/长安汽车 and 吴明轩 @ 梧桐车联/长安汽车
## 概述
该项目将使用**Stanford Alpaca**的数据生成方法，训练市场上流行的ChatLLM模型。
## 教程
### ChatGLM-6B
在使用项目前，需要在[huggingface](https://huggingface.co/THUDM/chatglm-6b/tree/main)中下载模型，并使用项目中📁`DolphinGen/pretraining/chatglm-6b/modeling_chatglm.py`文件替换下载的`modeling_chatglm.py`文件。<br>
该项目中我们使用`gradient_checkpointing`对模型做了优化，使得ChatLLM模型能够在单卡GTX3090设备中运行。

## 环境
```bash
accelerate==0.16.0
protobuf==3.20.0
peft==0.2.0
transformers=4.27.3
torch==1.13.1+cu116
```
## 数据
`data`目录为训练数据保存位置，可根据示例数据`zh_seed_tasks.json`生成自己个数据。

## 训练
`script`目录为项目运行的脚本存放位置。执行`script/train_script.sh`脚本即可运行模型。
```bash
bash script/train_script.sh
```
