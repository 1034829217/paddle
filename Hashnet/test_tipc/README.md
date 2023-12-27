
# 飞桨训推一体认证（TIPC）

## 1. 简介

飞桨除了基本的模型训练和预测，还提供了支持多端多平台的高性能推理部署工具。本文档提供了PaddleOCR中所有模型的飞桨训推一体认证 (Training and Inference Pipeline Certification(TIPC)) 信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。

<div align="center">
    <img src="docs/guide.png" width="1000">
</div>

## 2. 测试工具简介
### 目录介绍

```shell
test_tipc/
├── configs/                            # 配置文件目录
    ├── hashnet_16                      # hashnet_16 模型的测试配置文件目录 
        ├── train_infer_python.txt      # hashnet_16 模型基础训练推理测试配置文件
    ├── hashnet_32                      # hashnet_32 模型的测试配置文件目录 
        ├── train_infer_python.txt      # hashnet_32 模型基础训练推理测试配置文件
    ├── hashnet_48                      # hashnet_48 模型的测试配置文件目录 
        ├── train_infer_python.txt      # hashnet_48 模型基础训练推理测试配置文件
    ├── hashnet_64                      # hashnet_64 模型的测试配置文件目录 
        ├── train_infer_python.txt      # hashnet_64 模型基础训练推理测试配置文件
├──docs                                 # 文档目录
    ├── test_train_inference_python.md  # 基础训练推理测试说明文档
├── test_train_inference_python.sh      # TIPC基础训练推理测试解析脚本
├── common_func.sh                      # TIPC基础训练推理测试常用函数
└── README.md                           # 使用文档
```

### 测试流程概述

使用本工具，可以测试不同功能的支持情况，以及预测结果是否对齐。

本项目已完成基础训练预测测试，具体介绍及流程请参阅：[Linux GPU/CPU 基础训练推理测试](docs/test_train_inference_python.md)。

运行要测试的功能对应的测试脚本`test_train_inference_python.sh`，产出log，由log可以看到不同配置是否运行成功，结果如下：

```bash
Run successfully with command - python3.7 main_single_gpu.py --bit=64 --dataset='coco_lite' --data-path='./datasets/coco_lite/' --output-dir=./log/hashnet_64/lite_train_lite_infer/norm_train_gpus_0 --epoch=2   --batch-size=10!  
Run successfully with command - python3.7 main_single_gpu.py --eval --bit=64 --dataset='coco_lite' --data-path='./datasets/coco_lite/' --pretrained=./log/hashnet_64/lite_train_lite_infer/norm_train_gpus_0/model_best_64!  
Run successfully with command - python3.7 export_model.py --bit=64 --pretrained=./log/hashnet_64/lite_train_lite_infer/norm_train_gpus_0/model_best_64 --save-inference-dir=./log/hashnet_64/lite_train_lite_infer/norm_train_gpus_0!  
Run successfully with command - python3.7 deploy/inference_python/infer.py --bit=64 --dataset='coco_lite' --data-path='./datasets/coco_lite/' --save-path='./tipc_output/' --use-gpu=True --model-dir=./log/hashnet_64/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=True > ./log/hashnet_64/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  
Run successfully with command - python3.7 deploy/inference_python/infer.py --bit=64 --dataset='coco_lite' --data-path='./datasets/coco_lite/' --save-path='./tipc_output/' --use-gpu=False --model-dir=./log/hashnet_64/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=True > ./log/hashnet_64/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !  
```