# 基于 PaddlePaddle 实现 HashNet(ICCV2017)

简体中文 | [English](./README_en.md)

   * [paddle_hashnet](#基于-paddlepaddle-实现-hashneticcv2017)
      * [一、简介](#一简介)
      * [二、复现精度](#二复现精度)
      * [三、数据集](#三数据集)
      * [四、环境依赖](#四环境依赖)
      * [五、快速开始](#五快速开始)
         * [step1: 下载本项目及训练权重](#step1-下载本项目及训练权重)
         * [step2: 修改参数](#step2-修改参数)
         * [step3: 验证模型](#step3-验证模型)
         * [step4: 训练模型](#step4-训练模型)
         * [step5: 验证预测](#step5-验证预测)
      * [六、TIPC](#六tipc)
      * [七、代码结构与详细说明](#七代码结构与详细说明)
      * [八、模型信息](#八模型信息)
      * [九、参考及引用](#九参考及引用)

- 原论文：[HashNet: Deep Learning to Hash by Continuation](https://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf).

- 官方原版代码（基于caffe/PyTorch）[HashNet](https://github.com/thuml/HashNet).

- 第三方参考代码（基于PyTorch）[DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch).

## 一、简介

对于大规模的最近邻搜索问题，比如**图像检索 Deep Hashing**等，哈希学习被广泛应用。然而现有基于深度学习的哈希学习方法需要先学习一个连续表征，再通过单独的二值化来生成二进制的哈希编码，这导致检索质量严重下降。HashNet则提出了两点改进方式：1. 针对不平衡分布的数据做了一个均衡化；2. 针对符号激活进行了改进，即让激活函数 $h=tanh(\beta z)$ 中的 $\beta$ 在训练过程中不断变化最终逼近1。下图展示了 HashNet 的主要架构。

<p align="center">
<img src="./resources/algorithm.png" alt="drawing" width="90%" height="90%"/>
    <h4 align="center">HashNet 架构</h4>
</p>

## 二、复现精度

|      | 16bits | 32bits | 48bits | 64bits|
|  ----  | ----  |  ----  |  ----  |  ----  |
| 验收指标  | 0.622  |  0.682  |  0.715  |  0.727  |
| 复现结果  | 0.619  |  0.682  |  0.715  |  0.734  |

本项目（基于 PaddlePaddle ）依次跑 16/32/48/64 bits 的结果罗列在上表中，且已将训练得到的模型参数与训练日志 log 存放于[output](output)文件夹下。由于训练时设置了随机数种子，理论上是可复现的。

## 三、数据集

MS COCO（即 [COCO2014](https://cocodataset.org) ）

- COCO2014 是一个图像识别、分割和字幕数据集。它包含 82,783 张训练图像和 40,504 张验证图像，共 80 个类别。

- 对其中没有类别信息的图像进行剪枝后，将训练图像和验证图像相结合，得到 122218 张图像。然后随机抽取 5000 张图像作为查询集，其余用作数据库；此外,从数据库中随机抽取 10,000 张图像作为训练集。数据集处理代码详见 [utils/datasets.py](utils/datasets.py)。另外数据集分割好的list放在 [./data/coco/](./data/coco/) 路径下。

- 需要**注意**的是：通过对比发现，原作者的list与第三方参考代码 [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch) 中的list略有不同，不过经过测试，两种list最终跑出来精度差不多。本项目复现的时候采用与原作者一样的list。

## 四、环境依赖

本人环境配置：

- Python: 3.7.11
- [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html): 2.2.2
- 硬件：NVIDIA 2080Ti * 2

## 五、快速开始

### step1: 下载本项目及训练权重

```
git clone https://github.com/hatimwen/paddle_hashnet.git
cd paddle_hashnet
```

- 由于权重比较多，加起来有 1 个 GB ，因此我放到百度网盘里了，烦请下载后按照 [六、代码结构与详细说明](#六代码结构与详细说明) 排列各个权重文件。

- 下载链接：[BaiduNetdisk](https://pan.baidu.com/s/1vQvv6aSuqMcqR3PEqxs91g), 提取码: pa1c 。

### step2: 修改参数

请根据实际情况，修改 [scripts](./scripts/) 中想运行脚本的配置内容（如：data_path, batch_size等）。

### step3: 验证模型

- **注意**：需要提前下载并排列好 [BaiduNetdisk](https://pan.baidu.com/s/1vQvv6aSuqMcqR3PEqxs91g) 中的各个预训练模型。

- 多卡，直接运行该脚本：

```shell
sh scripts/test_multi_gpu.sh
```

- 单卡，直接运行该脚本：

```shell
sh scripts/test_single_gpu.sh
```

### step4: 训练模型

- 多卡，直接运行该脚本（本项目运行场景为双卡，因此建议用双卡跑此脚本复现）：

```shell
sh scripts/train_multi_gpu.sh
```

- 单卡，直接运行该脚本：

```shell
sh scripts/train_single_gpu.sh
```

### step5: 验证预测

- 由于为数据库编码用时较长，因此已将通过 各个bits 的 HashNet 编码得到的数据库编码存在 `./output/database_code_*.npy` 。亦可将其删去后运行 [predict.py](predict.py) ，会在第一次预测的时候自动保存数据库编码。

- 以 64 bits 为例，验证预测的命令如下：

```shell
python predict.py \
--bit 64 \
--data_path ./datasets/COCO2014/ \
--img ./resources/COCO_val2014_000000403864.jpg \
--save_path ./output \
--show
```

<p align="center">
<img src="./resources/COCO_val2014_000000403864.jpg"/>
    <h4 align="center">验证图片</h4>
</p>

输出结果为:

```
----- Pretrained: Load model state from output/weights_64
----- Load code of database from ./output/database_code.npy
----- Predicted Hamm_min: 0.0
----- Found Mateched Pic: ./datasets/COCO2014/val2014/COCO_val2014_000000403864.jpg
----- Save Mateched Pic in: ./output/COCO_val2014_000000403864.jpg
```

<p align="center">
<img src="./output/COCO_val2014_000000403864.jpg"/>
    <h4 align="center">匹配到的图片</h4>
</p>

显然，匹配结果正确。

## 六、TIPC

- 本项目为 16/32/48/64 bits 分别写了对应的 TIPC 配置文件， 均位于 [test_tipc/configs](test_tipc/configs/) 文件夹下；另外方便起见， [scripts/tipc.sh](scripts/tipc.sh) 是一个直接跑所有 bits 的脚本；

- 详细日志放置在 [test_tipc/output](test_tipc/output/) 目录下；

- 具体 TIPC 介绍及使用流程请参阅：[test_tipc/README.md](test_tipc/README.md)。

## 七、代码结构与详细说明

```
|-- paddle_hashnet
    |-- data                # 数据集list
        |-- coco                # 暂时仅验证coco数据集
            |-- database.txt        # 数据库list
            |-- test.txt            # 测试集list
            |-- train.txt           # 训练集list
        |-- coco_lite           # 用于 TIPC 验证的少量数据list
            |-- database.txt        # 数据库list
            |-- test.txt            # 测试集list
            |-- train.txt           # 训练集list
    |-- datasets            # 数据集存放位置
        |-- coco_lite           # 用于 TIPC 验证的少量数据集
            |-- train2014           # 训练集图片
            |-- val2014             # 测试集图片
    |-- deploy
        |-- inference_python
            |-- infer.py            # TIPC 推理代码
            |-- README.md           # TIPC 推理流程介绍
    |-- models              # 模型定义
        |-- __init__.py
        |-- alexnet.py          # AlexNet 定义，注意这里有略微有别于 paddle 集成的 AlexNet
        |-- hashnet.py          # HashNet 算法定义
    |-- output              # 日志及模型文件
        |-- test                # 测试日志
            |-- log_16.txt          # 16bits的测试日志
            |-- log_32.txt          # 32bits的测试日志
            |-- log_48.txt          # 48bits的测试日志
            |-- log_64.txt          # 64bits的测试日志
        |-- train               # 训练日志
            |-- log_16.txt          # 16bits的训练日志
            |-- log_32.txt          # 32bits的训练日志
            |-- log_48.txt          # 48bits的训练日志
            |-- log_64.txt          # 64bits的训练日志
        |-- weights_16.pdparams     # 16bits的模型权重
        |-- weights_32.pdparams     # 32bits的模型权重
        |-- weights_48.pdparams     # 48bits的模型权重
        |-- weights_64.pdparams     # 64bits的模型权重
        |-- database_code_16.npy    # 数据库通过HashNet得到的16bits编码
        |-- database_code_32.npy    # 数据库通过HashNet得到的32bits编码
        |-- database_code_48.npy    # 数据库通过HashNet得到的48bits编码
        |-- database_code_64.npy    # 数据库通过HashNet得到的64bits编码
    |-- scripts
        |-- test_multi_gpu.sh   # 多卡测试脚本
        |-- test_single_gpu.sh  # 单卡测试脚本
        |-- tipc.sh             # TIPC 脚本
        |-- train_multi_gpu.sh  # 多卡训练脚本
        |-- train_single_gpu.sh # 单卡训练脚本
    |-- test_tipc               # 飞桨训推一体认证（TIPC）
    |-- utils
        |-- datasets.py         # dataset, dataloader, transforms
        |-- loss.py             # HashNetLoss 定义
        |-- lr_scheduler.py     # 学习率策略定义
        |-- tools.py            # mAP计算；随机数种子固定函数；database_code计算
    |-- export_model.py     # 模型动态转静态代码
    |-- main_multi_gpu.py   # 多卡训练测试代码
    |-- main_single_gpu.py  # 单卡训练测试代码
    |-- predict.py          # 预测演示代码
    |-- README.md
```

## 八、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | 文洪涛 |
| Email | hatimwen@163.com |
| 时间 | 2022.04 |
| 框架版本 | Paddle 2.2.2 |
| 应用场景 | 图像检索 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [预训练模型 提取码: pa1c](https://pan.baidu.com/s/1vQvv6aSuqMcqR3PEqxs91g)  |
| 在线运行 | [AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3964755?shared=1)|
| License | [Apache 2.0 license](LICENCE)|

## 九、参考及引用

```BibTeX
@inproceedings{cao2017hashnet,
  title={Hashnet: Deep learning to hash by continuation},
  author={Cao, Zhangjie and Long, Mingsheng and Wang, Jianmin and Yu, Philip S},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5608--5617},
  year={2017}
}
```

- [PaddlePaddle](https://github.com/paddlepaddle/paddle)

最后，非常感谢百度举办的[飞桨论文复现挑战赛（第六期）](https://aistudio.baidu.com/aistudio/competition/detail/205/0/introduction)让本人对 PaddlePaddle 理解更加深刻。
