# Paddle-BUTD-Captioning

## 一、简介
参考论文：《Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering》[论文链接](https://ieeexplore.ieee.org/document/8578734)

在人类视觉系统中，存在自上而下(Top-Down Attention)和自下而上(Bottom-Up Attention)两种注意机制。前者注意力由当前任务所决定，我们会根据当前任务聚焦于与任务紧密相关的部分，后者注意力指的是我们会被显著的、突出的事物所吸引。 视觉注意大部分属于自上而下类型，图像作为输入，建模注意权值分布，然后作用于CNN提取的图像特征。然而，这种方法的注意作用图像对应于下图的左图，没有考虑图片的内容。对于人类来说，注意力会更加集中在图片的目标或其他显著区域，所以论文作者引进自下而上注意(Bottom-Up Attention)机制，如下图的右图所示，注意力作用于显著物体上。

![对比图](image/comparison.jpg)

Caption model结构如下图所示，模型共有2个LSTM模块，一个是Language LSTM，另一个是Top-Down Attention LSTM。 本文的Bottom-Up Attention 用的是目标检测(object detection)领域的Faster R-CNN方法来提取。

![模型结构](image/model.png)

[参考项目地址链接](https://github.com/ruotianluo/ImageCaptioning.pytorch)

## 二、复现精度
代码在coco2014数据集上训练，复现精度：

Cross-entropy Training
|Bleu_1|Bleu_2|Bleu_3|Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|0.761|0.598|0.459|0.350|0.272|0.562|1.107|0.203|

SCST(Self-critical Sequence Training)
|Bleu_1|Bleu_2|Bleu_3|Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|0.799|0.641|0.493|0.373|0.275|0.580|1.202|0.209|

## 三、数据集
coco2014 image captions [论文](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)，采用“Karpathy” data split [论文](https://arxiv.org/pdf/1412.2306v2.pdf)

[coco2014数据集下载](https://aistudio.baidu.com/aistudio/datasetdetail/28191)

- 数据集总大小：123287张
  - 训练集：113287张
  - 验证集：5000张
  - 测试集：5000张
- 标签文件：dataset_coco.json

## 四、环境依赖
paddlepaddle-gpu==2.1.2  cuda 10.2

opencv-python==4.5.3.56

yacs==0.1.7

yaml==0.2.5

## 五、快速开始

### step1: 加载数据
加载预处理数据文件全放在本repo的data/下 

[“Karpathy” data split json](https://aistudio.baidu.com/aistudio/datasetdetail/107078)

通过Faster R-CNN模型提取的Bottom-up 原始特征文件[链接](https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/data/README.md)

生成cocotalk_label.h5和cocoktalk.json文件
```bash
python scripts/prepro_labels.py
```

生成cocobu_att、池化特征cocobu_fc、边框特征cocobu_box
```bash
python scripts/make_bu_data.py
```

可以直接[加载上述预训练数据](https://aistudio.baidu.com/aistudio/datasetdetail/107198)。其中cocobu_att分成cocobu_att_train和cocobu_att_val上传，加载完成后，要合并成cocobu_att

**Install dependencies**
```bash
pip install -r requestments.txt
```

### step2: 训练
训练过程过程分为两步：Cross-entropy Training和SCST(Self-critical Sequence Training)

第一步Cross-entropy Training：

```bash
python3 train.py --cfg configs/updown.yml  
```

第二步SCST(Self-critical Sequence Training)：

```bash
python3 train.py --cfg configs/updown_rl.yml
```

训练的模型数据和日志会放在本repo的log/下

### step3: 验证评估

验证模型
```bash
python eval.py
```

测试时程序会加载本repo的log/下保存的训练模型数据，我们最终验证评估的是使用SCST优化的模型。

可以[下载训练好的模型数据](https://aistudio.baidu.com/aistudio/datasetdetail/108181)，放到本repo的log/下，然后直接执行验证指令。

## 六、代码结构与参数说明

### 6.1 代码结构

```
├─config                          # 配置
├─models                          # 模型
├─misc                            # 工具以及测试代码
├─modules                         # 损失函数模块
├─data                            # 训练数据目录
├─scripts                         # 预处理文件
│  eval.py                        # 评估
│  dataloader.py                  # 加载器
│  README.md                      # readme
│  requirements.txt               # 依赖
│  train.py                       # 训练
```
### 6.2 参数说明

可以在config文件中设置训练与评估相关参数

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | Lieber |
| 时间 | 2021.09 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 多模态 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [训练好的模型](https://aistudio.baidu.com/aistudio/datasetdetail/108181)|
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/2345929)|
| 复现总结 | [论坛](https://aistudio.baidu.com/paddle/forum/topic/show/991202)|
