​		该文档给出了适用于NNIE（Neural Network Inference Engine）的中文语音唤醒词模型的训练、转换方法说明。

**1. 背景知识**

**1.1 NNIE**

​		NNIE是华为海思推出的神经网络推理引擎，是一种硬件加速器，对于深度卷积神经网络等计算机视觉领域的模型有较完善的支持。

**1.2悟空模型**

​		悟空模型是NNIE支持的模型格式，后缀一般为.wk。目前悟空模型只能从caffe模型转换而来，无法直接训练生成。

**2.**   **模型训练与转换**

**2.1** **Caffe格式的中文KWS模型**

​		如1.2节所述，现有NNIE工具链只支持将Caffe模型转换为悟空模型，因此需要先训练出Caffe格式的KWS模型。模型方面，我们参考论文《

[Convolutional]: https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

》，实现了基于卷积神经网络的唤醒词识别模型。针对中文场景，我们采用出门问问公司开源的中文唤醒词数据集《

[MobvoiHotwords]: http://www.openslr.org/87/

》进行训练。Caffe版本的中文KWS模型的原型及预训练结果在caffe_models目录下给出。

​		在进行模型训练之前，需要对数据集进行了特征预处理，主要包括计算MFCC值和归一化等。其中，MFCC特征计算和归一化的实现逻辑在utils/feature_preprocessing.py中。

**2.2** **悟空格式的中文KWS模型**

​		采用nnie_mapper工具可以将2.1中训练出的caffe模型转换为悟空模型，转换过程中自动完成了量化操作。具体如下：

​		关于nnie_mapper的更多用法可以参考《***待更新***》。

**2.3** **其他格式的模型转换为悟空模型**

​		对于Pytorch、Tesnorflow等其它格式的模型，可以先将它们转换为caffe模型，然后按照2.2节中描述的方法将其转换为悟空模型。