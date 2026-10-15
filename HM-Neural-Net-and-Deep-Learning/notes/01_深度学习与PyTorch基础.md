# 01 深度学习与 PyTorch 基础

> **对应课程**：黑马《神经网络与深度学习》Day01–Day02（分P 1–18）
> **覆盖内容**：深度学习概念 / 特点 / 常见模型 / 应用场景 · PyTorch 框架简介 · 张量创建 · 类型转换 · 数值计算 · 运算函数
> **整理日期**：2026-10-15

---

## 1 深度学习简介

### 1.1 深度学习概念

- 深度学习是机器学习的一类算法, 以人工神经网络为结构, 可以实现自动提取特征
- 深度学习核心思想是人工神经网络为结构, 自动提取特征

### 1.2 深度学习特点

- 自动提取特征
- 解释性差
- 大量数据和高性能计算能力
- 非线性转换(引入非线性因素)

### 1.3 深度学习模型

- ANN 人工神经网络  感知机

- CNN 卷积神经网络  图像/视频
- RNN 循环神经网络 NLP
- transformer  RNN衍生出来的
- 自编学习器
- ...

### 1.4 深度学习应用场景

- 自然语言处理NLP
  - 生成式AI AIGC 大模型
  - 机器翻译
  - 语音识别
  - ...
- 计算机视觉CV
  - 图像识别
  - 面部解锁
  - 视频合成
  - ...
- 推荐系统
  - 电影
  - 音乐
  - 文章
  - 视频
  - 商品

## 2 PyTorch框架简介

- pytorch是深度学习的框架, python的第三方包, 数据是以张量类型存在
- pytorch特点
  - 数据类型是张量类型
  - 自动微分模块, 自动求导/梯度
  - 可以在GPU/TPU/NPU上运行, 加速运行
  - 兼容各种平台 系统/硬件(显卡)
- pytorch目前更新到2.5版本

## 3 张量创建

### 3.1 什么是张量

- 张量是矩阵, 可以是多维
  - 0维->标量
  - 1维->[1 2 3 4 5]
  - 2维->[[1 2 3],[4 5 6]]
  - 3维...
- 张量是通过类创建出来的对象, 提供各种方法和属性

### 3.2 基本创建方式

- torch.tensor(data=): 指定数据

- torch.Tensor(data=, size=): 指定数据或形状

- torch.IntTensor(data=)/FloatTensor(): 指定数据

  ```python
  import torch
  import numpy as np
  
  
  # torch.tensor(data=, dtype=,): 根据指定数据或指定元素类型创建张量
  # data: 数据
  # dtype: 元素类型
  def dm01():
  	list1 = [[1., 2, 3], [4, 5, 6]]  # 创建的张量为float32
  	int1 = 10
      # array默认类型是float64, 所以创建的张量为float64
  	n1 = np.array([[1., 2., 3.], [4., 5., 6.]])
  	t1 = torch.tensor(data=list1)
  	t2 = torch.tensor(data=int1)
  	t3 = torch.tensor(data=n1)
  	print('t1的值->', t1)
  	print('t1类型->', type(t1))
  	print('t1元素类型->', t1.dtype)
  	print('t2的值->', t2)
  	print('t2类型->', type(t2))
  	print('t3的值->', t3)
  	print('t3类型->', type(t3))
  
  
  # torch.Tensor(data=, size=): 根据指定数据或指定形状创建张量
  # data: 数据
  # size: 形状, 接收元组 (0轴, 1轴, ...) -> 元组有多少个元素就是多少维张量, 对应维度上值就是数据个数
  def dm02():
  	# 指定数据
  	t1 = torch.Tensor(data=[[1.1, 1.2, 1.3], [2.2, 2.3, 2.4]])
  	print('t1的值->', t1)
  	print('t1类型->', type(t1))
  	# 指定形状
  	t2 = torch.Tensor(size=(2, 3))
  	print('t2的值->', t2)
  	print('t2类型->', type(t2))
  
  
  # torch.IntTensor(data=)/LongTensor()/FloatTensor()/DoubleTensor(): 创建指定类型的张量
  # data: 数据
  def dm03():
  	# 如果元素类型不是指定类型, 会自动转换
  	t1 = torch.IntTensor([[1.1, 2, 3.7], [4, 5, 6]])
  	t2 = torch.FloatTensor([[1.1, 2, 3.7], [4, 5, 6]])
  	print('t1的值->', t1)
  	print('t1类型->', type(t1))
  	print('t1元素类型->', t1.dtype)
  	print('t2的值->', t2)
  	print('t2类型->', type(t2))
  	print('t2元素类型->', t2.dtype)
  
  
  if __name__ == '__main__':
  	dm01()
  	# dm02()
  	# dm03()
  ```

### 3.3 线性和随机张量

- 线性张量

  - torch.arange()
  - torch.linspace()

- 随机张量

  - torch.rand()/randn()
  - torch.randint()
  - torch.initial_seed()
  - torch.manual_seed()

  ```python
  import torch
  
  
  # torch.arange(start=, end=, step=): 创建指定步长的线性张量  左闭右开
  # start: 起始值
  # end: 结束值
  # step: 步长, 默认1
  # torch.linspace(start=, end=, steps=): 创建指定元素个数的线性张量  左闭右闭
  # start: 起始值
  # end: 结束值
  # steps: 元素个数
  # step=(end-start)/(steps-1)  value_i=start+step*i
  def dm01():
  	t1 = torch.arange(start=0, end=10, step=2)
  	print('t1的值是->', t1)
  	print('t1类型是->', type(t1))
  	t2 = torch.linspace(start=0, end=9, steps=9)
  	print('t2的值是->', t2)
  	print('t2类型是->', type(t2))
  
  
  # torch.rand(size=)/randn(size=): 创建指定形状的随机浮点类型张量
  # torch.randint(low=, high=, size=): 创建指定形状指定范围随机整数类型张量  左闭右开
  # low: 最小值
  # high: 最大值
  # size: 形状, 元组
  
  # torch.initial_seed(): 查看随机种子数
  # torch.manual_seed(seed=): 设置随机种子数
  def dm02():
  	# (5, 4): 5行4列
  	t1 = torch.rand(size=(5, 4))
  	print('t1的值是->', t1)
  	print('t1类型->', type(t1))
  	print('t1元素类型->', t1.dtype)
  	print('t1随机种子数->', torch.initial_seed())
  	# 设置随机种子数
  	torch.manual_seed(seed=66)
  	t2 = torch.randint(low=0, high=10, size=(2, 3))
  	print('t2的值是->', t2)
  	print('t2类型->', type(t2))
  	print('t2元素类型->', t2.dtype)
  	print('t2随机种子数->', torch.initial_seed())
  
  
  
  if __name__ == '__main__':
  	# dm01()
  	dm02()
  ```

### 3.4 0/1/指定值张量

- torch.ones/zeros/full(size=[, fill_value=])

- torch.ones_like/zeros_like/full_like(input=tensor[, fill_value=])

  ```python
  import torch
  
  
  # torch.ones(size=): 根据形状创建全1张量
  # torch.ones_like(input=): 根据指定张量的形状创建全1张量
  def dm01():
  	t1 = torch.ones(size=(2, 3))
  	print('t1的值是->', t1)
  	print('t1的形状是->', t1.shape)
  	print('t1的元素类型是->', t1.dtype)
  	# 形状: (5, )
  	t2 = torch.tensor(data=[1, 2, 3, 4, 5])
  	t3 = torch.ones_like(input=t2)
  	print('t2的形状是->', t2.shape)
  	print('t3的值是->', t3)
  	print('t3的形状是->', t3.shape)
  
  
  # torch.zeros(size=): 根据形状创建全0张量
  # torch.zeros_like(input=): 根据指定张量的形状创建全0张量
  def dm02():
  	t1 = torch.zeros(size=(2, 3))
  	print('t1的值是->', t1)
  	print('t1的形状是->', t1.shape)
  	print('t1的元素类型是->', t1.dtype)
  	# 形状: (5, )
  	t2 = torch.tensor(data=[1, 2, 3, 4, 5])
  	t3 = torch.zeros_like(input=t2)
  	print('t2的形状是->', t2.shape)
  	print('t3的值是->', t3)
  	print('t3的形状是->', t3.shape)
  
  
  # torch.full(size=, fill_value=): 根据形状和指定值创建指定值的张量
  # torch.full_like(input=, fill_value=): 根据指定张量形状和指定值创建指定值的张量
  def dm03():
  	t1 = torch.full(size=(2, 3, 4), fill_value=10)
  	t2 = torch.tensor(data=[[1, 2], [3, 4]])
  	t3 = torch.full_like(input=t2, fill_value=100)
  	print('t1的值是->', t1)
  	print('t1的形状是->', t1.shape)
  	print('t3的值是->', t3)
  	print('t3的形状是->', t3.shape)
  
  
  if __name__ == '__main__':
  	# dm01()
  	# dm02()
  	dm03()
  ```

### 3.5 指定元素类型张量

- tensor.type(dtype=)

- tensor.half()/float()/double()/short()/int()/long()

  ```python
  import torch
  
  
  # torch.tensor(data=, dtype=):
  # dtype: 指定元素类型, 浮点类型默认是float32
  
  # tensor.type(dtype=): 修改张量元素类型
  # torch.float32
  # torch.FloatTensor
  # torch.cuda.FloatTensor
  def dm01():
  	t1 = torch.tensor(data=[[1., 2., 3.], [4., 5., 6.]], dtype=torch.float16)
  	print('t1的元素类型->', t1.dtype)
  	# 转换成float32
  	t2 = t1.type(dtype=torch.FloatTensor)
  	t3 = t1.type(dtype=torch.int64)
  	print('t2的元素类型->', t2.dtype)
  	print('t3的元素类型->', t3.dtype)
  
  
  # tensor.half()/float()/double()/short()/int()/long()
  def dm02():
  	t1 = torch.tensor(data=[1, 2])
  	print('t1的元素类型->', t1.dtype)
  	# t2 = t1.half()
  	t2 = t1.int()
  	print(t2)
  	print('t2的元素类型->', t2.dtype)
  
  
  if __name__ == '__main__':
  	# dm01()
  	dm02()
  ```

## 4 张量类型转换

### 4.1 张量转换为NumPy数组

```python
import torch
import numpy as np


# 张量转换成numpy数组
# tensor.numpy(): 共享内存, 修改一个另外一个也跟着变, 可以通过copy()函数不共享内存
def dm01():
	t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
	print('t1->', t1)
	# 转换成numpy数组
	# n1 = t1.numpy()
	n1 = t1.numpy().copy()
	print('n1->', n1)
	print('n1的类型->', type(n1))
	# 修改n1的第一个值
	# [0][0]->第一行第一列的元素
	n1[0][0] = 100
	print('n1修改后->', n1)
	print('t1->', t1)
```

### 4.2 NumPy数组转换为张量

```python
# numpy数组转换成张量
# torch.from_numpy(ndarray): 共享内存, 对ndarray数组进行copy()
# torch.tensor(data=ndarray): 不共享内存
def dm02():
	n1 = np.array([[1, 2, 3], [4, 5, 6]])
	# 转换成张量
	# 共享内存
	t1 = torch.from_numpy(n1)
	# 不共享内存
	# t1 = torch.from_numpy(n1.copy())
	# t1 = torch.tensor(data=n1)
	print('t1->', t1)
	print('t1类型->', type(t1))
	# 修改张量元素
	t1[0][0] = 8888
	print('t1修改后->', t1)
	print('n1->', n1)
```

### 4.3 提取标量张量的数值

```python
import torch


# tensor.item(): 提取单个元素张量的数值, 张量可以是标量张量/一维张量/二维张量...只要是单个元素即可
def dm01():
	# 数值转换成张量
	# 标量
	t1 = torch.tensor(data=10)
	# 一维
	# t1 = torch.tensor(data=[10])
	# 二维
	# t1 = torch.tensor(data=[[10]])
	print('t1->', t1)
	print('t1形状->', t1.shape)
	# 单个元素张量转换成数值, 提取数值
	print('t1.item()->', t1.item())


if __name__ == '__main__':
	dm01()
```

## 5 张量数值计算

### 5.1 基本运算

- `+` `-` `*` `/` `-`

- tensor/torch.add() sub() mul() div() neg()

- `tensor/torch.add_()` `sub_()` `mul_()` `div_()` `neg_()`

  ```python
  import torch
  
  
  # 运算: 张量和数值之间运算, 张量和张量之间运算
  # + - * / -
  # add(other=) sub() mul() div() neg()  不修改原张量
  # add_() sub_() mul_() div_() neg_()  修改原张量
  
  def dm01():
  	# 创建张量
  	t1 = torch.tensor(data=[1, 2, 3, 4])
  	# 张量和数值运算
  	t2 = t1 + 10
  	print('t2->', t2)
  	# 张量之间运算, 对应位置的元素进行计算
  	t3 = t1 + t2
  	print('t3->', t3)
  
  	# add() 不修改原张量
  	t1.add(other=100)
  	t4 = torch.add(input=t1, other=100)
  	print('t4->', t4)
  
  	# neg_() 修改原张量, 负号
  	t5 = t1.neg_()
  	print('t1->', t1)
  	print('t5->', t5)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

### 5.2 点乘运算

- 对应位置的元素进行乘法计算, 一般要求张量形状相同

  ```python
  import torch
  
  
  # 点乘: 又称为阿达玛积, 张量元素级乘法, 对应位置的元素进行点乘, 一般要求两个张量形状相同  *  mul()
  def dm01():
  	# t1 = torch.tensor(data=[[1, 2], [3, 4]])
  	# (2, )
  	t1 = torch.tensor(data=[1, 2])
  	# (2, 2)
  	t2 = torch.tensor(data=[[5, 6], [7, 8]])
  	t3 = t1 * t2
  	print('t3->', t3)
  	t4 = torch.mul(input=t1, other=t2)
  	print('t4->', t4)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

### 5.3 矩阵乘法运算

- 第一个矩阵的行数据和第二个矩阵的列数据相乘

  ```python
  import torch
  
  
  # 矩阵乘法: (n, m) * (m, p) = (n, p)  第一个矩阵的行和第二个矩阵的列相乘  @  torch.matmul(input=, ohter=)
  def dm01():
  	# (2, 2)
  	t1 = torch.tensor(data=[[1, 2],
  							[3, 4]])
  	# (2, 3)
  	t2 = torch.tensor(data=[[5, 6, 7],
  							[8, 9, 10]])
  
  	# @
  	t3 = t1 @ t2
  	print('t3->', t3)
  	# torch.matmul(): 不同形状, 只要后边维度符合矩阵乘法规则即可
  	t4 = torch.matmul(input=t1, other=t2)
  	print('t4->', t4)
  
  
  if __name__ == '__main__':
  	dm01()
  ```

## 6 张量运算函数

- mean()

- sum()

- min()/max()

- dim: 按不同维度计算

- exp(): 指数

- sqrt(): 平方根

- pow(): 幂次方

- log()/log2()/log10(): 对数

  ```python
  import torch
  
  
  def dm01():
  	# 创建张量
  	t1 = torch.tensor(data=[[1., 2, 3, 4],
  							[5, 6, 7, 8]])
  
  	# dim=0 按列
  	# dim=1 按行
  	# 平均值
  	print('所有值平均值->', t1.mean())
  	print('按列平均值->', t1.mean(dim=0))
  	print('按行平均值->', t1.mean(dim=1))
  	# 求和
  	print('所有值求和->', t1.sum())
  	print('按列求和->', t1.sum(dim=0))
  	print('按行求和->', t1.sum(dim=1))
  	# sqrt: 开方 平方根
  	print('所有值开方->', t1.sqrt())
  	# pow: 幂次方  x^n
  	# exponent:几次方
  	print('幂次方->',torch.pow(input=t1, exponent=2))
  	# exp: 指数 e^x  张量的元素值就是x
  	print('指数->', torch.exp(input=t1))
  	# log: 对数  log(x)->以e为底  log2()  log10()
  	print('以e为底对数->', torch.log(input=t1))
  	print('以2为底对数->', t1.log2())
  	print('以10为底对数->', t1.log10())
  
  
  if __name__ == '__main__':
  	dm01()
  ```


---

# 附一 · 课程讲义

> 来源：黑马《神经网络与深度学习》配套讲义，按主题归档到本节。


## 什么是深度学习[知道]

**深度学习**（Deep Learning）是机器学习的分支，是一种以**人工神经网络为架构**，**对数据进行特征学习**的算法。深度学习中的形容词“深度”是指在网络中使用多层。

![1733800163423](assets/1733800163423.png)

深度学习核心思想是==通过模仿人脑的神经网络来处理和分析复杂的数据，从大量数据中自动提取复杂特征==，擅长处理高维数据，如图像、语音和文本。

![1733800312800](assets/1733800312800.png)


## 深度学习特点[知道]

- **多层非线性变换**：深度学习模型由多个层次组成，每一层都应用非线性激活函数对输入数据进行变换。较低的层级通常捕捉到简单的特征（如边缘、颜色等），而更高的层级则可以识别更复杂的模式（如物体或面部识别）。

- **自动特征提取**：与传统机器学习算法不同，深度学习能够自动从原始数据中学习到有用的特征，而不需要人工特征工程。这使得深度学习在许多领域中表现出色。

- **大数据和计算能力**：深度学习模型通常需要大量的标注数据和强大的计算资源（如GPU）来进行训练。大数据和高性能计算使得深度学习在图像识别、自然语言处理等领域取得了显著突破。
- **可解释性差**：深度学习模型内部的运作机制相对不透明，被称为“黑箱”，这意味着理解模型为什么做出特定决策可能会比较困难。这对某些应用场景来说是一个挑战。


## 常见的深度学习模型[了解]

- **卷积神经网络 (Convolutional Neural Networks, CNN)**：
  - 主要用于图像处理任务，如图像分类、目标检测、图像分割等。
  - 特点是使用卷积层来自动提取图像中的局部特征，并通过池化层减少参数数量，提高计算效率。
- **循环神经网络 (Recurrent Neural Networks, RNN)**：
  - 适用于处理序列数据，例如自然语言处理（NLP）、语音识别等。
  - RNN具有记忆功能，可以处理输入数据的时间依赖性，但标准RNN难以捕捉长期依赖关系。
- **自编码器 (Autoencoders)**：
  - 一种无监督学习模型，通常用于降维、特征学习或者异常检测。
  - 自编码器由编码器和解码器两部分组成，前者将输入压缩成一个较低维度的表示，后者尝试从这个低维表示重建原始输入。
- **生成对抗网络 (Generative Adversarial Networks, GAN)**：
  - 包含两个子网络：生成器和判别器。生成器负责创建看起来真实的假样本，而判别器则试图区分真假样本。
  - GAN广泛应用于图像生成、视频合成等领域。
- **Transformer**：
  - 主要用于自然语言处理（NLP）任务，尤其是机器翻译、文本生成等。
  - Transformer摒弃了传统的递归结构，采用自注意力机制（self-attention），使得它能够并行处理整个句子的信息，在机器翻译、文本摘要等任务中表现出色。


## 深度学习应用场景[了解]

- ==计算机视觉（Computer Vision）==

  - **图像分类**：将图像分为不同的类别。常用于人脸识别、物体检测等。
    - 自动标注社交媒体照片、医疗影像中的病变检测。
  - **目标检测（Object Detection）**：在图像或视频中定位并分类多个对象。
    - 自动驾驶中的行人检测、监控视频中的入侵检测。
  - **面部识别**：通过分析面部特征进行身份验证或分类。
    - 手机解锁、安防监控系统。

  - **图像生成**：基于输入生成新的图像，如风格转换、图像超分辨率等。
    - 艺术风格迁移、老旧照片修复。

- ==自然语言处理（Natural Language Processing, NLP）==

  - **机器翻译**：使用深度学习模型将一种语言的文本自动翻译成另一种语言。
    - Google翻译、实时语音翻译。
  - **情感分析**：分析文本中的情感倾向，如正面、负面或中性。
    - 社交媒体监控、产品评论分析。

  - **文本生成**：生成符合语法和语义的自然语言文本。
    - 自动写作助手、新闻生成。

  - **语音识别**：将语音转化为文字。
    - 智能助手（如Siri、Alexa）、自动字幕生成。
  - **聊天机器人（Chatbot）**：通过深度学习理解用户输入并生成合理的回应。
    - 客服机器人、虚拟助手（如GPT类模型）。

- ==推荐系统（Recommendation Systems）==

  - **电影、音乐推荐**：根据用户历史的评分和行为，推荐相关的电影、音乐或电视剧。
    - Netflix、Spotify的个性化推荐。
  - **电商推荐**：根据用户的购买历史和浏览习惯推荐商品。
    - 亚马逊、淘宝的商品推荐系统。
  - **社交媒体推荐**：分析用户的社交行为，推荐相关内容或朋友。
    - Facebook、Instagram的内容推荐。

  ...

![1733802918559](assets/1733802918559.png)


## 深度学习发展史[了解]

- 早期探索

  - 20世纪40年代：沃尔特·皮茨（Walter Pitts）和沃伦·麦卡洛克（Warren McCulloch）等开始模仿生物神经系统来构建计算模型，如McCulloch-Pitts神经元
  - 1958年：弗兰克·罗森布拉特（Frank Rosenblatt）提出感知器概念，能够进行简单的二分类任务
  - 1960年代末：出现了多层感知器（MLP），但当时由于计算能力和数据量的限制，这些模型的应用受到很大限制
- 挑战与瓶颈

  - 1986年：**反向传播算法（Backpropagation）的提出标志着神经网络研究的一个重要突破。杰弗里·辛顿（Geoffrey Hinton）和大卫·鲁梅尔哈特**（David Rumelhart）等人提出了反向传播算法，使得多层神经网络（即深层网络）能够通过梯度下降优化参数，解决复杂的非线性问题。
  - 虽然神经网络方法在一些领域表现不错，但由于计算资源的限制以及对复杂数据（如图像和语音）的处理能力较弱，深度学习未能广泛应用。此时，**支持向量机（SVM）**、**决策树**等传统机器学习方法成为主流。
- 复兴与突破

  - **2006年**：**杰弗里·辛顿**和其团队提出了**深度信念网络（DBN）**，标志着深度学习的复兴。他们引入了无监督预训练的技术，使得深层网络能够有效训练。这为深度学习的发展奠定了基础。
  - **2012年**：深度学习的一个重要突破是**AlexNet**的出现。**亚历克斯·克里泽夫斯基**（Alex Krizhevsky）在**ImageNet图像分类竞赛**中使用了一个深度卷积神经网络，显著提升了图像分类的精度，比传统方法提高了20%以上。AlexNet的成功标志着深度学习在计算机视觉领域的成功应用。
  - **2014年**：生成对抗网络（**GANs**）由**伊恩·古德费洛**（Ian Goodfellow）等人提出，开启了生成模型的新时代，能够生成非常逼真的图像、音频和视频。
  - **2015年**：**ResNet**（残差网络）由**何凯明**（Kaiming He）等提出，解决了深度网络中的梯度消失和梯度爆炸问题，允许训练极深的网络（如50层、152层），极大推动了深度学习在图像识别任务中的应用。
- 爆发期
  - **2016年：**Google AlphaGo 战胜李世石（人工智能第三次浪潮），AlphaGo 展现了深度强化学习（Deep Reinforcement Learning）在解决复杂问题上的巨大潜力，将其推向了公众视野。
  - **2017年：**自然语言处理NLP的Transformer框架出现，奠定了后续预训练语言模型（如 BERT 和 GPT）的基础。
  - **2018年：**BERT和GPT的出现，基于Transformer架构的预训练语言模型的代表。
  - **2022年：**ChatGPT的出现，进入到大模型AIGC发展的阶段，开启了 AI 与人交互的新模式，使人们可以更容易地使用 AI 并从中受益。

![1733811105932](assets/1733811105932.png)


## PyTorch框架简介

### 什么是PyTorch

> PyTorch是一个基于Python的科学计算包
>
> PyTorch安装
>
> ```sh
> pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

- PyTorch一个基于Python语言的深度学习框架，它将数据封装成张量（Tensor）来进行处理。
- PyTorch提供了灵活且高效的工具，用于构建、训练和部署机器学习和深度学习模型。
- PyTorch广泛应用于学术研究和工业界，特别是在计算机视觉、自然语言处理、强化学习等领域。

### PyTorch特点

> **PyTorch与TensorFlow的比较**
>
> - **PyTorch与TensorFlow的区别**：PyTorch是基于动态图（动态计算图）的，而TensorFlow 1.x是基于静态计算图的（TensorFlow 2.x支持动态图）。这使得PyTorch在灵活性和调试方面优于TensorFlow，尤其是在研究和原型设计中。此外，PyTorch的API设计更加贴近Python，易于学习和使用。
> - **TensorFlow 2.x**（引入了Eager Execution）和PyTorch都支持动态图，但PyTorch因其更直观的编程模式和调试支持，在学术界和一些工业界应用中更为流行。

- **类似于NumPy的张量计算**
  - PyTorch中的基本数据结构是张量（Tensor），它与NumPy中的数组类似，但PyTorch的张量具有GPU加速的能力（通过CUDA），这使得深度学习模型能够高效地在GPU上运行。
- **自动微分系统**
  - PyTorch提供了强大的**自动微分**功能（`autograd`模块），能够自动计算模型中每个参数的梯度。
  - 自动微分使得梯度计算过程变得简洁和高效，并且支持复杂的模型和动态计算图。
- **深度学习库**
  - PyTorch提供了一个名为**torch.nn**的子模块，用于构建神经网络。它包括了大量的预构建的层（如全连接层、卷积层、循环神经网络层等），损失函数（如交叉熵、均方误差等），以及优化算法（如SGD、Adam等）。
  - `torch.nn.Module`是PyTorch中构建神经网络的基础类，用户可以通过继承该类来定义自己的神经网络架构。

- **动态计算图**
  - PyTorch使用动态计算图机制，允许在运行时构建和修改模型结构，具有更高的灵活性，适合于研究人员进行实验和模型调试。
- **GPU加速（CUDA支持）**
  - PyTorch提供对GPU的良好支持，能够在NVIDIA的CUDA设备上高效地进行计算。用户只需要将数据和模型转移到GPU上，PyTorch会自动优化计算过程。
  - 通过简单的`tensor.to(device)`方法，可以轻松地将模型和数据从CPU转移到GPU或从一个GPU转移到另一个GPU。
- **跨平台支持**
  - PyTorch支持在多种硬件平台（如CPU、GPU、TPU等）上运行，并且支持不同操作系统（如Linux、Windows、macOS）以及分布式计算环境（如多GPU、分布式训练）。

### PyTorch发展历史

- **Torch**

  Torch是最早的Torch框架，最初由Ronan Collobert、Clement Farabet等人开发。它是一个科学计算框架，提供了多维张量操作和科学计算工具。

- **Torch7**

  Torch7是Torch的一个后续版本，引入了Lua编程语言，并在深度学习领域取得了一定的成功。遗憾的是，随着pytorch的普及，Torch便不再维护，Torch7也就成为了Torch的最后一个版本。

- **Pytorch 0.1.0**

  在Torch的基础上，Facebook人工智能研究院（FAIR）于2016年发布了PyTorch的第一个版本，标志着PyTorch的正式诞生。

  初始版本的PyTorch主要基于Torch7，但引入了更加Pythonic的设计风格，使得深度学习模型的定义和调试更加直观和灵活。

- **Pytorch 0.2.0**

  该版本首次引入了动态图机制，使得用户能够在构建神经网络时更加灵活。作为Pytorch后期制胜tensorflow的关键机制，该版本象征着Pytorch进入了一个新的阶段。

- **Pytorch 1.0.0**

  2018年发布了Pytorch的首个稳定版本，引入了Eager模式简化了模型的构建和训练过程。

- **Pytorch 2.0**

  Pytorch2.0引入了torch.compile，可以支持对训练过程的加速，同时引入了TorchDynamo，主要替换torch.jit.trace和torch.jit.script。另外在这个版本中编译器性能大幅提升，分布式运行方面也做了一定的优化。

![1733816358935](assets/1733816358935.png)


## 张量创建

### 什么是张量

> 张量是PyTorch中的核心数据抽象

- PyTorch中的张量就是元素为同一种数据类型的多维矩阵，与NumPy数组类似。

- PyTorch中，张量以"类"的形式封装起来，对张量的一些运算、处理的方法（数值计算、矩阵操作、自动求导）被封装在类中。

![1733817329667](assets/1733817329667.png)

==多个二维张量组成三维张量==

![1733817373409](assets/1733817373409.png)

==多个三维张量组成四维张量==

==多个四维张量组成五维张量==

![1733817419397](assets/1733817419397.png)

### 基本创建方式

> - 张量的数据类型有
>
>   ![1662778155632](assets/1662778155632.png)
>
> - 张量中默认的数据类型是**float32(torch.FloatTensor)**

- torch.tensor(data=, dtype=) 根据指定数据创建张量

  ```python
  import torch  # 需要安装torch模块
  import numpy as np
  
  
  # 1. 创建张量标量
  data = torch.tensor(10)
  print(data)
  # 2. numpy 数组, 由于data为float64, 张量元素类型也是float64
  data = np.random.randn(2, 3)
  data = torch.tensor(data)
  print(data)
  # 3. 列表, 浮点类型默认float32
  data = [[10., 20., 30.], [40., 50., 60.]]
  data = torch.tensor(data)
  print(data)
  ```

- torch.Tensor(size=) 根据形状创建张量, 其也可用来创建指定数据的张量

  ```python
  # 1. 创建2行3列的张量, 默认 dtype 为 float32
  data = torch.Tensor(2, 3)
  print(data)
  # 2. 注意: 如果传递列表, 则创建包含指定元素的张量
  data = torch.Tensor([10])
  print(data)
  data = torch.Tensor([10, 20])
  print(data)
  ```

- torch.IntTensor()/torch.FloatTensor() 创建指定类型的张量

  ```python
  # 1. 创建2行3列, dtype 为 int32 的张量
  data = torch.IntTensor(2, 3)
  print(data)
  # 2. 注意: 如果传递的元素类型不正确, 则会进行类型转换
  data = torch.IntTensor([2.5, 3.3])
  print(data)
  # 3. 其他的类型
  data = torch.ShortTensor()  # int16
  data = torch.LongTensor()   # int64
  data = torch.FloatTensor()  # float32
  data = torch.DoubleTensor() # float64
  ```

### 线性和随机张量

- torch.arange(start=, end=, step=)：固定步长线性张量

- torch.linspace(start=, end=, steps=)：固定元素数线性张量

  ```python
  # 1. 在指定区间按照步长生成元素 [start, end, step) 左闭右开
  data = torch.arange(0, 10, 2)
  print(data)
  
  # 2. 在指定区间按照元素个数生成 [start, end, steps] 左闭右闭
  # step = (end-start) / (steps-1)
  # value_i = start + step * i
  data = torch.linspace(0, 9, 10)
  print(data)
  ```

- torch.randn/rand(size=) 创建随机浮点类型张量

- torch.randint(low=, high=, size=) 创建随机整数类型张量 左闭右开

- torch.initial_seed() 和 torch.manual_seed(seed=) 随机种子设置

  ```python
  # 1. 创建随机张量
  data = torch.randn(2, 3)  # 创建2行3列张量
  print(data)
  # 查看随机数种子
  print('随机数种子:', torch.initial_seed())
  
  # 2. 随机数种子设置 
  torch.manual_seed(100)
  data = torch.randn(2, 3)
  print(data)
  print('随机数种子:', torch.initial_seed())
  ```

### 指定值张量

- torch.zeros(size=) 和 torch.zeros_like(input=) 创建全0张量

  ```python
  # 1. 创建指定形状全0张量
  data = torch.zeros(2, 3)
  print(data)
  
  # 2. 根据张量形状创建全0张量
  data = torch.zeros_like(data)
  print(data)
  ```

- torch.ones(size=) 和 torch.ones_like(input=) 创建全1张量

  ```python
  # 1. 创建指定形状全1张量
  data = torch.ones(2, 3)
  print(data)
  
  # 2. 根据张量形状创建全1张量
  data = torch.ones_like(data)
  print(data)
  ```

- torch.full(size=, fill_value=) 和 torch.full_like(input=, fill_value=) 创建全为指定值张量

  ```python
  # 1. 创建指定形状指定值的张量
  data = torch.full([2, 3], 10)
  print(data)
  
  # 2. 根据张量形状创建指定值的张量
  data = torch.full_like(data, 20)
  print(data)
  ```

### 指定元素类型张量

- data.type(dtype=)

  ```python
  data = torch.full([2, 3], 10)
  print(data.dtype)
  # 将 data 元素类型转换为 float64 类型
  data = data.type(torch.DoubleTensor)
  print(data.dtype)
  # 转换为其他类型
  # data = data.type(torch.ShortTensor)
  # data = data.type(torch.IntTensor)
  # data = data.type(torch.LongTensor)
  # data = data.type(torch.FloatTensor)
  # data = data.type(dtype=torch.float16)
  ```

- data.half/float/double/short/int/long()

  ```python
  data = torch.full([2, 3], 10)
  print(data.dtype)
  # 将 data 元素类型转换为 float64 类型
  data = data.double()
  print(data.dtype)
  # 转换为其他类型
  # data = data.short()
  # data = data.int()
  # data = data.long()
  # data = data.float()
  ```


## 张量类型转换

### 张量转换为NumPy数组

- 使用 t.numpy() 函数可以将张量转换为 ndarray 数组，但是共享内存，可以使用copy函数避免共享。

  ```python
  # 1. 将张量转换为 numpy 数组
  data_tensor = torch.tensor([2, 3, 4])
  # 使用张量对象中的 numpy 函数进行转换
  data_numpy = data_tensor.numpy()
  print(type(data_tensor))
  print(type(data_numpy))
  # 注意: data_tensor 和 data_numpy 共享内存
  # 修改其中的一个，另外一个也会发生改变
  # data_tensor[0] = 100
  data_numpy[0] = 100
  print(data_tensor)
  print(data_numpy)
  
  # 2. 对象拷贝避免共享内存
  data_tensor = torch.tensor([2, 3, 4])
  # 使用张量对象中的 numpy 函数进行转换，通过copy方法拷贝对象
  data_numpy = data_tensor.numpy().copy()
  print(type(data_tensor))
  print(type(data_numpy))
  # 注意: data_tensor 和 data_numpy 此时不共享内存
  # 修改其中的一个，另外一个不会发生改变
  # data_tensor[0] = 100
  data_numpy[0] = 100
  print(data_tensor)
  print(data_numpy)
  ```

### NumPy数组转换为张量

- 使用 torch.from_numpy(ndarray=) 可以将ndarray数组转换为 tensor张量，默认共享内存，使用copy函数避免共享。

- 使用 torch.tensor(data=) 可以将ndarray数组转换为tensor张量，默认不共享内存。

  ```python
  data_numpy = np.array([2, 3, 4])
  # 将 numpy 数组转换为张量类型
  # 1. torch.from_numpy(ndarray)
  data_tensor = torch.from_numpy(data_numpy)
  # nunpy 和 tensor 共享内存
  # data_numpy[0] = 100
  data_tensor[0] = 100
  print(data_tensor)
  print(data_numpy)
  
  # 2. torch.tensor(ndarray)
  data_numpy = np.array([2, 3, 4])
  data_tensor = torch.tensor(data_numpy)
  # nunpy 和 tensor 不共享内存
  # data_numpy[0] = 100
  data_tensor[0] = 100
  print(data_tensor)
  print(data_numpy)
  ```

### 提取标量张量的数值

- 对于只有一个元素的张量，使用item()函数将该值从张量中提取出来

  ```python
  # 当张量只包含一个元素时, 可以通过 item() 函数提取出该值
  data = torch.tensor([30,])
  print(data.item())
  data = torch.tensor(30)
  print(data.item())
  ```


## 张量数值计算

### 基本运算

加减乘除取负号：

- +、-、*、/、-

- add(other=)、sub、mul、div、neg 

- `add_(other=)`、`sub_`、`mul_`、`div_`、`neg_`（其中带下划线的版本会修改原数据）

  ```python
  data = torch.randint(0, 10, [2, 3])
  print(data)
  # 1. 不修改原数据
  new_data = data.add(10)  # 等价 new_data = data + 10
  print(new_data)
  # 2. 直接修改原数据 注意: 带下划线的函数为修改原数据本身
  data.add_(10)  # 等价 data += 10
  print(data)
  # 3. 其他函数
  print(data.sub(100))
  print(data.mul(100))
  print(data.div(100))
  print(data.neg())
  ```

### 点乘运算

> 点乘（Hadamard）也称为元素级乘积，指的是相同形状的张量对应位置的元素相乘，使用mul和运算符 * 实现。
>
> ![1733825192893](assets/1733825192893.png)

```python
data1 = torch.tensor([[1, 2], [3, 4]])
data2 = torch.tensor([[5, 6], [7, 8]])
# 第一种方式
data = torch.mul(data1, data2)
print(data)
# 第二种方式
data = data1 * data2
print(data)
```

###  矩阵乘法运算

> 矩阵乘法运算要求第一个矩阵 shape: (n, m)，第二个矩阵 shape: (m, p), 两个矩阵点积运算 shape 为: (n, p)。

- 运算符 @ 用于进行两个矩阵的乘积运算

- torch.matmul(input=, other=) 对进行乘积运算的两矩阵形状没有限定。对于输入shape不同的张量, 对应的最后几个维度必须符合矩阵运算规则

  ```python
  # 点积运算
  data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
  data2 = torch.tensor([[5, 6], [7, 8]])
  # 方式一:
  data3 = data1 @ data2
  print("data3-->", data3)
  # 方式二:
  data4 = torch.matmul(data1, data2)
  print("data4-->", data4)
  ```


## 张量运算函数

- `tensor.mean(dim=)`:平均值

- `tensor.sum(dim=)`:求和

- `tensor.min/max(dim=)`:最小值/最大值

- `tensor.pow(exponent=)`:幂次方 $$x^n$$

- `tensor.sqrt(dim=)`:平方根

- `tensor.exp()`:指数 $$e^x$$

- `tensor.log(dim=)`:对数 以e为底

- dim=0按列计算,dim=1按行计算

  ```python
  import torch
  
  
  data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
  print(data)
  # 1. 计算均值
  # 注意: tensor 必须为 Float 或者 Double 类型
  print(data.mean())
  print(data.mean(dim=0))  # 按列计算均值
  print(data.mean(dim=1))  # 按行计算均值
  # 2. 计算总和
  print(data.sum())
  print(data.sum(dim=0))
  print(data.sum(dim=1))
  # 3. 计算平方
  print(torch.pow(data，2))
  # 4. 计算平方根
  print(data.sqrt())
  # 5. 指数计算, e^n 次方
  print(data.exp())
  # 6. 对数计算
  print(data.log())  # 以 e 为底
  print(data.log2())
  print(data.log10())
  ```


---

# 附二 · 课程课件

> 来源：配套 PPT 的文字提取。图表、公式和代码截图不在其中，需要看原图请对照课件。


## 00-深度学习简介（第 1–10 页）· 深度学习简介

**[第 1 页]**

    深度学习简介

**[第 2 页]**

    知道什么是深度学习
    知道深度学习应用场景
    了解深度学习发展史

**[第 3 页]**

    什么是深度学习
    机器学习是实现人工智能的一种途径。
    深度学习是机器学习的一个子集，也就是说深度学习是实现机器学习的一种方法。 
    深度学习是机器学习中一种基于对数据进行特征学习的算法。
    深度学习是基于人工神经网络，深度是指网络中使用多层，每层都通过非线性变换处理数据，并逐渐提取出更复杂、更抽象的特征。

**[第 4 页]**

    什么是深度学习
    传统机器学习算法依赖人工设计特征，并进行特征提取；而深度学习方法不需要人工，而是依赖算法自动提取特征。
    深度学习通过模仿人脑的神经网络来处理和分析复杂的数据，从大量数据中自动提取复杂特征。这也是深度学习被看做黑盒子，可解释性差的原因。 
    深度学习尤其擅长处理高维数据，如图像、语音和文本。

**[第 5 页]**

    深度学习特点
    多层非线性变换：深度学习模型由多个层次组成，每一层都应用非线性激活函数对输入数据进行变换。较低的层级通常捕捉到简单的特征（如边缘、颜色等），而更高的层级则可以识别更复杂的模式（如物体或面部识别）。
    
    自动特征提取：与传统机器学习算法不同，深度学习能够自动从原始数据中学习到有用的特征，而不需要人工特征工程。这使得深度学习在许多领域中表现出色。
    
    大数据和计算能力：深度学习模型通常需要大量的标注数据和强大的计算资源（如GPU）来进行训练。大数据和高性能计算使得深度学习在图像识别、自然语言处理等领域取得了显著突破。
    
    可解释性差：深度学习模型内部的运作机制相对不透明，被称为“黑箱”，这意味着理解模型为什么做出特定决策可能会比较困难。这对某些应用场景来说是一个挑战。

**[第 6 页]**

    常见的深度学习模型
    卷积神经网络 (Convolutional Neural Networks, CNN)
    循环神经网络 (Recurrent Neural Networks, RNN)
    自编码器 (Autoencoders)
    生成对抗网络 (Generative Adversarial Networks, GAN)
    Transformer
    深度强化学习（Deep Reinforcement Learning, DRL）
    图神经网络（GNN, Graph Neural Network）

**[第 7 页]**

    应用场景
    计算机视觉
    自然语言处理
    自动驾驶
    推荐系统
    医疗健康
    工业与制造业
    语音与音频处理
    ...

**[第 8 页]**

    人工智能发展历史
    专家系统占主导
    1950：图灵设计国际象棋程序
    1962：IBM Arthur Samuel 的跳棋程序战胜人类高手（人工智能第一次浪潮）
    主要用统计模型解决问题
    1993：Vapnik提出SVM
    1997：IBM 深蓝战胜卡斯帕罗夫（人工智能第二次浪潮）
    神经网络、深度学习流派
    2012： AlexNet深度学习的开山之作
    2016：Google AlphaGO 战胜李世石（人工智能第三次浪潮）
    大规模预训练模型
    2017-至今
    神经网络
    21世纪初期
    统计主义
    20世纪80-2000
    符号主义 
    20世纪50-70
    大规模预训练模型
    2017：自然语言处理NLP的Transformer框架出现
    2018：Bert和GPT的出现
    2022：chatGPT的出现，进入到大模型AIGC发展的阶段

**[第 9 页]**

    深度学习发展历史
    20世纪40年代：沃尔特·皮茨（Walter Pitts）和沃伦·麦卡洛克（Warren McCulloch）等开始模仿生物神经系统来构建计算模型，如McCulloch-Pitts神经元。
    1957：弗兰克·罗森布拉特（Frank Rosenblatt）提出感知器概念，能够进行简单的二分类任务
    1960年代末：出现了多层感知器（MLP），但当时由于计算能力和数据量的限制，这些模型的应用受到很大限制。
    1986：David Rumelhart等人发表了关于反向传播（Backpropagation, BP）算法的研究成果，使得多层神经网络能够通过梯度下降优化参数，解决复杂的非线性问题。
    由于计算资源的限制以及对复杂数据（如图像和语音）的处理能力较弱，深度学习未能广泛应用
    2006：杰弗里·辛顿（Geoffrey Hinton）和其团队提出了深度置信网络（DBN），标志着深度学习的复兴。
    2012：亚历克斯·克里泽夫斯基（Alex Krizhevsky）等人设计的卷积神经网络（CNN）AlexNet在ImageNet图像识别挑战赛中取得巨大成功。
    爆发期
    2016-至今
    复兴与突破
    2000s-2010s
    挑战与瓶颈
    1980s-1990s
    早期探索 
    1940s-1980s
    2016：Google AlphaGO 战胜李世石（人工智能第三次浪潮）
    2017：自然语言处理NLP的Transformer框架出现
    2018：Bert和GPT的出现
    2022：chatGPT的出现，进入到大模型AIGC发展的阶段

**[第 10 页]**

    深度学习与机器学习的关系和差别？
    深度学习特点
    常见的深度学习模型
    深度学习的应用场景


## 01-PyTorch基本使用（第 1–48 页）· PyTorch 简介与张量基础

**[第 1 页]**

    PyTorch框架

**[第 2 页]**

    什么是PyTorch
    PyTorch一个基于Python语言的深度学习框架，它将数据封装成张量（Tensor）来进行处理。
    PyTorch提供了灵活且高效的工具，用于构建、训练和部署机器学习和深度学习模型。
    PyTorch广泛应用于学术研究和工业界，特别是在计算机视觉、自然语言处理、强化学习等领域。
    
    
    
    
    
    
    PyTorch的安装：
    pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple

**[第 3 页]**

    PyTorch特点
    类似于NumPy的张量计算
    自动微分系统
    深度学习库
    动态计算图
    GPU加速（CUDA支持）
    支持多种应用场景
    跨平台支持
    ...

**[第 4 页]**

    PyTorch发展历史
    2016年Facebook正式发布了PyTorch的第一个版本。
    2018年PyTorch发布了1.0版本，标志着其正式进入生产级应用阶段。

**[第 5 页]**

    张量的创建
    张量的类型转换
    张量数值计算
    张量运算函数
    张量索引操作
    张量形状操作
    张量拼接操作
    自动微分模块
    案例-线性回归案例

**[第 6 页]**

    掌握张量创建方法
    知道线性和随机张量的创建方法
    知道0-1张量的创建方法
    知道张量元素类型的转换方法

**[第 7 页]**

    什么是张量
    PyTorch中的张量就是元素为同一种数据类型的多维矩阵。在PyTorch中，张量以 "类" 的形式封装起来，对张量的一些运算、处理的方法被封装在类中。
    PyTorch张量与NumPy数组类似，但PyTorch的张量具有GPU加速的能力（通过CUDA），这使得深度学习模型能够高效地在GPU上运行。
    PyTorch提供了对张量的强大支持，可以进行高效的数值计算、矩阵操作、自动求导等。
    张量是 PyTorch 中的核心数据抽象，PyTorch 支持各种张量子类型。通常地，一维张量称为向量/矢量（vector），二维张量称为矩阵（matrix）。

**[第 8 页]**

    张量基本创建方式
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量, 其也可用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

**[第 9 页]**

    基本创建方式
    1、torch.tensor() 根据指定数据创建张量
    import torch  # 需要安装torch模块
    import numpy as np
    # 1. 创建张量标量
    data = torch.tensor(10)
    print(data)
    # 2. numpy 数组, 由于 data 为 float64, 下面代码也使用该类型
    data = np.random.randn(2, 3)
    data = torch.tensor(data)
    print(data)
    # 3. 列表, 下面代码使用默认元素类型 float32
    data = [[10., 20., 30.], [40., 50., 60.]]
    data = torch.tensor(data)
    print(data)
    输出结果:
    tensor(10)
    tensor([[ 0.1345,  0.1149,  0.2435],
            [ 0.8026, -0.6744, -1.0918]], dtype=torch.float64)
    tensor([[10., 20., 30.],
            [40., 50., 60.]])

**[第 10 页]**

    基本创建方式
    2.torch.Tensor() 根据指定形状创建张量，也可以用来创建指定数据的张量
    # 1. 创建2行3列的张量, 默认 dtype 为 float32
    data = torch.Tensor(2, 3)
    print(data)
    # 2. 注意: 如果传递列表, 则创建包含指定元素的张量
    data = torch.Tensor([10])
    print(data)
    data = torch.Tensor([10, 20])
    print(data)
    输出结果:
    tensor([[0.0000e+00, 3.6893e+19, 2.2018e+05],
            [4.6577e-10, 2.4158e-12, 1.1625e+33]])
    tensor([10.])
    tensor([10., 20.])

**[第 11 页]**

    基本创建方式
    3、torch.IntTensor()、torch.FloatTensor()、torch.DoubleTensor() 创建指定类型的张量
    # 1. 创建2行3列, dtype 为 int32 的张量
    data = torch.IntTensor(2, 3)
    print(data)
    # 2. 注意: 如果传递的元素类型不正确, 则会进行类型转换
    data = torch.IntTensor([2.5, 3.3])
    print(data)
    # 3. 其他的类型
    data = torch.ShortTensor()  # int16
    data = torch.LongTensor()   # int64
    data = torch.FloatTensor()  # float32
    data = torch.DoubleTensor() # float64
    输出结果:
    tensor([[         0, 1610612736, 1213662609],
            [ 805308409,  156041223,          1]], dtype=torch.int32)
    tensor([2, 3], dtype=torch.int32)

**[第 12 页]**

    创建线性和随机张量
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置
    torch.rand/randn() 创建随机浮点类型张量
    torch.randint(low, high, size=()) 创建随机整数类型张量

**[第 13 页]**

    创建线性和随机张量
    1、torch.arange()、torch.linspace() 创建线性张量
    # 1. 在指定区间按照步长生成元素 [start, end, step)
    data = torch.arange(0, 10, 2)
    print(data)
    
    # 2. 在指定区间按照元素个数生成 [start, end, steps]
    data = torch.linspace(0, 9, 10)
    print(data)
    输出结果:
    tensor([0, 2, 4, 6, 8])
    tensor([ 0.0000,  1.2222,  2.4444,  3.6667,  4.8889,  6.1111,  7.3333,  8.5556,
             9.7778, 11.0000])

**[第 14 页]**

    创建线性和随机张量
    2、torch.initial_seed()、torch.manual_seed() 随机数种子设置，torch.randn() 创建随机浮点类型张量
    # 1. 创建随机张量
    data = torch.randn(2, 3)  # 创建2行3列张量
    print(data)
    # 查看随机数种子
    print('随机数种子:', torch.initial_seed())
    
    # 2. 随机数种子设置 
    torch.manual_seed(100)
    data = torch.randn(2, 3)
    print(data)
    print('随机数种子:', torch.initial_seed())
    输出结果:
    tensor([[-0.5209, -0.2439, -1.1780],
            [ 0.8133,  1.1442,  0.6790]])
    随机数种子: 4508475192273306739
    tensor([[ 0.3607, -0.2859, -0.3938],
            [ 0.2429, -1.3833, -2.3134]])
    随机数种子: 100

**[第 15 页]**

    创建0、1、指定值张量
    torch.ones 和 torch.ones_like 创建全1张量
    torch.zeros 和 torch.zeros_like 创建全0张量
    torch.full 和 torch.full_like 创建全为指定值张量

**[第 16 页]**

    创建0、1、指定值张量
    1、torch.zeros()、torch.zeros_like() 创建全0张量
    # 1. 创建指定形状全0张量
    data = torch.zeros(2, 3)
    print(data)
    
    # 2. 根据张量形状创建全0张量
    data = torch.zeros_like(data)
    print(data)
    输出结果:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])
    tensor([[0., 0., 0.],
            [0., 0., 0.]])

**[第 17 页]**

    创建0、1、指定值张量
    2、torch.ones()、torch.ones_like() 创建全1张量
    # 1. 创建指定形状全1张量
    data = torch.ones(2, 3)
    print(data)
    
    # 2. 根据张量形状创建全1张量
    data = torch.ones_like(data)
    print(data)
    输出结果:
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    tensor([[1., 1., 1.],
            [1., 1., 1.]])

**[第 18 页]**

    创建0、1、指定值张量
    3、torch.full()、torch.full_like() 创建全为指定值张量
    # 1. 创建指定形状指定值的张量
    data = torch.full([2, 3], 10)
    print(data)
    
    # 2. 根据张量形状创建指定值的张量
    data = torch.full_like(data, 20)
    print(data)
    输出结果:
    tensor([[10, 10, 10],
            [10, 10, 10]])
    tensor([[20, 20, 20],
            [20, 20, 20]])

**[第 19 页]**

    张量元素类型转换
    data.type(torch.DoubleTensor)
    data.half/double/float/short/int/long()

**[第 20 页]**

    张量元素类型转换
    1、data.type(torch.DoubleTensor)
    data = torch.full([2, 3], 10)
    print(data.dtype)
    # 将 data 元素类型转换为 float64 类型
    data = data.type(torch.DoubleTensor)
    print(data.dtype)
    # 转换为其他类型
    # data = data.type(torch.ShortTensor)
    # data = data.type(torch.IntTensor)
    # data = data.type(torch.LongTensor)
    # data = data.type(torch.FloatTensor)
    # data = data.type(dtype=torch.float16)
    输出结果:
    torch.int64
    torch.float64

**[第 21 页]**

    张量元素类型转换
    2、data.half/double/float/short/int/long()
    data = torch.full([2, 3], 10)
    print(data.dtype)
    # 将 data 元素类型转换为 float64 类型
    data = data.double()
    print(data.dtype)
    # 转换为其他类型
    # data = data.short()
    # data = data.int()
    # data = data.long()
    # data = data.float()
    输出结果:
    torch.int64
    torch.float64

**[第 22 页]**

    创建张量的方式
    torch.tensor() 根据指定数据创建张量
    torch.Tensor() 根据形状创建张量, 其也可用来创建指定数据的张量
    torch.IntTensor()、torch.FloatTensor()、torch.DoubleTensor() 创建指定类型的张量
    创建线性和随机张量
    torch.arange() 和 torch.linspace() 创建线性张量
    torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置
    torch.randn() 创建随机张量
    创建0、1、指定值张量
    torch.ones() 和 torch.ones_like() 创建全1张量
    torch.zeros() 和 torch.zeros_like() 创建全0张量
    torch.full() 和 torch.full_like() 创建全为指定值张量
    张量元素类型转换
    data.type(dtype=) 
    data.half/double/float/short/int/long()

**[第 23 页]**

    张量的创建
    张量的类型转换
    张量数值计算
    张量运算函数
    张量索引操作
    张量形状操作
    张量拼接操作
    自动微分模块
    案例-线性回归案例

**[第 24 页]**

    掌握张量转换为Numpy数组的方法
    掌握Numpy数组转换为张量的方法
    掌握标量张量和数字转换方法

**[第 25 页]**

    张量转换为NumPy数组
    使用 Tensor.numpy 函数可以将张量转换为 ndarray 数组，但是共享内存，可以使用 copy 函数避免共享。

**[第 26 页]**

    张量转换为NumPy数组
    使用Tensor.numpy()函数可以将张量转换为ndarray数组，但是共享内存，可以使用copy()函数避免共享
    # 1. 将张量转换为 numpy 数组
    data_tensor = torch.tensor([2, 3, 4])
    # 使用张量对象中的 numpy 函数进行转换
    data_numpy = data_tensor.numpy()
    print(type(data_tensor))
    print(type(data_numpy))
    # 注意: data_tensor 和 data_numpy 共享内存
    # 修改其中的一个，另外一个也会发生改变
    # data_tensor[0] = 100
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)
    输出结果:
    <class 'torch.Tensor'>
    <class 'numpy.ndarray'>
    tensor([100,   3,   4])
    [100   3   4]

**[第 27 页]**

    张量转换为NumPy数组
    使用Tensor.numpy()函数可以将张量转换为ndarray数组，但是共享内存，可以使用copy()函数避免共享
    # 2. 对象拷贝避免共享内存
    data_tensor = torch.tensor([2, 3, 4])
    # 使用张量对象中的 numpy 函数进行转换，通过copy方法拷贝对象
    data_numpy = data_tensor.numpy().copy()
    print(type(data_tensor))
    print(type(data_numpy))
    # 注意: data_tensor 和 data_numpy 此时不共享内存
    # 修改其中的一个，另外一个不会发生改变
    # data_tensor[0] = 100
    data_numpy[0] = 100
    print(data_tensor)
    print(data_numpy)
    输出结果:
    <class 'torch.Tensor'>
    <class 'numpy.ndarray'>
    tensor([2, 3, 4])
    [100   3   4]

**[第 28 页]**

    NumPy数组转换为张量
    使用 from_numpy 可以将 ndarray 数组转换为 Tensor，默认共享内存，使用 copy 函数避免共享。
    使用 torch.tensor 可以将 ndarray 数组转换为 Tensor，默认不共享内存。

**[第 29 页]**

    NumPy数组转换为张量
    使用from_numpy()可以将ndarray数组转换为Tensor，共享内存，使用copy()函数避免共享
    data_numpy = np.array([2, 3, 4])
    # 将 numpy 数组转换为张量类型
    # 1. torch.from_numpy(ndarray)
    data_tensor = torch.from_numpy(data_numpy)
    # nunpy 和 tensor 共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100
    print(data_tensor)
    print(data_numpy)
    输出结果:
    tensor([100,   3,   4], dtype=torch.int32)
    [100   3   4]

**[第 30 页]**

    NumPy数组转换为张量
    使用torch.tensor()可以将ndarray数组转换为Tensor，不共享内存。
    # 2. torch.tensor(ndarray)
    data_numpy = np.array([2, 3, 4])
    data_tensor = torch.tensor(data_numpy)
    # numpy 和 tensor 不共享内存
    # data_numpy[0] = 100
    data_tensor[0] = 100
    print(data_tensor)
    print(data_numpy)
    输出结果:
    tensor([100,   3,   4], dtype=torch.int32)
    [2 3 4]

**[第 31 页]**

    标量张量和数字转换
    对于只有一个元素的张量，使用item()函数将该值从张量中提取出来
    # 当张量只包含一个元素时, 可以通过 item() 函数提取出该值
    data = torch.tensor([30,])
    print(data.item())
    data = torch.tensor(30)
    print(data.item())
    输出结果:
    30
    30

**[第 32 页]**

    张量转换为 numpy 数组
    data_tensor.numpy()
    data_tensor.numpy().copy()
     2. numpy 转换为张量
    torch.from_numpy(data_numpy)
    torch.tensor(data_numpy)
    3.标量张量和数字转换
    tensor.item()

**[第 33 页]**

    张量的创建
    张量的类型转换
    张量数值计算
    张量运算函数
    张量索引操作
    张量形状操作
    张量拼接操作
    自动微分模块
    案例-线性回归案例

**[第 34 页]**

    掌握张量基本运算
    掌握张量点乘运算
    掌握张量矩阵乘法运算

**[第 35 页]**

    张量基本运算
    加减乘除取负号：
    add、sub、mul、div、neg 
    add_、sub_、mul_、div_、neg_（其中带下划线的版本会修改原数据）

**[第 36 页]**

    张量基本运算
    data = torch.randint(0, 10, [2, 3])
    print(data)
    # 1. 不修改原数据
    new_data = data.add(10)  # 等价 new_data = data + 10
    print(new_data)
    # 2. 直接修改原数据 注意: 带下划线的函数为修改原数据本身
    data.add_(10)  # 等价 data += 10
    print(data)
    # 3. 其他函数
    print(data.sub(100))
    print(data.mul(100))
    print(data.div(100))
    print(data.neg())
    输出结果:
    tensor([[3, 7, 4],
            [0, 0, 6]])
    tensor([[13, 17, 14],
            [10, 10, 16]])
    tensor([[13, 17, 14],
            [10, 10, 16]])
    tensor([[-87, -83, -86],
            [-90, -90, -84]])
    tensor([[1300, 1700, 1400],
            [1000, 1000, 1600]])
    tensor([[0.1300, 0.1700, 0.1400],
            [0.1000, 0.1000, 0.1600]])
    tensor([[-13, -17, -14],
            [-10, -10, -16]])

**[第 37 页]**

    点乘运算
    点乘指（Hadamard）的是相同形状的张量对应位置的元素相乘，使用mul 和运算符 * 实现。

**[第 38 页]**

    点乘运算
    data1 = torch.tensor([[1, 2], [3, 4]])
    data2 = torch.tensor([[5, 6], [7, 8]])
    # 第一种方式
    data = torch.mul(data1, data2)
    print(data)
    # 第二种方式
    data = data1 * data2
    print(data)
    输出结果:
    tensor([[ 5, 12],
            [21, 32]])
    tensor([[ 5, 12],
            [21, 32]])

**[第 39 页]**

    矩阵乘法运算
    矩阵乘法运算要求第一个矩阵 shape: (n, m)，第二个矩阵 shape: (m, p), 两个矩阵点积运算 shape 为: (n, p)。
    运算符 @ 用于进行两个矩阵的乘积运算
    torch.matmul 对进行乘积运算的两矩阵形状没有限定。对于输入的 shape 不同的张量, 对应的最后几个维度必须符合矩阵运算规则

**[第 40 页]**

    矩阵乘法运算
    # 点积运算
    data1 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    data2 = torch.tensor([[5, 6], [7, 8]])
    # 方式一:
    data3 = data1 @ data2
    print("data3-->", data3)
    # 方式二:
    data4 = torch.matmul(data1, data2)
    print("data4-->", data4)
    输出结果:
    data3--> tensor([[19, 22],
            [43, 50],
            [67, 78]])
    data4--> tensor([[19, 22],
            [43, 50],
            [67, 78]])

**[第 41 页]**

    1. 张量基本运算函数
    add、sub、mul、div、neg等函数
     add_、sub_、mul_、div_、neg_等函数
    2. 张量的点乘运算
    mul和运算符*
    3. 矩阵乘法运算
    运算符@用于进行两个矩阵乘法运算
    torch.matmul 对进行点乘运算的两矩阵形状没有限定，对数输入的 shape 不同的张量, 对应的最后几个维度必须符合矩阵运算规则

**[第 42 页]**

    张量的创建
    张量的类型转换
    张量数值计算
    张量运算函数
    张量索引操作
    张量形状操作
    张量拼接操作
    自动微分模块
    案例-线性回归案例

**[第 43 页]**

    掌握张量相关的运算函数

**[第 44 页]**

    常见运算函数
    PyTorch 为每个张量封装很多实用的计算函数：
    均值
    平方根
    求和
    指数计算
    对数计算等等

**[第 45 页]**

    常见运算函数
    import torch
    
    data = torch.randint(0, 10, [2, 3], dtype=torch.float64)
    print(data)
    # 1. 计算均值
    # 注意: tensor 必须为 Float 或者 Double 类型
    print(data.mean())
    print(data.mean(dim=0))  # 按列计算均值
    print(data.mean(dim=1))  # 按行计算均值
    # 2. 计算总和
    print(data.sum())
    print(data.sum(dim=0))
    print(data.sum(dim=1))
    # 3. 计算平方
    print(torch.pow(data，2))
    输出结果:
    tensor([[4., 0., 7.],
            [6., 3., 5.]], dtype=torch.float64)
    
    tensor(4.1667, dtype=torch.float64)
    tensor([5.0000, 1.5000, 6.0000], dtype=torch.float64)
    tensor([3.6667, 4.6667], dtype=torch.float64)
    
    tensor(25., dtype=torch.float64)
    tensor([10.,  3., 12.], dtype=torch.float64)
    tensor([11., 14.], dtype=torch.float64)
    
    tensor([[16.,  0., 49.],
            [36.,  9., 25.]], dtype=torch.float64)

**[第 46 页]**

    常见运算函数
    # 4. 计算平方根
    print(data.sqrt())
    
    # 5. 指数计算, e^n 次方
    print(data.exp())
    
    # 6. 对数计算
    print(data.log())  # 以 e 为底
    print(data.log2())
    print(data.log10())
    输出结果:
    tensor([[2.0000, 0.0000, 2.6458],
            [2.4495, 1.7321, 2.2361]], dtype=torch.float64)
    
    tensor([[5.4598e+01, 1.0000e+00, 1.0966e+03],
            [4.0343e+02, 2.0086e+01, 1.4841e+02]], dtype=torch.float64)
    
    tensor([[1.3863,   -inf, 1.9459],
            [1.7918, 1.0986, 1.6094]], dtype=torch.float64)
    tensor([[2.0000,   -inf, 2.8074],
            [2.5850, 1.5850, 2.3219]], dtype=torch.float64)
    tensor([[0.6021,   -inf, 0.8451],
            [0.7782, 0.4771, 0.6990]], dtype=torch.float64)

**[第 47 页]**

    张量运算函数
     sum,mean,sqrt,pow,exp,log等

**[第 48 页]**

    张量的创建
    张量的类型转换
    张量数值计算
    张量运算函数
    张量索引操作
    张量形状操作
    张量拼接操作
    自动微分模块
    案例-线性回归案例
