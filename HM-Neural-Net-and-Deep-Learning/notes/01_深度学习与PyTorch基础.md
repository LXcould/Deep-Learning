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
