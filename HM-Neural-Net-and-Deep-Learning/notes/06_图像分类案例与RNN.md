# 06 图像分类案例与 RNN

> **对应课程**：黑马《神经网络与深度学习》Day07（分P 92–108）
> **覆盖内容**：CIFAR10 图像分类完整案例 · RNN 循环神经网络原理 · 词嵌入层 · 循环网络层 · 歌词生成案例
> **整理日期**：2026-10-15

---

## 1 图像分类案例

### 1.1 CIFAR10数据集介绍

- cifar数据是torchvision第三方包提供的数据集

- 训练集5w 测试集1w

- y标签 10个类别  10分类问题

- 一张图形状 (32, 32, 3)

  ```python
  import torch
  import torch.nn as nn
  from torchvision.datasets import CIFAR10
  from torchvision.transforms import ToTensor
  import torch.optim as optim
  from torch.utils.data import DataLoader
  import time
  import matplotlib.pyplot as plt
  from torchsummary import summary
  
  # 每批次样本数
  BATCH_SIZE = 8
  
  
  # todo: 1-加载数据集转换成张量数据集
  def create_dataset():
  	# root: 文件夹所在目录路径
  	# train: 是否加载训练集
  	# ToTensor(): 将图片数据转换成张量数据
  	train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor())
  	valid_dataset = CIFAR10(root='./data', train=False, transform=ToTensor())
  	return train_dataset, valid_dataset
  
  
  # todo: 2-构建卷积神经网络分类模型
  # todo: 3-模型训练
  # todo: 4-模型评估
  if __name__ == '__main__':
  	train_dataset, valid_dataset = create_dataset()
  	print('图片类别对应关系->', train_dataset.class_to_idx)
  	print('train_dataset->', train_dataset.data[0])
  	print('train_dataset.data.shape->', train_dataset.data.shape)
  	print('valid_dataset.data.shape->', valid_dataset.data.shape)
  	print('train_dataset.targets->', train_dataset.targets[0])
  	# 图像展示
  	plt.figure(figsize=(2, 2))
  	plt.imshow(train_dataset.data[1])
  	plt.title(train_dataset.targets[1])
  	plt.show()
  ```

### 1.2 构建分类神经网络模型

```python
# todo: 2-构建卷积神经网络分类模型
class ImageModel(nn.Module):
	# todo:2-1 构建init构造函数, 实现搭建神经网络
	def __init__(self):
		super().__init__()
		# 第1层卷积层
		# 输入通道3 一张RGB图像就是3通道
		# 输出通道6 6个神经元提取出6张特征图
		# 卷积核大小3
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
		# 第1层池化层
		# 窗口大小2*2
		# 步长2
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		# 第2层卷积层
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
		# 第2层池化层
		# 池化层输出的特征图 16*6*6
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		# 第1层隐藏层
		# in_features: 将最后池化层的16*6*6三维矩阵转换为一维矩阵
		# 一维矩阵就是池化层这图像
		self.linear1 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
		# 第2层隐藏层
		self.linear2 = nn.Linear(in_features=120, out_features=84)
		# 输出层
		# out_features: 10, 10分类问题
		self.out = nn.Linear(in_features=84, out_features=10)

	# todo:2-2 构建forward函数, 实现前向传播
	def forward(self, x):
		# 第1层 卷积+激活+池化 计算
		x = self.pool1(torch.relu(self.conv1(x)))
		# 第2层 卷积+激活+池化 计算
		# x->(8, 16, 6, 6) 8个样本, 每个样本是16*6*6
		x = self.pool2(torch.relu(self.conv2(x)))
		# 第1层隐藏层  只能接收二维数据集
		# 四维数据集转换成二维数据集
		# x.shape[0]: 每批样本数, 最后一批可能不够8个, 所以不是写死8
		# -1*8=8*16*6*6 -1=16*6*6=576
		x = x.reshape(shape=(x.shape[0], -1))
		x = torch.relu(self.linear1(x))
		# 第2层隐藏层
		x = torch.relu(self.linear2(x))
		# 输出层 没有使用softmax激活函数, 后续多分类交叉熵损失函数会自动进行softmax激活
		x = self.out(x)
		return x
	# todo: 3-模型训练


# todo: 4-模型评估
if __name__ == '__main__':
	train_dataset, valid_dataset = create_dataset()
	# print('图片类别对应关系->', train_dataset.class_to_idx)
	# print('train_dataset->', train_dataset.data[0])
	# print('train_dataset.data.shape->', train_dataset.data.shape)
	# print('valid_dataset.data.shape->', valid_dataset.data.shape)
	# print('train_dataset.targets->', train_dataset.targets[0])
	# # 图像展示
	# plt.figure(figsize=(2, 2))
	# plt.imshow(train_dataset.data[1])
	# plt.title(train_dataset.targets[1])
	# plt.show()
	model = ImageModel()
	summary(model, input_size=(3, 32, 32))
```

### 1.3 模型训练

```python
# todo: 3-模型训练
def train(train_dataset):
	# 创建数据加载器
	dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	# 创建模型对象
	model = ImageModel()
	# model.to(device='cuda')
	# 创建损失函数对象
	criterion = nn.CrossEntropyLoss()
	# 创建优化器对象
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	# 循环遍历epoch
	# 定义epoch变量
	epoch = 10
	for epoch_idx in range(epoch):
		# 定义总损失变量
		total_loss = 0.0
		# 定义预测正确样本个数变量
		total_correct = 0
		# 定义总样本数据变量
		total_samples = 0
		# 定义开始时间变量
		start = time.time()
		# 循环遍历数据加载器 min-batch
		for x, y in dataloader:
			# print('y->', y)
			# 切换训练模式
			model.train()
			# 模型预测y
			output = model(x)
			# print('output->', output)
			# 计算损失值 平均损失值
			loss = criterion(output, y)
			# print('loss->', loss)
			# 梯度清零
			optimizer.zero_grad()
			# 梯度计算
			loss.backward()
			# 参数更新
			optimizer.step()
			# 统计预测正确的样本个数
			# tensor([9, 9, 9, 9, 9, 9, 9, 9])
			# print(torch.argmax(output, dim=-1))
			# tensor([False, False, False, False, False, False, False, False])
			# print(torch.argmax(output, dim=-1) == y)
			# tensor(0)
			# print((torch.argmax(output, dim=-1) == y).sum())
			total_correct += (torch.argmax(output, dim=-1) == y).sum()
			# 统计当前批次的总损失值
			# loss.item(): 当前批次平均损失值
			total_loss += loss.item() * len(y)
			# 统计当前批次的样本数
			total_samples += len(y)
		end = time.time()
		print('epoch:%2s loss:%.5f acc:%.2f time:%.2fs' % (
			epoch_idx + 1, total_loss / total_samples, total_correct / total_samples, end - start))
	# 保存训练模型
	torch.save(obj=model.state_dict(), f='model/imagemodel.pth')


# todo: 4-模型评估
if __name__ == '__main__':
	train_dataset, valid_dataset = create_dataset()
	# 模型训练
	train(train_dataset)
```

### 1.4 模型评估

```python
# todo: 4-模型评估
def test(valid_dataset):
	# 创建测试集数据加载器
	dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
	# 创建模型对象, 加载训练模型参数
	model = ImageModel()
	model.load_state_dict(torch.load('model/imagemodel.pth'))
	# 定义统计预测正确样本个数变量 总样本数据变量
	total_correct = 0
	total_samples = 0
	# 遍历数据加载器
	for x, y in dataloader:
		# 切换推理模型
		model.eval()
		# 模型预测
		output = model(x)
		# 将预测分值转成类别
		y_pred = torch.argmax(output, dim=-1)
		print('y_pred->', y_pred)
		# 统计预测正确的样本个数
		total_correct += (y_pred == y).sum()
		# 统计总样本数
		total_samples += len(y)

	# 打印精度
	print('Acc: %.2f' % (total_correct / total_samples))

if __name__ == '__main__':
	train_dataset, valid_dataset = create_dataset()
	# 模型训练
	# train(train_dataset)
	# 模型预测
	test(valid_dataset)
```

### 1.5 网络性能优化

- 增加卷积层的卷积核数据量

- 增加全连接层神经元数量

- 减小学习率

- 增加dropout随机失活层

  ```python
  class ImageClassification(nn.Module):
  	def __init__(self):
  		super(ImageClassification, self).__init__()
  		self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=3)
  		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
  		self.conv2 = nn.Conv2d(32, 128, stride=1, kernel_size=3)
  		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
  
  		self.linear1 = nn.Linear(128 * 6 * 6, 2048)
  		self.linear2 = nn.Linear(2048, 2048)
  		self.out = nn.Linear(2048, 10)
  		# Dropout层，p表示神经元被丢弃的概率
  		self.dropout = nn.Dropout(p=0.5)
  
  	def forward(self, x):
  		x = torch.relu(self.conv1(x))
  		x = self.pool1(x)
  		x = torch.relu(self.conv2(x))
  		x = self.pool2(x)
  		# 由于最后一个批次可能不够 32，所以需要根据批次数量来 flatten
  		x = x.reshape(x.size(0), -1)
  		x = torch.relu(self.linear1(x))
  		# dropout正则化
  		# 训练集准确率远远高于测试准确率,模型产生了过拟合
  		x = self.dropout(x)
  		x = torch.relu(self.linear2(x))
  		x = self.dropout(x)
  		return self.out(x)
  ```

## 2 RNN介绍

### 2.1 什么是RNN循环神经网络

- RNN是一种处理序列化数据的神经网络计算模型
- 序列化数据
  - 根据时间步生成的数据, 前后数据有关联
  - 文本数据

### 2.2 RNN应用场景

- NLP: 文本生成, 机器翻译
- 语音翻译
- 音乐生成

## 3 词嵌入层

### 3.1 词嵌入层作用

- 词向量化表示
- 低维稠密向量能更好学习词的语义关系

### 3.2 词嵌入层工作流程

- 通过jieba等分词模块先对句子进行分词处理
- 获取每个词对应位置的下标值
- 将下标值转换成张量对象丢入词嵌入层进行词向量输出

### 3.3 词嵌入层API使用

```python
import torch
import jieba
import torch.nn as nn


def dm01():
	# 一句话包含多个词
	text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'
	# 使用jieba模块进行分词
	words = jieba.lcut(text)
	# 返回词的列表
	print('words->', words)
	# 创建词嵌入层
	# num_embeddings:词数量
	# embedding_dim:词向量维度
	embed = nn.Embedding(num_embeddings=len(words), embedding_dim=8)
	# 获取每个词对象的下标索引
	for i, word in enumerate(words):
		# 将词索引转换成张量对象 向量
		word_vec = embed(torch.tensor(data=i))
		print('word_vec->', word_vec)


if __name__ == '__main__':
	dm01()
```

## 4 循环网络层

### 4.1 RNN网络层原理

- 作用: 处理序列文本数据
- 每一层的输入有上一个时间步的隐藏状态和当前的词向量
- 每一层的输出有当前预测分值y1和当前时间步的隐藏状态
- 隐藏状态: 具有记忆功能以及上下文理解功能
- 如果有多层RNN层, 进行生成式AI, 只取最后一层的隐藏状态和预测输出, 将预测输出丢入到全连接神经网络中进行词的预测
  - 词汇表有多少个词, 最后就是预测多少个类别, 每个词就是一个类别, 获取概率最大的类别对应的词
- h1 = relu(wh0+b+wx+b)
- y1 = wh1+b

### 4.2 RNN网络层API使用

```python
import torch
import torch.nn as nn


def dm01():
	# 创建RNN层
	# input_size: 词向量维度
	# hidden_size: 隐藏状态向量维度
	# num_layers: 隐藏层数量, 默认1层
	rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
	# 输入x
	# (5, 32, 128)->(每个句子的词个数, 句子数, 词向量维度)
	# RNN对象input_size=词向量维度
	x = torch.randn(size=(5, 32, 128))
	# 上一个时间步的隐藏状态h0
	# (1,32,256->(隐藏层数量, 句子数, 隐藏状态向量维度)
	# RNN对象hidden_size=隐藏状态向量维度
	h0 = torch.randn(size=(1, 32, 256))
	# 调用RNN层输出当前预测值和当前的隐藏状态h1
	output, h1 = rnn(x, h0)
	print(output.shape, h1.shape)


if __name__ == '__main__':
	dm01()
```

## 5 文本生成案例

### 5.1 构建词表

```python
import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time


# 获取数据，并进行分词，构建词表
def build_vocab():
	# 数据集位置
	file_path = 'data/jaychou_lyrics.txt'
	# 分词结果存储位置
	# 唯一词列表
	unique_words = []
	# 每行文本分词列表, 文本数据替换成 每行词列表的结果
	# [[],[],[]]
	all_words = []
	# 遍历数据集中的每一行文本
	for line in open(file=file_path, mode='r', encoding='utf-8'):
		# 使用jieba分词,分割结果是一个列表
		words = jieba.lcut(line)
		# print('words->', words)
		# 所有的分词结果存储到all_words，其中包含重复的词组
		# [[],[],[]]
		all_words.append(words)
		# 遍历分词结果，去重后存储到unique_words
		# words->['想要', '有', '直升机', '\n']
		for word in words:
			# word词不再词汇表中, 就是一个新词
			if word not in unique_words:
				unique_words.append(word)

	# print('unique_words->', unique_words)
	# print('all_words->', all_words)

	# 语料中词的数量
	word_count = len(unique_words)
	# print('word_count->', word_count)
	# 词到索引映射 {词:词下标}
	word_to_index = {}
	for idx, word in enumerate(unique_words):
		word_to_index[word] = idx
	# print('word_to_index->', word_to_index)
	# 歌词文本用词表索引表示
	corpus_idx = []
	# print('all_words->', all_words)
	# 遍历每一行的分词结果
	# all_words->[['老街坊', ' ', '小', '弄堂', '\n'], ['消失', '的', ' ', '旧', '时光', ' ', '一九四三', '\n']]
	for words in all_words:
		# 临时存储每行词的索引下标
		temp = []
		# 获取每一行的词，并获取相应的索引
		# words->['老街坊', ' ', '小', '弄堂', '\n']
		for word in words:
			# 根据词获取字典中的词索引下标 idx=dict[word]
			temp.append(word_to_index[word])
		# 在每行词之间添加空格隔开 -> 将每行\n和下一行分开, 没有语义关系
		temp.append(word_to_index[' '])
		# 获取当前文档中每个词对应的索引
		# extend: 将temp列表元素拆分后存储到corpus_idx列表中
		corpus_idx.extend(temp)
	return unique_words, word_to_index, word_count, corpus_idx


if __name__ == '__main__':
	unique_words, word_to_index, word_count, corpus_idx = build_vocab()
	# print('unique_words->', unique_words)
	# print('word_count->', word_count)
	# print('corpus_idx->', corpus_idx)
	# print('word_to_index->', word_to_index)
```

### 5.2 构建数据集对象

```python
# 构建数据集对象型
# 创建类继承 torch.utils.data.Dataset基类
class LyricsDataset(torch.utils.data.Dataset):
	# 定义构造方法
	# corpus_idx: 歌词文本用词表索引表示
	# num_chars: 每句话的词数量
	def __init__(self, corpus_idx, num_chars):
		self.corpus_idx = corpus_idx
		self.num_chars = num_chars
		# 统计歌词文本中有多少个词, 不去重
		self.word_count = len(corpus_idx)
		# 歌词文本能生成多少个句子
		self.number = self.word_count // self.num_chars

	# 重写__len__魔法方法, len()->输出返回值
	def __len__(self):
		return self.number

	# 重写__getitem__魔法方法 obj[idx]执行此方法 遍历数据加载器执行此方法
	def __getitem__(self, idx):
		# 设置起始下标值 start, 不能超过word_count-num_chars-1
		# -1: y的值要x基础上后移一位
		start = min(max(idx, 0), self.word_count - self.num_chars - 1)
        # 
		end = start + self.num_chars
		# 获取x
		x = self.corpus_idx[start: end]
		y = self.corpus_idx[start + 1: end + 1]
		return torch.tensor(x), torch.tensor(y)


if __name__ == '__main__':
	unique_words, word_to_index, word_count, corpus_idx = build_vocab()
	# print('unique_words->', unique_words)
	# print('word_count->', word_count)
	# print('corpus_idx->', corpus_idx)
	# print('word_to_index->', word_to_index)
	print('corpus_idx->', len(corpus_idx))
	dataset = LyricsDataset(corpus_idx, 5)
	print(len(dataset))
	# 获取第1组x和y
	x, y = dataset[49135]
	print('x->', x)
	print('y->', y)
```

### 5.3 构建网络模型

```python
class TextGenerator(nn.Module):
	def __init__(self, unique_word_count):
		super(TextGenerator, self).__init__()
		# 初始化词嵌入层: 语料中词的数量, 词向量的维度为128
		self.ebd = nn.Embedding(unique_word_count, 128)
		# 循环网络层: 词向量维度128, 隐藏向量维度256, 网络层数1
		self.rnn = nn.RNN(128, 256, 1)
		# 输出层: 特征向量维度256与隐藏向量维度相同, 词表中词的个数
		self.out = nn.Linear(256, unique_word_count)

	def forward(self, inputs, hidden):
		# 输出维度: (batch, seq_len, 词向量维度128)
		# batch：句子数量
		# seq_len： 句子长度， 每个句子由多少个词 词数量
		embed = self.ebd(inputs)
		# rnn层x的表示形式为(seq_len, batch, 词向量维度128)
		# output的表示形式与输入x类似，为(seq_len, batch, 词向量维度256)
		# 前后的hidden形状要一样, 所以DataLoader加载器的batch数要能被整数
		output, hidden = self.rnn(embed.transpose(0, 1), hidden)
		# 全连接层输入二维数据， 词数量*词维度
		# 输入维度: (seq_len*batch, 词向量维度256)
		# 输出维度: (seq_len*batch, 语料中词的数量)
		# output: 每个词的分值分布，后续结合softmax输出概率分布
		# output.shape[-1]: 词向量维度表示
		output = self.out(output.reshape(shape=(-1, output.shape[-1])))
		# 网络输出结果
		return output, hidden

	def init_hidden(self, bs):
		# 隐藏层的初始化:[网络层数, batch, 隐藏层向量维度]
		return torch.zeros(1, bs, 256)
    
if __name__ == "__main__":
	# 获取数据
	unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
	model = TextGenerator(unique_word_count)
	for named, parameter in model.named_parameters():
		print(named)
		print(parameter)
```

### 5.4 构建训练函数

```python
def train():
	# 构建词典
	unique_words, word_to_index, unique_word_count, corpus_idx = build_vocab()
	# 数据集 LyricsDataset对象，并实现了 __getitem__ 方法
	lyrics = LyricsDataset(corpus_idx=corpus_idx, num_chars=32)
	# 查看句子数量
	# print(lyrics.number)
	# 初始化模型
	model = TextGenerator(unique_word_count)
	# 数据加载器 DataLoader对象，并将lyrics dataset对象传递给它
	lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=5)
	# 损失函数
	criterion = nn.CrossEntropyLoss()
	# 优化方法
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	# 训练轮数
	epoch = 10
	for epoch_idx in range(epoch):
		# 训练时间
		start = time.time()
		iter_num = 0  # 迭代次数
		# 训练损失
		total_loss = 0.0
		# 遍历数据集 DataLoader 会在后台调用 dataset.__getitem__(index) 来获取每个样本的数据和标签，并将它们组合成一个 batch
		for x, y in lyrics_dataloader:
			print('y.shape->', y.shape)
			# 隐藏状态的初始化
			hidden = model.init_hidden(bs=5)
			# 模型计算
			output, hidden = model(x, hidden)
			print('output.shape->', output.shape)
			# 计算损失
			# y形状为(batch, seq_len), 需要转换成一维向量->160个词的下标索引
			# output形状为(seq_len, batch, 词向量维度)
			# 需要先将y进行维度交换(和output保持一致)再改变形状
			y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
			loss = criterion(output, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			iter_num += 1  # 迭代次数加1
			total_loss += loss.item()
		# 打印训练信息
		print('epoch %3s loss: %.5f time %.2f' % (epoch_idx + 1, total_loss / iter_num, time.time() - start))
	# 模型存储
	torch.save(model.state_dict(), 'model/lyrics_model_%d.pth' % epoch)
    
if __name__ == "__main__":
	train()
```

### 5.5 构建预测函数

```python
def predict(start_word, sentence_length):
	# 构建词典
	unique_words, word_to_index, unique_word_count, _ = build_vocab()
	# 构建模型
	model = TextGenerator(unique_word_count)
	# 加载参数
	model.load_state_dict(torch.load('model/lyrics_model_10.pth'))
	# 隐藏状态
	hidden = model.init_hidden(bs=1)
	# 将起始词转换为索引
	word_idx = word_to_index[start_word]
	# 产生的词的索引存放位置
	generate_sentence = [word_idx]
	# 遍历到句子长度，获取每一个词
	for _ in range(sentence_length):
		# 模型预测
		output, hidden = model(torch.tensor([[word_idx]]), hidden)
		# 获取预测结果
		word_idx = torch.argmax(output)
		generate_sentence.append(word_idx)
	# 根据产生的索引获取对应的词，并进行打印
	for idx in generate_sentence:
		print(unique_words[idx], end='')


if __name__ == '__main__':
	# 调用预测函数
	predict('分手', 50)
```
