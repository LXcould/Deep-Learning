# 深度学习入门笔记

> 对应课程：黑马程序员《神经网络与深度学习》—— 🔗 [BV1c5yrBcEEX](https://www.bilibili.com/video/BV1c5yrBcEEX)
> 全课 108 集 / 31 小时 40 分，本目录是按 Day01–Day07 顺序整理的课堂笔记。

---

## 目录结构

```
HM-Neural-Net-and-Deep-Learning/
├── notes/          课堂笔记（6 篇，含代码示例）
│   └── assets/     笔记里用到的插图
├── papers/         参数初始化相关的两篇原始论文
└── mindmaps/       思维导图（.xmind 原件 + 转换出的 .md）
```

---

## 一、课堂笔记

| 文件 | 对应课程 | 覆盖内容 |
|---|---|---|
| [01_深度学习与PyTorch基础](notes/01_深度学习与PyTorch基础.md) | Day01–Day02（P1–18） | 深度学习概念 / 特点 / 常见模型 / 应用场景 · PyTorch 简介 · 张量创建 · 类型转换 · 数值计算 · 运算函数 |
| [02_张量索引与形状操作](notes/02_张量索引与形状操作.md) | Day02（P19–29） | 索引取值 · reshape · squeeze / unsqueeze · transpose / permute · view / contiguous · cat / stack · 自动微分 |
| [03_线性回归与神经网络基础](notes/03_线性回归与神经网络基础.md) | Day03（P30–43） | PyTorch 构建线性回归 · 人工神经网络原理 · 激活函数 · 参数初始化 |
| [04_损失函数与优化方法](notes/04_损失函数与优化方法.md) | Day04–Day05（P44–68） | 网络搭建与参数计算 · 损失函数 · 反向传播 BP · 优化器 · 学习率衰减 |
| [05_正则化与CNN基础](notes/05_正则化与CNN基础.md) | Day05–Day06（P69–91） | 正则化（L1 / L2 / Dropout / BN）· 手机价格分类案例 · 图像基础 · CNN 卷积层与池化层 |
| [06_图像分类案例与RNN](notes/06_图像分类案例与RNN.md) | Day07（P92–108） | CIFAR10 图像分类完整案例 · RNN 原理 · 词嵌入层 · 循环网络层 · 歌词生成案例 |

**说明**：笔记按课程进度顺序编号，跟着视频走时按序号翻对应文件即可。

**每篇笔记的结构**：

1. **课堂笔记**（开头部分）—— 跟课时自己手敲的笔记和代码
2. **附一 · 课程讲义** —— 黑马官方讲义，按主题归档到对应章节
3. **附二 · 课程课件** —— 配套 PPT 的文字提取（图表、公式、截图未包含，需要看图请对照原课件）

复习原理看讲义，回忆上课细节看笔记，快速过一遍看课件。

---

## 二、参数初始化论文

笔记 03 里讲到的 Xavier 和 Kaiming 初始化，对应的原始论文：

| 文件 | 论文 | 提出的初始化方法 |
|---|---|---|
| [Glorot2010](papers/Glorot2010_Understanding-the-Difficulty-of-Training-Deep-Feedforward-Neural-Networks.pdf) | Glorot & Bengio, 2010 | **Xavier 初始化**（也叫 Glorot 初始化），对应 `nn.init.xavier_normal_` |
| [He2015](papers/He2015_Delving-Deep-into-Rectifiers.pdf) | He et al., 2015 | **Kaiming 初始化**（He 初始化），专为 ReLU 设计，对应 `nn.init.kaiming_normal_` |

**为什么留这两篇**：面试问"权重为什么不能全初始化为 0""Xavier 和 Kaiming 有什么区别"，答案就在这两篇里。现在看不懂很正常，等工作里真用到或者面试前再回来读。

---

## 三、思维导图

`.xmind` 是原始文件（需要 XMind 软件打开），`.md` 是自动转换出来的大纲版本，**在 GitHub 上能直接看**。

| 导图 | 覆盖内容 |
|---|---|
| [PyTorch框架总结](mindmaps/PyTorch框架总结.md) | 从深度学习介绍到 PyTorch 全部核心 API 的框架总览 |
| [人工神经网络基础](mindmaps/人工神经网络基础.md) | ANN 原理、激活函数、参数初始化、损失函数、优化器 |
| [卷积神经网络CNN](mindmaps/卷积神经网络CNN.md) | 卷积层、池化层、经典网络结构 |
| [循环神经网络RNN](mindmaps/循环神经网络RNN.md) | RNN 原理、词嵌入、序列建模 |

---

## 学习进度

- [x] Day01 深度学习介绍 + PyTorch 张量创建
- [x] Day02 张量运算 · 形状操作 · 自动微分
- [x] Day03 线性回归 · 神经网络 · 激活函数 · 参数初始化
- [x] Day04 网络搭建 · 损失函数 · 反向传播
- [x] Day05 优化器 · 学习率衰减 · 正则化
- [x] Day06 ANN 案例 · CNN 卷积层与池化层
- [x] Day07 图像分类案例 · RNN 文本生成

**待补充**（这门课没有覆盖，后续要单独补）：

- [ ] 注意力机制（Attention）
- [ ] Transformer 架构
- [ ] LSTM / GRU
- [ ] 图神经网络（GNN）

---

## 一个提醒

这些是**跟着课程记的笔记**，不是教材。后面复现论文、自己做项目时，**遇到具体问题要回来查对应的那一篇**——尤其是张量形状操作（笔记 02）和损失函数、优化器那两块（笔记 04），它们是你后面调试和调参时翻得最勤的。
