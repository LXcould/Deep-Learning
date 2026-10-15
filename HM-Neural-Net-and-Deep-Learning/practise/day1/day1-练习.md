# Day1 练习

> **覆盖范围**：深度学习简介 · PyTorch 框架 · 张量创建 · 张量类型转换
> **对应课程**：黑马《神经网络与深度学习》Day01（分P 1–13）

---

## 一、要点速览

### 1. 深度学习

- 是机器学习的一个子集，以**人工神经网络**为结构，**自动提取特征**（传统机器学习靠人工设计特征）
- 特点：自动提取特征 · 可解释性差 · 需要大量数据和算力 · 引入非线性变换
- 常见模型：**ANN**（人工神经网络）· **CNN**（图像）· **RNN**（序列）· **Transformer**（RNN 衍生）· GAN

### 2. PyTorch

- 基于 Python 的深度学习框架，把数据封装成**张量（Tensor）**来处理
- 特点：类 NumPy 的张量计算 · 自动微分 · 动态计算图 · GPU 加速（CUDA）
- 2016 年 Facebook 发布，2018 年 1.0 版进入生产级应用

### 3. 张量创建（核心 API）

| 用途 | API |
|---|---|
| 从已有数据创建 | `torch.tensor(data)` |
| 全 0 / 全 1 | `torch.zeros(size)` / `torch.ones(size)` |
| 指定值 | `torch.full(size, fill_value)` |
| 等差序列 | `torch.arange(start, end, step)` |
| 等距等分 | `torch.linspace(start, end, steps)` |
| 均匀分布随机 | `torch.rand(size)` —— 取值 [0, 1) |
| 标准正态随机 | `torch.randn(size)` —— 均值 0、标准差 1 |
| 随机整数 | `torch.randint(low, high, size)` |

**三个关键属性**：`.shape`（形状）· `.dtype`（元素类型）· `.ndim`（维度数）

### 4. 类型转换

| 转换方向 | 写法 | 内存 |
|---|---|---|
| 张量 → ndarray | `t.numpy()` | **共享内存** |
| ndarray → 张量 | `torch.from_numpy(arr)` | **共享内存** |
| ndarray → 张量 | `torch.tensor(arr)` | 不共享 |
| 标量张量 → Python 数字 | `t.item()` | —— |
| 元素类型转换 | `t.float()` / `t.long()` / `t.type(torch.float64)` | —— |
| 用 dtype 参数直接指定 | `torch.tensor(data, dtype=torch.float32)` | —— |

⚠️ **最容易错的一点**：`.numpy()` 和 `torch.from_numpy()` 是**共享内存**的，改一个另一个跟着变。要断开用 `.copy()` 或 `.clone()`。

---

## 二、练习

**1.** 创建形状为 `(3, 4)` 的全 0 张量和全 1 张量，分别打印它们的 `shape` 和 `dtype`。

**2.** 用 `torch.arange` 生成 0 到 9 的整数张量；再用 `torch.linspace` 在 0 和 1 之间生成 5 个等距点。

**3.** 创建两个 `(2, 3)` 的随机张量，一个服从标准正态分布，一个取值在 [0, 1) 之间均匀分布。

**4.** 创建形状 `(3, 3)`、每个元素都是 7 的张量。

**5.** 创建整数张量 `[1, 2, 3]`，把它转换成 `float32` 类型（写出两种写法）。

**6.** 用 `dtype` 参数直接创建 `[1.5, 2.5]` 的 `float64` 张量。

**7.** 下面这段代码打印出什么？为什么？

```python
import numpy as np
import torch

a = np.array([1, 2, 3])
t = torch.from_numpy(a)
t[0] = 99
print(a)
```

**8.** 接上题：如果不想让 `a` 跟着变，写出两种写法。

**9.** `torch.randn(2, 3, 4)` 的 `shape` 和 `ndim` 分别是什么？把其中第一个元素取成 Python 数字怎么写？

---

## 三、答案

**1.**

```python
z = torch.zeros(3, 4)
o = torch.ones(3, 4)
print(z.shape, z.dtype)   # torch.Size([3, 4]) torch.float32
print(o.shape, o.dtype)   # torch.Size([3, 4]) torch.float32
```

**2.**

```python
print(torch.arange(0, 10))        # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(torch.linspace(0, 1, 5))    # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

**3.**

```python
print(torch.randn(2, 3))   # 标准正态分布，均值 0、标准差 1
print(torch.rand(2, 3))    # [0, 1) 均匀分布
```

**4.**

```python
print(torch.full((3, 3), 7))
```

**5.**

```python
t = torch.tensor([1, 2, 3])
print(t.dtype)          # torch.int64

t = t.float()           # 写法一
t = t.type(torch.float32)   # 写法二
print(t.dtype)          # torch.float32
```

**6.**

```python
t = torch.tensor([1.5, 2.5], dtype=torch.float64)
print(t.dtype)          # torch.float64
```

**7.** 打印 `[99 2 3]`。

因为 `torch.from_numpy(a)` 和原 ndarray **共享同一块内存**，修改张量 `t` 等于修改原数组 `a`。这是 PyTorch 为了省内存做的设计，但也最容易踩坑。

**8.**

```python
t = torch.tensor(a)                    # 写法一：直接拷贝
t = torch.from_numpy(a).clone()        # 写法二：先共享，再复制一份
```

**9.**

```python
t = torch.randn(2, 3, 4)
print(t.shape)        # torch.Size([2, 3, 4])
print(t.ndim)         # 3
print(t[0, 0, 0].item())   # 取单个元素 → Python float
```

> 注意：`item()` 只能用在**只有一个元素**的张量上，`t.item()` 对 `(2,3,4)` 会直接报错。
