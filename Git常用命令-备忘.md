# Git / GitHub 日常使用备忘

> 仓库：https://github.com/LXcould/First-repo
> 本地目录：`D:\...\ChatGPT\ai学习`
> 分支：`main`

---

## 一、每天就这三条（99% 的情况）

```
git add .
git commit -m "今天做了什么"
git push
```

顺序是：**收文件 → 存档 → 送到 GitHub。**

**什么时候敲**：每天学习结束时一次。不用每改一行就提交，也不用攒好几天。

**提交信息怎么写**：写清楚"加了什么"，比如

- `添加数据容器笔记`
- `完成列表切片练习`
- `添加函数进阶笔记`

半年后你 `git log` 看到的就是这些，写清楚才能找回东西。

---

## 二、偶尔会用到

### 看看自己改了什么

```
git status          # 哪些文件被改了
git diff            # 具体改了哪几行
```

### 看历史存档点

```
git log --oneline
```

更方便的是去 GitHub 网页，点 **commits**，能看到每次改了哪些文件、具体改了什么。

### 单个文件改坏了，想丢弃改动

```
git restore 文件名
```

（注意：这会丢掉你没提交的修改，回不来了，想好了再敲。）

### 想在 GitHub 网页上直接改文件

点文件 → 右上角铅笔图标 → 改 → **Commit changes**。

**改完记得在本地敲 `git pull` 拉下来**，否则本地和远程不一致，下次 push 会被拒绝。

### 换了台电脑

```
git clone git@github.com:LXcould/First-repo.git
```

拉下来之后，每天照旧三条命令。

---

## 三、四条原则

1. **每天都 push。** 只要推到 GitHub 了，换电脑、重装系统都不怕。只在本地攒着，硬盘坏了就全没了。
2. **提交信息写具体。** 别写 `update`、`111`、`学习`。
3. **私钥不外传。** `C:\Users\86186\.ssh\id_ed25519` 留在自己电脑上，不发任何人、不上传任何网站。`id_ed25519.pub` 才是给 GitHub 的。
4. **别推大数据集。** GitHub 单个文件限制 100MB。数据集、模型权重、`__pycache__`、虚拟环境这些不要提交（以后学 `.gitignore` 来排除）。

---

## 四、常见报错速查

| 报错 | 原因 | 怎么解决 |
|---|---|---|
| `src refspec xxx does not match any` | 分支名写错了，或不存在这个分支 | 先 `git branch` 看当前分支叫什么，你现在是 `main` |
| `'origin' does not appear to be a git repository` | 远程仓库没配 | `git remote add origin git@github.com:LXcould/First-repo.git` |
| `rejected ... fetch first` | 远程有你本地没有的提交 | 先 `git pull`，再 `git push` |
| `Permission denied (publickey)` | SSH 密钥没配对 | 检查 GitHub 上的公钥、以及 `.ssh` 里的密钥 |

报错不要慌，**把原文复制下来去查**，Git 的报错信息都很具体。
