# 从零手写 Qwen 系列模型

本项目从零手写了 Qwen2 系列模型，模型的架构基于 Qwen2（transformer) 架构及相关技术进行。

本项目从零手写了 Qwen2-xxB-Instruction 模型的所有细节，包含:

- Embedding (嵌入层)
- Rope (旋转位置编码)
- Attention（GQA, 分组查询注意力机制)
- FFN (前馈神经网络)
- RmsNorm (归一化层)
- Linear（线性层）
- Softmax （分类器）
- post or generation process （后处理生成）

并在实现以上细节的基础上，搭建了**对话型**的 AI 大模型框架，借助 Qwen2 系列模型的预训练权重，实现了从零手写大模型的完整功能：

那就是 —— **手写的模型可在 CPU/GPU 上运行、并且可完整的完成对话功能**。

## 本项目支持的特性

### 模型列表

- Qwen2-0.5B-Instruction
- Qwen2-1.5B-Instruction
- Qwen2-7B-Instruction
- Qwen2-72B-Instruction

更多模型及 Qwen3 or MOE 模型正在支持中，欢迎关注。

### AI 智能体

- 支持将手写的 Qwen2 模型部署到前端页面

---

# 项目学习路线

本仓库的内容不包含 Qwen2 的技术细节（文档）的介绍，仅包含源码。项目的详细说明见：[从零手写 Qwen 系列模型](https://www.yuque.com/yuqueyonghupftxbc/ai100/lc1bna1l1dl2zp39)。

如果你希望系统的学习 AI 大模型尤其是 Qwen 系列大模型的技术细节，欢迎订阅我的 AI 大模型技术专栏：[Transformer 通关秘籍](https://www.yuque.com/yuqueyonghupftxbc/ai100/wvyi8axax45piuxq)。

技术专栏包含了几十万字内容和图解，详细梳理了大模型技术的前世今生，以及手把手带你玩转本项目的详实资料，还有有更加详细的**注释版本**的源代码可供学习，同时我也提供**一对一的学习辅导**服务，确保你完全学会，欢迎来撩（联系我）。

## 在这里订阅技术专栏

- [Transformer 通关秘籍](https://www.yuque.com/yuqueyonghupftxbc/ai100/wvyi8axax45piuxq)

- [从零手写大模型实战](https://www.yuque.com/yuqueyonghupftxbc/ai100/lc1bna1l1dl2zp39)

## 撩一撩（联系我）

- 微信号：ddcsggcs (加微信备注：大模型)

- 微信公众号：[见这里](https://mp.weixin.qq.com/s/lKwSvfpMt7iNqa83_HlQug)

---

# 仓库说明

本项目的代码仓库仅保留了运行 Qwen 模型最核心的代码，无用或冗余的代码已经删除，这样使得模型的代码逻辑更加清晰，方便大家进行大模型技术的学习。

因为 Qwen 系列模型使用的架构或者说技术完全相同，因此本项目在从零手写完一个系列（如 Qwen2 系列）后，可以无缝、完整的运行 **Qwen2-0.5、1.5B、7B、72B** 等模型，并可以与手写的模型进行对话，同时仓库中还提供了 CPU 版本与 GPU 版本。

## 仓库目录结构

- models: 利用 transformers 库运行 Qwen 模型的示例代码
- src：手写的 Qwen 系列模型的源代码目录（包含多个版本） 
- frontend: 前端代码目录，用于构建智能体
- scripts：用来维护仓库的一些脚本（无需关注）
- requirements.txt：项目的依赖库
- env.py：用来配置本仓库所需环境的脚本

## 如何配置环境

**下载仓库**
```bash
git clone git@github.com:dongdongcan/write_qwen_from_scratch.git
```

**环境配置**

本仓库使用 python 的虚拟环境来进行执行，因此需要先配置好 python 的虚拟环境。

Linux 环境下，在仓库根目录下执行：

```bash
python3 env.py
```
执行完上述命令后，会输出如何进入虚拟环境的**提示**，提示类似于：

```shell
Use the following cmd to enter virtual environment
>>> source /home/dongdongcan/write_qwen_from_scratch/.venv/bin/activate
```

**进入虚拟环境**

执行输出的提示命令，进入虚拟环境

```bash
source /home/dongdongcan/write_qwen_from_scratch/.venv/bin/activate
```

**安装依赖**

在虚拟环境下，执行：

```bash
python3 env.py --install
```
完成本项目所需的所有依赖库的安装后，即可运行 Qwen 模型。

---

# 项目引用

本仓库基于 Qwen2 的技术：Qwen model (Apache 2.0 License) Copyright 2024 Alibaba Cloud

本仓库使用到了 transformers 库，引用如下：

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```
