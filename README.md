## 说明

本仓库从零手写了 Qwen2 系列模型，模型架构基于 Qwen 大模型以及相关的技术进行。

本项目从零手写了 Qwen2.0-0.5B-Instruction 模型的所有细节，包含 embedding, rope, attention（GQA）, ffn, rms_norm, linear, softmax 以及后处理等，并在实现细节的基础上，搭建了 AI 大模型的对话框架，借助模型的预训练权重，实现了手写大模型的完整功能：手写完后的模型可运行、可与之对话。

本仓库保留了大模型技术最核心的代码，无用或冗余的代码已经删除，这样使得整体代码逻辑更加清晰，方便大家进行大模型技术的学习。

因为 Qwen2 系列架构相同，因此本仓库手写完 Qwen 架构后，你可以完整的运行 *Qwen2-0.5、1.5B、7B、72B* 等模型，并可以与手写的模型进行对话，同时还提供了 CPU 版本与 GPU 版本，见 src 目录。

本仓库不对 Qwen2 的技术作过多介绍，如果你希望系统的学习 AI 大模型的实现技术细节，可以订阅我的 AI 大模型技术专栏，也可找我进行一对一教学。

## AI 大模型技术专栏
[Transformer 通关秘籍](https://www.yuque.com/yuqueyonghupftxbc/ai100/wvyi8axax45piuxq)
[从零手写大模型实战](https://www.yuque.com/yuqueyonghupftxbc/ai100/lc1bna1l1dl2zp39)

## 联系我
微信：ddcsggcs
微信公众号：[见这里](https://mp.weixin.qq.com/s/lKwSvfpMt7iNqa83_HlQug)

## 引用

本仓库使用到了 transformer 库，引用如下：

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

## Cite
This project is based on:
- Qwen model (Apache 2.0 License) Copyright 2024 Alibaba Cloud
- 🤗 Transformers library (Apache 2.0 License) Copyright 2018- The Hugging Face team.
