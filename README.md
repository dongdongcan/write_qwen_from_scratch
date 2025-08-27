## 说明

本仓库基于阿里云的 qwen 大模型以及相关的技术进行。

本项目从零手写了 Qwen2.0-0.5B-Instruction 模型的所有细节，包含 embedding, rope, attention（GQA）, ffn, rms_norm, linear, softmax 以及后处理等，并在实现细节的基础上，搭建了AI大模型的对话框架，借助模型的预训练权重，实现了手写大模型的完整功能：可运行、可对话。

本仓库保留了大模型技术最核心的代码，无用或冗余的代码已经删除，这样使得整体代码逻辑更加清晰，方便大家进行大模型技术的学习。

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