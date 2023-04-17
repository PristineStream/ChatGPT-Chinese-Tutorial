# ChatGPT-Chinese-Tutorial <br> ChatGPT中文学习和实践资料汇总


[![Awesome](./logo.png)]() 
[![Code License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/yzfly/awesome-chatgpt-zh/blob/main/LICENSE)

## ChatGPT使用指南
- [ChatGPT官方地址](https://chat.openai.com/)

- [ChatGPT注册指南](https://www.chatgpto.com/articles/ChatGPT%E6%B3%A8%E5%86%8C.html)

- [ChatGPT中文调教指南](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)

- [工程师应用分享](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)

- [ChatGPT相关产品](https://gpt3demo.com/)

## ChatGPT复现开源技术

- [Colossal Chat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)：ColossalChat是第一个基于LLaMA预训练模型开源完整RLHF pipline实现，包括有监督数据收集、有监督微调、奖励模型训练和强化学习微调。您可以开始用1.6GB的GPU内存复制ChatGPT训练过程，并在训练过程中体验7.73倍的加速。
- [DeepSpeed Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat/chinese)：微软官方开源，一键式RLHF训练，让你的类ChatGPT千亿大模型提速省钱15倍，使用 DeepSpeed-Chat 的 RLHF 示例轻松训练你的第一个 类ChatGPT 模型——[训练代码](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)。


## ChatGPT论文学习
从上往下阅读学习

| 名称  | 说明 | 学习资料
| ----  | ----  | ----
|[Attention Is All You Need](https://arxiv.org/abs/1706.03762v4) | Transformer模型说明 | 1. [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) <br> 2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
| <br>[（GPT-1）Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) <br>[（GPT-2）Language Models are Unsupervised Multitask Learners](https://life-extension.github.io/2020/05/27/GPT%E6%8A%80%E6%9C%AF%E5%88%9D%E6%8E%A2/language-models.pdf) <br>[（GPT-3）Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |GPT系列 |http://people.ee.duke.edu/~lcarin/Dixin2.22.2019.pdf <br> [GPT: Generative Pre-Trained Transformer (2018)](https://kikaben.com/gpt-generative-pre-training-transformer-2018/)<br>[ GPT-2: Too Dangerous To Release (2019)](https://kikaben.com/gpt-2-2019) <br> [Better language models and their implications](https://openai.com/research/better-language-models)<br> [GPT-3: In-Context Few-Shot Learner (2020)](https://kikaben.com/gpt-2-2019/)	
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | PPO算法 | https://huggingface.co/blog/deep-rl-ppo	
| [Augmenting Reinforcement Learning with Human Feedback](https://www.ias.informatik.tu-darmstadt.de/uploads/Research/ICML2011/icml11il-knox.pdf) | RHLF | https://huggingface.co/blog/rlhf<br>[CS224N-Lecture 11: Prompting, Instruction Finetuning, and RLHF](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture11-prompting-rlhf.pdf)
| [Training language models to follow instructions with human feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) | InstructGPT(ChatGPT) | https://openai.com/blog/chatgpt<br>[InstructGPT论文视频精读](https://www.bilibili.com/video/BV1hd4y187CR/?vd_source=e2f9282e52e2f67ccd395c2b20014d76)
| [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | GPT-4 | https://openai.com/research/gpt-4<br>[GPT4论文视频精读](https://www.bilibili.com/video/BV1vM4y1U7b5/?vd_source=e2f9282e52e2f67ccd395c2b20014d76)
|[GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models](https://arxiv.org/abs/2303.10130) | GPT对美国劳动力市场影响研究 | https://openai.com/research/gpts-are-gpts

## 类ChatGPT论文学习
| 名称  | 说明 | 学习资料
| ----  | ----  | ----
| [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) | LaMDA模型 | [LaMDA: Towards Safe, Grounded, and High-Quality Dialog Models for Everything](https://ai.googleblog.com/2022/01/lamda-towards-safe-grounded-and-high.html)
| [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://aclanthology.org/2022.acl-long.26/)<br>[GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414) | GLM系列  | http://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/<br>https://chatglm.cn/blog
|[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | LLaMA |https://ai.facebook.com/blog/large-language-model-llama-meta-ai/
|[BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) |BLOOM  |[Understand BLOOM, the Largest Open-Access AI, and Run It on Your Local Computer](https://towardsdatascience.com/run-bloom-the-largest-open-access-ai-model-on-your-desktop-computer-f48e1e2a9a32)

## 对话大模型Finetune开源
### LLaMA：英文预训练大模型(最大65B)
- [Facebook官方](https://github.com/facebookresearch/llama)
- [Instruct\-tune with lora 包含多种语言的lora（中文、日文、德文）开箱即用）](https://github.com/search?q=LLaMA)
- [中文LLaMA模型和指令精调的Alpaca大模型，专注于中文LLaMA的开源项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [纯C/C++版的LLaMA模型部署推理](https://github.com/ggerganov/llama.cpp)
- [在不同系统一键运行LLaMA和Alpaca](https://github.com/cocktailpeanut/dalai)
- [WebUI一键部署LLaMA](https://github.com/oobabooga/text-generation-webui)
- [Rust版的LLaMA模型部署推理](https://github.com/rustformers/llama-rs)
- [BELLE：专注于在开源预训练大语言模型的基础上训练出具有指令表现能力的中文语言大模型](https://github.com/LianjiaTech/BELLE)
- [ChatLLaMA中文对话模型的开源项目](https://github.com/ydli-ai/Chinese-ChatLLaMA)

### GLM：中英双语预训练大模型(最大130B)
- [GLM官方预训练](https://github.com/THUDM/GLM)
- [GLM-130B官方](https://github.com/THUDM/GLM-130B)
- [ChatGLM-6B:类ChatGPT技术训练的开源中英双语对话模型](https://github.com/THUDM/ChatGLM-6B)
- [ChatGLM-6B官方P-Tuning](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning)
- [ChatGLM-6B 基于Lora的Finetune 支持模型并行](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b)
- [ChatGLM-6B 基于Lora的Finetune Peft实现 单卡版本 学习推荐](https://github.com/mymusise/ChatGLM-Tuning)
- [基于本地知识的 ChatGLM 问答应用](https://github.com/imClumsyPanda/langchain-ChatGLM)


## LLM相关课程
| 名称  | 说明 | 链接
| ----  | ----  | ----
| LLM CS324 | 自然语言处理（NLP）领域已经被大量的预训练语言模型所改变。它们构成了各种任务中所有最先进系统的基础，并显示出生成流畅文本和执行少量学习的能力。同时，由于这些模型难以理解，并带来了新的道德和可扩展性挑战。在本课程中，学生将学习有关大型语言模型的建模，理论，伦理和系统方面的基础知识，并获得对应的实践经验。 |[链接](https://stanford-cs324.github.io/winter2022/)
| 【李沐】InstructGPT 背后的技术 | InstructGPT 论文精读| [链接](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.788&vd_source=71b548de6de953e10b96b6547ada83f2)
|【李宏毅】ChatGPT 是怎么炼成的 | GPT社会化的过程 | [链接](https://www.bilibili.com/video/BV1U84y167i3/?p=1&vd_source=71b548de6de953e10b96b6547ada83f2) 
|【Aston Zhang】Chain of Thought | 结合in-context fewshot 和中间步骤来改善LLM的arithmetic, commonsense, and symbolic reasoning等推理能力 | [链接](https://www.bilibili.com/video/BV1t8411e7Ug/?spm_id_from=333.788&vd_source=1e55c5426b48b37e901ff0f78992e33f)