# ChatGPT-Chinese-Tutorial <br> ChatGPT 与大语言模型（LLM）中文学习资料汇总


[![Awesome](./logo.png)]() 
[![Code License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/yzfly/awesome-chatgpt-zh/blob/main/LICENSE)

> 本仓库持续更新中文 LLM 学习资源，涵盖国产大模型、开源模型、论文精读、微调部署、AI Agent、Prompt 工程等内容。
>
> 最近更新：2026 年 6 月

---

## 目录

- [大模型产品与平台](#大模型产品与平台)
- [开源大模型](#开源大模型)
- [论文精读](#论文精读)
- [微调与训练](#微调与训练)
- [本地部署](#本地部署)
- [AI 应用开发平台](#ai-应用开发平台)
- [AI Agent 资源](#ai-agent-资源)
- [MCP 资源](#mcp-资源)
- [Prompt 工程](#prompt-工程)
- [学习课程与视频](#学习课程与视频)
- [综合资源索引](#综合资源索引)

---

## 大模型产品与平台

| 名称 | 开发者 | 链接 | 说明 |
| ---- | ---- | ---- | ---- |
| ChatGPT | OpenAI | [chat.openai.com](https://chat.openai.com/) | GPT-4o / GPT-5 系列，全球用户量最大的 AI 助手 |
| DeepSeek | 深度求索 | [chat.deepseek.com](https://chat.deepseek.com/) | DeepSeek-R1 / V3，强推理能力，开源标杆 |
| 通义千问 | 阿里巴巴 | [tongyi.aliyun.com](https://tongyi.aliyun.com/) | Qwen3 系列，国内综合能力领先 |
| 文心一言 | 百度 | [yiyan.baidu.com](https://yiyan.baidu.com/) | ERNIE 4.5 系列，中文理解能力突出 |
| Kimi | 月之暗面 | [kimi.moonshot.cn](https://kimi.moonshot.cn/) | 超长上下文处理，支持文件解析 |
| 豆包 | 字节跳动 | [www.doubao.com](https://www.doubao.com/) | Doubao 模型，多模态能力强 |
| 智谱清言 | 智谱 AI | [chatglm.cn](https://chatglm.cn/) | GLM-4 系列，清华技术背景 |
| 腾讯元宝 | 腾讯 | [yuanbao.tencent.com](https://yuanbao.tencent.com/) | 混元大模型，集成微信生态 |

---

## 开源大模型

| 模型 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| DeepSeek-R1 / V3 | 深度求索 | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) ⭐92k | 671B MoE，推理能力比肩 OpenAI-o1，含多种蒸馏版本 |
| Qwen3 系列 | 阿里巴巴 | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) ⭐27k | 0.6B-235B 全尺寸，支持 100 万上下文 |
| GLM-4 | 智谱 AI / 清华 | [THUDM/GLM-4](https://github.com/THUDM/GLM-4) ⭐7k | 多语言多模态，支持函数调用与深度思考 |
| Yi 系列 | 零一万物 | [01-ai/Yi](https://github.com/01-ai/Yi) ⭐7.8k | 双语 LLM，支持 200K 上下文，Apache 2.0 协议 |
| InternLM 系列 | 上海 AI 实验室 | [InternLM/InternLM](https://github.com/InternLM/InternLM) ⭐7.2k | 支持 1M 上下文，推理与代码能力强 |
| LLaMA 3 / 4 | Meta | [meta-llama/llama-models](https://github.com/meta-llama/llama-models) | 全球最流行的开源基座模型 |
| BLOOM | BigScience | [bigscience-workshop/bloom](https://github.com/bigscience-workshop/model_card) | 176B 参数多语言模型 |

---

## 论文精读

### 基础论文（从 Transformer 到 GPT 系列）

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

### 重要模型论文

| 名称  | 说明 | 学习资料
| ----  | ----  | ----
| [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) | LaMDA模型 | [LaMDA: Towards Safe, Grounded, and High-Quality Dialog Models for Everything](https://ai.googleblog.com/2022/01/lamda-towards-safe-grounded-and-high.html)
| [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://aclanthology.org/2022.acl-long.26/)<br>[GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414) | GLM系列  | http://keg.cs.tsinghua.edu.cn/glm-130b/posts/glm-130b/<br>https://chatglm.cn/blog
|[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | LLaMA |https://ai.facebook.com/blog/large-language-model-llama-meta-ai/
|[BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) |BLOOM  |[Understand BLOOM, the Largest Open-Access AI, and Run It on Your Local Computer](https://towardsdatascience.com/run-bloom-the-largest-open-access-ai-model-on-your-desktop-computer-f48e1e2a9a32)
| [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | DeepSeek-R1 | [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
| [Qwen Technical Report](https://arxiv.org/abs/2309.16609) | 通义千问 | [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)

---

## 微调与训练

### 微调框架

- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)：⭐72k，一站式 LLM 微调框架，支持 100+ 模型（Qwen、DeepSeek、LLaMA、GLM 等），提供 LoRA/QLoRA/全参微调，内置 WebUI，ACL 2024 论文
- [Colossal Chat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)：基于 LLaMA 的完整 RLHF 训练流程，支持 DeepSeek-V3/R1 LoRA 微调
- [DeepSpeed Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat/chinese)：微软官方开源，一键式 RLHF 训练——[训练代码](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

### 微调教程

- [开源大模型食用指南（self-llm）](https://github.com/datawhalechina/self-llm)：⭐30.8k，针对中文开发者的 Linux 环境下开源模型部署与微调教程，覆盖 50+ 主流模型
- [从零开始构建大模型（happy-llm）](https://github.com/datawhalechina/happy-llm)：⭐31k，从 NLP 基础到手搓 LLaMA2，涵盖 Pretrain、SFT、PEFT 全流程

### 经典微调项目

<details>
<summary>LLaMA 系列（点击展开）</summary>

- [Facebook 官方 LLaMA](https://github.com/facebookresearch/llama)
- [中文LLaMA模型和指令精调的Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [纯C/C++版的LLaMA模型部署推理（llama.cpp）](https://github.com/ggerganov/llama.cpp)
- [在不同系统一键运行LLaMA和Alpaca](https://github.com/cocktailpeanut/dalai)
- [WebUI一键部署LLaMA](https://github.com/oobabooga/text-generation-webui)
- [Rust版的LLaMA模型部署推理](https://github.com/rustformers/llama-rs)
- [BELLE：开源中文指令大模型](https://github.com/LianjiaTech/BELLE)
- [ChatLLaMA中文对话模型](https://github.com/ydli-ai/Chinese-ChatLLaMA)

</details>

<details>
<summary>GLM 系列（点击展开）</summary>

- [GLM官方预训练](https://github.com/THUDM/GLM)
- [GLM-130B官方](https://github.com/THUDM/GLM-130B)
- [ChatGLM-6B:开源中英双语对话模型](https://github.com/THUDM/ChatGLM-6B)
- [ChatGLM-6B官方P-Tuning](https://github.com/THUDM/ChatGLM-6B/tree/main/ptuning)
- [ChatGLM-6B 基于Lora的Finetune 支持模型并行](https://github.com/yuanzhoulvpi2017/zero_nlp/tree/main/simple_thu_chatglm6b)
- [ChatGLM-6B 基于Lora的Finetune Peft实现](https://github.com/mymusise/ChatGLM-Tuning)
- [基于本地知识的 ChatGLM 问答应用](https://github.com/imClumsyPanda/langchain-ChatGLM)

</details>

---

## 本地部署

- [Ollama](https://ollama.com/)：一键本地运行开源大模型（DeepSeek、Qwen、LLaMA 等），支持 macOS / Linux / Windows
- [Open WebUI](https://github.com/open-webui/open-webui)：配合 Ollama 使用的 Web 界面，类 ChatGPT 体验
- [动手学 Ollama（handy-ollama）](https://datawhalechina.github.io/handy-ollama/)：Datawhale 出品的 Ollama 中文教程，含可视化部署和应用案例
- [llama.cpp](https://github.com/ggerganov/llama.cpp)：纯 C/C++ 实现的 LLM 推理，支持 CPU 运行

---

## AI 应用开发平台

| 名称 | GitHub | 说明 |
| ---- | ---- | ---- |
| Dify | [langgenius/dify](https://github.com/langgenius/dify) ⭐145k | LLM 应用开发平台，可视化编排工作流，支持 RAG 和 Agent |
| RAGFlow | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) ⭐82k | 基于深度文档理解的开源 RAG 引擎 |
| FastGPT | [labring/FastGPT](https://github.com/labring/FastGPT) ⭐28k | 开源知识库问答系统，可视化工作流编排 |
| Coze（扣子） | [coze.cn](https://www.coze.cn/) | 字节跳动 Bot 开发平台，零代码创建 AI 机器人 |

---

## AI Agent 资源

- [从零开始构建智能体（hello-agents）](https://github.com/datawhalechina/hello-agents)：Datawhale 出品，涵盖 ReAct 范式、Coze、LangGraph、多智能体协作等
- [AI Agent 速成指南（ai-agents-from-zero）](https://github.com/didilili/ai-agents-from-zero)：系统教程 + 实战项目，覆盖 LangChain / LangGraph / Coze / Dify / MCP

---

## MCP 资源

[Model Context Protocol（MCP）](https://modelcontextprotocol.io/)是 Anthropic 发起的开放协议，用于将 LLM 与外部工具和数据源连接。

- [Awesome-MCP-ZH](https://github.com/yzfly/awesome-mcp-zh)：MCP 中文资源精选，含指南、Servers、Clients 汇总
- [MCP 中文文档](https://mcp.fleeto.us/)：Model Context Protocol 规范中文翻译
- [MCP 中文站](https://mcpcn.com/)：MCP 入门介绍与生态导航
- [MCP 编程极速入门](https://github.com/liaokongVFX/MCP-Chinese-Getting-Started-Guide)：面向开发者的 MCP 快速上手指南

---

## Prompt 工程

- [Prompt Engineering Guide 中文版](https://www.promptingguide.ai/zh)：最全面的提示词工程指南，涵盖技巧、应用和模型指南
- [面向开发者的 Prompt 工程（吴恩达 × OpenAI）](https://prompt-engineering.xiniushu.com/)：官方课程中文翻译版
- [阿里百炼 Prompt 指南](https://help.aliyun.com/zh/model-studio/prompt-engineering-guide)：阿里云大模型服务平台的实用提示词设计指南
- [ChatGPT 中文调教指南](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)：中文 Prompt 精选集

---

## 学习课程与视频

| 名称  | 说明 | 链接
| ----  | ----  | ----
| 【李宏毅】2025 生成式 AI 导论 | 42 集全，涵盖 LLM、AI Agent、神经网络等 | [B站](https://www.bilibili.com/video/BV1mXpuz7E9v/)
| 【吴恩达】LLM 大模型教程（2025） | 大模型入门到进阶，附课件代码 | [B站](https://www.bilibili.com/video/BV1sMEyzhEM3/)
| 【李宏毅】ChatGPT 是怎么炼成的 | GPT社会化的过程 | [B站](https://www.bilibili.com/video/BV1U84y167i3/?p=1&vd_source=71b548de6de953e10b96b6547ada83f2) 
| 【李沐】InstructGPT 背后的技术 | InstructGPT 论文精读| [B站](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.788&vd_source=71b548de6de953e10b96b6547ada83f2)
|【Aston Zhang】Chain of Thought | 结合in-context fewshot 改善推理能力 | [B站](https://www.bilibili.com/video/BV1t8411e7Ug/?spm_id_from=333.788&vd_source=1e55c5426b48b37e901ff0f78992e33f)
| LLM CS324 | 斯坦福大语言模型课程，涵盖建模、伦理和系统方面 |[链接](https://stanford-cs324.github.io/winter2022/)
| 吴恩达大模型课程中文版（llm-cookbook） | ⭐24.2k，11 门课程翻译 + 可运行代码 | [GitHub](https://github.com/datawhalechina/llm-cookbook)
| 动手学大模型应用开发（llm-universe） | 面向小白的 LLM 应用开发教程 | [GitHub](https://github.com/datawhalechina/llm-universe)
| 大模型基础（so-large-lm） | 从数据准备到训练策略的全链路知识 | [GitHub](https://github.com/datawhalechina/so-large-lm)

---

## 综合资源索引

- [Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)：⭐22.6k，最全面的中文 LLM 资源索引，涵盖模型、数据集、微调框架、应用等
- [awesome-chatgpt-zh](https://github.com/yzfly/awesome-chatgpt-zh)：⭐11.5k，ChatGPT 中文指南，含工具、应用、开发资源
- [中国大模型列表（awesome-LLMs-In-China）](https://github.com/wgwang/awesome-LLMs-In-China)：国产大模型全景图
- [通往 AGI 之路](https://wiki.waytoagi.com/)：中文 AI 知识社区，含 LLM 开源模型及数据集集合
