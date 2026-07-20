# ChatGPT-Chinese-Tutorial <br> ChatGPT 与大语言模型（LLM）中文学习资料汇总


[![Awesome](./logo.png)]() 
[![Code License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/yzfly/awesome-chatgpt-zh/blob/main/LICENSE)

> 本仓库持续更新中文 LLM 学习资源，涵盖国产大模型、开源模型、论文精读、微调部署、强化学习训练、AI Agent、RAG、MCP、A2A、上下文工程、Prompt 工程、推理优化、安全对齐、自进化智能体等内容。
>
> 最近更新：2026 年 7 月 20 日

---

<!-- WEEKLY_CHINESE_LLM_UPDATE:START -->
## 每周精选更新

> 自动生成时间：2026-07-20。每周从全网筛选近期 ChatGPT / LLM / Agent / RAG / MCP / 后训练 / 多模态等高价值学习资源；候选资料不限中文，英文资料也会纳入，最终统一用中文表达学习价值。

| 推荐 | 方向 | 资源 | 来源 | 推荐理由 |
| ---- | ---- | ---- | ---- | ---- |
| 1 | RAG / AI 搜索 | [Aryansingh009/awesome-llm-knowledge-systems](https://github.com/Aryansingh009/awesome-llm-knowledge-systems) | GitHub | 英文资源，建议关注：Map the evolution of LLM knowledge systems from prompt engineering to harness engineering with a comprehensive guide for modern RAG and context architectures.（⭐ 0；更新 2026-07-19） |
| 2 | RAG / AI 搜索 | [AJiang4015/RAG-demo](https://github.com/AJiang4015/RAG-demo) | GitHub | 英文资源，建议关注：RAG (Retrieval-Augmented Generation) tutorial code - document loading, text splitting, embedding, FAISS index, advanced retrieval strategies (HyDE, Multi-Query, Hybrid Search, Rera...（⭐ 0；更新 2026-07-08） |
| 3 | AI Agent / 工具调用 | [Rahulchaube1/Rahul-Chaube-Skills](https://github.com/Rahulchaube1/Rahul-Chaube-Skills) | GitHub | 英文资源，建议关注：The most comprehensive open-source AI skills library for LLMs, AI agents, prompt engineering, deep research, and production AI systems.（⭐ 3；更新 2026-07-19） |
| 4 | AI Agent / 工具调用 | [LLM-Powered Agentic AI for 5G/6G Networks: A Tutorial and Survey on Architectures, Protocols, and Standardization](http://arxiv.org/abs/2607.16066v1) | arXiv | 英文资源，建议关注：Agentic Artificial Intelligence (AI), enabled by Large Language Models, marks a shift from rule-based automation toward autonomous, goal-driven control of Next-Generation Networks ...（paper；更新 2026-07-17） |
| 5 | AI Agent / 工具调用 | [BayesPO: Bayesian Prompt Optimization via Parallel-Tempered Gradient-Guided Discrete MCMC](http://arxiv.org/abs/2607.16001v1) | arXiv | 英文资源，建议关注：Prompt optimization adapts large language models (LLMs) without updating model parameters, but many automatic prompt optimizers remain heuristic search procedures over candidate in...（paper；更新 2026-07-17） |
| 6 | AI Agent / 工具调用 | [jxf20250119/QWQ-coding-agent-powered-by-Pi](https://github.com/jxf20250119/QWQ-coding-agent-powered-by-Pi) | GitHub | 英文资源，建议关注：An open-source desktop AI coding agent (Electron + Pi) that combines deep single-agent reasoning with multi-agent parallel orchestration, a self-improving autoresearch loop, and ti...（⭐ 0；更新 2026-07-13） |
| 7 | 后训练 / 强化学习 | [Dustin0420/llm-course-ch4-chinese-sentiment](https://huggingface.co/Dustin0420/llm-course-ch4-chinese-sentiment) | Hugging Face | 近期更新模型，tags: transformers, safetensors, bert, text-classification, arxiv:1910.09700, text-embeddings-inference, endpoints_compatible, region:us（likes 0；更新 2026-04-21） |
| 8 | AI Agent / 工具调用 | [When Do Multi-Agent Systems Help? An Information Bottleneck Perspective](http://arxiv.org/abs/2607.16133v1) | arXiv | 英文资源，建议关注：LLM powered multi-agent systems (MAS) have emerged as a promising paradigm for complex tasks. However, their advantages over single-agent systems (SAS) remain unclear, with perform...（paper；更新 2026-07-17） |

<!-- WEEKLY_CHINESE_LLM_UPDATE:END -->

---

## 目录

- [大模型产品与平台](#大模型产品与平台)
- [开源大模型](#开源大模型)
- [论文精读](#论文精读)
- [微调与训练](#微调与训练)
- [强化学习训练（RL for LLM）](#强化学习训练rl-for-llm)
- [自进化与自我改进](#自进化与自我改进)
- [本地部署与推理优化](#本地部署与推理优化)
- [AI 应用开发平台](#ai-应用开发平台)
- [RAG 资源](#rag-资源)
- [AI Agent 资源](#ai-agent-资源)
- [MCP 资源](#mcp-资源)
- [A2A 协议资源](#a2a-协议资源)
- [上下文工程（Context Engineering）](#上下文工程context-engineering)
- [AI 编程工具](#ai-编程工具)
- [Prompt 工程](#prompt-工程)
- [学习课程与视频](#学习课程与视频)
- [书籍与教材](#书籍与教材)
- [评测与基准](#评测与基准)
- [LLM 安全与对齐](#llm-安全与对齐)
- [综合资源索引](#综合资源索引)

---

## 大模型产品与平台

| 名称 | 开发者 | 链接 | 说明 |
| ---- | ---- | ---- | ---- |
| ChatGPT | OpenAI | [chat.openai.com](https://chat.openai.com/) | GPT-4o / GPT-5.5 系列，全球用户量最大的 AI 助手，400K 上下文 |
| Claude | Anthropic | [claude.ai](https://claude.ai/) | Claude 4 系列（Opus/Sonnet/Haiku/Fable），长上下文与编程能力突出 |
| DeepSeek | 深度求索 | [chat.deepseek.com](https://chat.deepseek.com/) | DeepSeek-V4 / R1，强推理能力，开源标杆 |
| 通义千问 | 阿里巴巴 | [tongyi.aliyun.com](https://tongyi.aliyun.com/) | Qwen3.7 系列，国内综合能力领先，全球下载量最大的开源 LLM 家族 |
| 文心一言 | 百度 | [yiyan.baidu.com](https://yiyan.baidu.com/) | ERNIE 4.5 / 5.1 系列，中文理解能力突出 |
| Kimi | 月之暗面 | [kimi.moonshot.cn](https://kimi.moonshot.cn/) | Kimi K2.6，超长上下文处理，Agent 能力强 |
| 豆包 | 字节跳动 | [www.doubao.com](https://www.doubao.com/) | Doubao 模型，多模态能力强 |
| 智谱清言 | 智谱 AI (Z.ai) | [chatglm.cn](https://chatglm.cn/) | GLM-5.1 系列，清华技术背景，MoE 架构 |
| 腾讯元宝 | 腾讯 | [yuanbao.tencent.com](https://yuanbao.tencent.com/) | 混元大模型，集成微信生态 |
| 小米 MiMo | 小米 | [mimo.mi.com](https://mimo.mi.com/) | MiMo-V2.5 系列，推理与 Agent 能力突出，MIT 开源 |
| MiniMax (海螺 AI) | 稀宇科技 | [hailuoai.com](https://hailuoai.com/) | MiniMax-M3，原生多模态，100 万 token 上下文 |

---

## 开源大模型

### 旗舰模型

| 模型 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| DeepSeek-V4-Pro | 深度求索 | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) ⭐103k | 1.6T MoE（49B 激活），100 万上下文，LiveCodeBench 93.5%，MIT 开源 |
| DeepSeek-R1 | 深度求索 | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) ⭐92k | 671B MoE，RL 驱动的长链推理，比肩 OpenAI o1，含 1.5B-70B 蒸馏版 |
| Qwen3.7-Max | 阿里巴巴 | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) ⭐27k | Agent 时代旗舰，代码/工作流自动化/持续自主执行 |
| Qwen3.6 | 阿里巴巴 | [QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6) ⭐3.5k | 27B Dense + 35B-A3B MoE，编码能力达旗舰水平，Apache 2.0 |
| Qwen3.5 | 阿里巴巴 | [QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5) ⭐6k | 0.8B-397B 全系列 8 个尺寸，原生多模态，262K 上下文 |
| Qwen3 | 阿里巴巴 | [QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) ⭐27k | 0.6B-235B，支持思考/非思考模式切换，Apache 2.0 |
| GLM-5.1 | 智谱 AI (Z.ai) | [zai-org/GLM-5](https://github.com/zai-org/GLM-5) | 754B MoE，SWE-Bench Pro SOTA，开放权重 |
| GLM-4-32B | 智谱 AI | [zai-org/GLM-4](https://github.com/zai-org/GLM-4) ⭐3k | 含 GLM-Z1 推理模型和 Rumination 深度沉思模型，MIT 开源 |
| DeepSeek-V4-Flash | 深度求索 | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) ⭐103k | 284B MoE（13B 激活），100 万上下文，高性价比混合推理模型 |
| Kimi K2.7 Code | 月之暗面 | [MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) ⭐18k | 1T MoE，编程性价比最高，推理 token 降 30%，MCP 工具调用提升显著 |
| MiMo-V2.5-Pro | 小米 | [XiaomiMiMo/MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash) | 309B MoE（A15B），推理与 Agent 性能接近 Kimi-K2-Thinking，MIT 开源 |
| MiniMax-M3 | 稀宇科技 | -- | 原生多模态旗舰，稀疏注意力支持 100 万上下文，视频理解 + 编码 + Agent |
| LLaMA 4 | Meta | [meta-llama/llama-models](https://github.com/meta-llama/llama-models) | 全球最流行的开源基座模型 |

### 端侧 / 小模型

| 模型 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| MiniCPM5-1B | 清华 OpenBMB | [OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM) ⭐5k | 1B SOTA 端侧模型，支持长上下文和工具调用 |
| MiniCPM-o 4.5 | 清华 OpenBMB | [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) ⭐25.4k | 8B 全能多模态，全双工音视频对话，手机端可运行 |
| Qwen3.5-0.8B/2B | 阿里巴巴 | [QwenLM/Qwen3.5](https://github.com/QwenLM/Qwen3.5) | 轻量原生多模态，适合边缘部署 |
| DeepSeek-R1-Distill | 深度求索 | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | 1.5B/7B/8B/14B/32B/70B 蒸馏版，推理能力平民化 |
| MiMo-V2-Flash | 小米 | [XiaomiMiMo/MiMo-V2-Flash](https://github.com/XiaomiMiMo/MiMo-V2-Flash) | 309B MoE（A15B），高效推理与 Agent 端侧模型，MIT 开源 |

### 多模态模型

| 模型 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| InternVL3.5 | 上海 AI 实验室 | [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) ⭐8k | 8B/78B/241B-A28B 视觉语言模型，Apache 2.0 |
| Qwen3-Omni | 阿里巴巴 | -- | 全能多模态，支持 any-to-any 交互（文本/图像/音频） |
| Qwen3-TTS | 阿里巴巴 | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) ⭐17k | 开源语音合成，支持流式生成和声音克隆 |
| Wan 2.1 | 阿里巴巴 | [Wan-AI/Wan2.1](https://github.com/Wan-AI/Wan2.1) ⭐15k | 视频生成模型，VBench 第一（86.22%），超越 Sora，Apache 2.0 |
| HunyuanVideo 1.5 | 腾讯 | [Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo) ⭐10k | 开源视频生成，支持文本/图片生成视频 |
| GOT-OCR 2.0 | 阶跃星辰 | [stepfun-ai/GOT-OCR2_0](https://github.com/stepfun-ai/GOT-OCR2_0) ⭐5k | 通用 OCR，视觉语言架构，多语言文档识别 |
| Janus-Pro-7B | 深度求索 | -- | 统一多模态（图像理解 + 图像生成），MIT 开源 |

### 代码专用模型

| 模型 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| Qwen3-Coder | 阿里巴巴 | [QwenLM/Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) ⭐5k | 480B-A35B MoE，SWE-Bench 编程基准领先 |
| Kimi K2.7 Code | 月之暗面 | [MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2) ⭐18k | 1T MoE 编程专用，推理 token 降 30%，编程性价比最高的开源模型 |
| DeepSeek-V4-Pro | 深度求索 | -- | LiveCodeBench 93.5%，算法编程专家，MIT 开源 |
| MiMo Code | 小米 | [XiaomiMiMo/MiMo-Code](https://github.com/XiaomiMiMo/MiMo-Code) | 小米终端 AI 编程助手，MIT 开源 |

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

### 2025-2026 前沿论文

| 名称 | 说明 | 学习资料
| ---- | ---- | ----
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | DeepSeek-V3 MoE 架构与低成本训练 | [GitHub](https://github.com/deepseek-ai/DeepSeek-V3)
| [A Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046) | 自进化智能体综述（TMLR 2026） | [Awesome-Self-Evolving-Agents](https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents)
| [SELF: Self-Evolution with Language Feedback](https://arxiv.org/abs/2310.00533) | 通过语言反馈实现 LLM 自我进化 | --
| [Self-Evolved Reward Learning for LLMs](https://arxiv.org/abs/2411.00418) | 自进化奖励学习，减少人类标注依赖 | --
| [EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](https://arxiv.org/abs/2510.16079) | 经验驱动的自进化 Agent 闭环框架 | --
| [SkillRL: Evolving Agents via Recursive Skill-Augmented RL](https://arxiv.org/abs/2602.08234) | 递归技能增强的 RL 进化 Agent | --
| [Self-Improvement of LLMs: A Technical Overview](https://arxiv.org/abs/2603.25681) | LLM 自我改进技术综述（2026） | --
| [QA-GraphRAG: Query-Adaptive Graph RAG](https://hub.baai.ac.cn/view/54807) | 面向图检索增强生成的查询自适应框架（VLDB 2026） | --
| [DeepSeek-V4 Technical Report](https://www.qbitai.com/2026/04/406809.html) | DeepSeek-V4 1.6T MoE 架构，100 万上下文，484 天换代全公开 | [GitHub](https://github.com/deepseek-ai/DeepSeek-V3)
| [InfiAgent: Self-Evolving Pyramid Agent](https://arxiv.org/abs/2509.22502) | DAG 金字塔多 Agent 框架，agent-as-a-tool 自演化机制 | --
| [Self-Evolving Agents 综述（IEEE CIM 2026）](https://zhuanlan.zhihu.com/p/2046485780617506861) | 近 400 篇文献，统一自进化 Agent 学术定义与四大分支 | --

---

## 微调与训练

### 微调框架

- [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)：⭐72k，一站式 LLM 微调框架，支持 100+ 模型（Qwen、DeepSeek、LLaMA、GLM 等），提供 LoRA/QLoRA/全参微调，内置 WebUI，ACL 2024 论文
- [Colossal Chat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat)：基于 LLaMA 的完整 RLHF 训练流程，支持 DeepSeek-V3/R1 LoRA 微调
- [DeepSpeed Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat/chinese)：微软官方开源，一键式 RLHF 训练——[训练代码](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [ms-swift v4.0](https://github.com/modelscope/ms-swift)：阿里 ModelScope 出品，2026 年 3 月发布 v4.0 大版本，支持 600+ LLM 和 300+ 多模态模型，含 GRPO/GKD 训练（AAAI 2025 论文）

### 微调教程

- [开源大模型食用指南（self-llm）](https://github.com/datawhalechina/self-llm)：⭐30.8k，针对中文开发者的 Linux 环境下开源模型部署与微调教程，覆盖 50+ 主流模型
- [从零开始构建大模型（happy-llm）](https://github.com/datawhalechina/happy-llm)：⭐31k，从 NLP 基础到手搓 LLaMA2，涵盖 Pretrain、SFT、PEFT 全流程
- [DIY-LLM 全栈构建课程](https://github.com/datawhalechina/diy-llm)：Datawhale 出品，覆盖预训练数据工程、Tokenizer、Transformer、MoE、GPU 编程（CUDA/Triton）、分布式训练、Scaling Laws、推理优化及对齐（SFT/RLHF/GRPO），含 6 个渐进式作业
- [MiniMind](https://github.com/jingyaogong/minimind)：⭐40k+，2 小时从零训练 64M 参数 LLM，覆盖 Pretrain/SFT/LoRA/RLHF-DPO/RLAIF（PPO/GRPO/SPO）/Agentic RL 全流程代码，含视觉版 [MiniMind-V](https://github.com/jingyaogong/minimind-v) 和全模态版 [MiniMind-O](https://github.com/jingyaogong/minimind-o)
- [NLP 到 LLM 算法全栈教程（base-llm）](https://github.com/datawhalechina/base-llm)：Datawhale 出品，从 NLP 基础到 LLM 的算法全栈教程，含 RLHF 专题
- [大模型算法实战教程（llm-algo-leetcode）](https://github.com/datawhalechina/llm-algo-leetcode)：面向大模型入门到进阶的算法实战，覆盖原理讲解、测试用例与 CUDA/Triton 实战
- [LLM/RL 算法原理图（LLM-RL-Visualized）](https://github.com/changyeyu/LLM-RL-Visualized)：100+ 原创 LLM/RL 原理图，涵盖 Transformer/SFT/DPO/GRPO/RLHF/RAG，配套《大模型算法》书籍

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

## 强化学习训练（RL for LLM）

### 训练框架

| 名称 | GitHub | 说明 |
| ---- | ---- | ---- |
| OpenRLHF | [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) ⭐8.9k | 高性能 Agentic RL 框架，Ray + vLLM 加速，支持 PPO/DAPO/REINFORCE++/Async RL/VLM RLHF |
| OpenRLHF-M | [OpenRLHF/OpenRLHF-M](https://github.com/OpenRLHF/OpenRLHF-M) | 多模态 RLHF 专用框架，支持 Qwen3.5 等视觉语言模型端到端 RL 训练 |
| TRL | [huggingface/trl](https://github.com/huggingface/trl) | HuggingFace 官方 RL 训练库，支持 PPO/DPO/GRPO/KTO 等多种算法 |
| veRL | [volcengine/verl](https://github.com/volcengine/verl) | 字节跳动火山引擎出品，灵活的 RL 训练框架，支持 FSDP/Megatron 后端 |
| DeepSpeed Chat | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | 微软一键式 RLHF 训练，InstructGPT 全流程复现 |

### 核心算法与教程

| 算法 | 说明 | 资源 |
| ---- | ---- | ---- |
| RLHF (PPO) | 基于人类反馈的强化学习，ChatGPT 核心技术 | [HuggingFace RLHF 博客](https://huggingface.co/blog/rlhf)、[InstructGPT 论文精读（B站）](https://www.bilibili.com/video/BV1hd4y187CR/) |
| GRPO | DeepSeek-R1 使用的群体相对策略优化，无需 Critic 模型 | [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)、[GRPO 深度解析](https://cameronrwolfe.substack.com/p/grpo) |
| GRPO++ | GRPO 实战技巧集合，让 RL 真正 work | [GRPO++ 技巧详解](https://cameronrwolfe.substack.com/p/grpo-tricks) |
| REINFORCE++ | ProRL V2 使用的算法，训练 SOTA 推理模型 | [OpenRLHF 文档](https://github.com/OpenRLHF/OpenRLHF) |
| DPO | 直接偏好优化，将奖励模型"吸收"进 LM 优化目标 | [DPO 原理详解（GitBook）](https://yeasy.gitbook.io/llm_internals/di-er-bu-fen-xun-lian-pian/08_alignment/8.3_dpo) |
| KTO | 基于 Kahneman-Tversky 的对齐算法，仅需二元反馈 | [TRL 文档](https://huggingface.co/docs/trl/) |
| Agentic RL | Agent 环境中的 RL 训练，多步交互奖励 | [LLM Post-Training 全景指南](https://qingkeai.online/archives/RLHF-GRPO-AgenticRL) |
| 对齐技术综述 | RLHF/RLAIF/PPO/DPO 全家桶 | [一文看尽 LLM 对齐技术（CSDN）](https://blog.csdn.net/pythonhy/article/details/141192664) |
| RLVR | 基于可验证奖励的 RL，2026 年新方向 | [LLM Post-Training 全景指南](https://qingkeai.online/archives/RLHF-GRPO-AgenticRL) |
| RL Post-Training 全景 | 从 RLHF 到 GRPO 再到 Agentic RL | [HuggingFace 指南](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms) |
| MiniMind RL 全流程 | 从零实现 RLHF/DPO/PPO/GRPO/Agentic RL | [MiniMind GitHub](https://github.com/jingyaogong/minimind) |
| LLM-RL 可视化原理图 | 100+ 原创原理图，直观理解 RL 算法 | [LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized) |

---

## 自进化与自我改进

自进化（Self-Evolution）是 2025-2026 年 Agent 研究的核心方向，目标是让智能体在无需人类持续干预下自主提升能力。

### 四大进化路径

| 路径 | 说明 | 代表工作 |
| ---- | ---- | ---- |
| 模型进化 | Agent 利用自生成数据或学习课程精炼自身参数 | SELF、Self-Evolved Reward Learning |
| 记忆进化 | 从历史经验中学习，存储并检索以指导未来行动 | Letta/MemGPT、MemSkill |
| 工具进化 | 自主创建、精炼和复用工具以扩展能力 | SkillRL |
| 工作流进化 | 自主优化执行流水线和协作结构 | CORAL、EvolveR |

### 前沿论文

| 名称 | 说明 | 资源 |
| ---- | ---- | ---- |
| 自进化 Agent 综述（IEEE CIM 2026） | 统一学术定义，划分单体经验闭环/技能演化/多体协同/元算法开放进化四大分支，汇总近 400 篇文献 | [知乎解读](https://zhuanlan.zhihu.com/p/2046485780617506861) |
| 自进化 Agent 综述（TMLR 2026） | 77 页综述，系统回答"什么/何时/如何/哪里进化"，迈向人工超级智能 | [arXiv](https://arxiv.org/abs/2507.21046) |
| InfiAgent（arXiv 2025） | 基于 DAG 的金字塔多 Agent 框架，agent-as-a-tool 机制，双重审计 + 智能路由 + 自演化 | [论文](https://arxiv.org/abs/2509.22502) |

### 核心资源

- [Awesome-Self-Evolving-Agents](https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents)：⭐251，自进化 Agent 论文、benchmark 和开源项目合集
- [自进化 Agent 综述深度解读（中文）](https://xiao-zi-chen.github.io/2026/03/23/self-evolving-agent-survey-deep-dive/)：超详细中文解读，三大范式——模型中心、环境中心、协同进化
- [Self-Evolving Agent 经典论文介绍（知乎）](https://zhuanlan.zhihu.com/p/2046485780617506861)：2026 年经典论文汇总与技术路线分析
- [hello-generic-agent](https://github.com/datawhalechina/hello-generic-agent)：Datawhale 出品，自进化智能体入门，涵盖记忆系统、上下文压缩、自进化机制
- [Letta（原 MemGPT）](https://github.com/letta-ai/letta)：⭐21k，有状态 Agent 平台，持久记忆 + 自我改进，Agent 可重写自身记忆/技能/提示
- [VoltAgent/awesome-ai-agent-papers](https://github.com/VoltAgent/awesome-ai-agent-papers)：2026 年 AI Agent 研究论文精选合集，涵盖记忆、评估、工作流、自主系统

---

## 本地部署与推理优化

### 部署工具

- [Ollama](https://ollama.com/)：一键本地运行开源大模型（DeepSeek、Qwen、LLaMA 等），支持 macOS / Linux / Windows
- [Open WebUI](https://github.com/open-webui/open-webui)：配合 Ollama 使用的 Web 界面，类 ChatGPT 体验
- [动手学 Ollama（handy-ollama）](https://datawhalechina.github.io/handy-ollama/)：Datawhale 出品的 Ollama 中文教程，含可视化部署和应用案例
- [llama.cpp](https://github.com/ggerganov/llama.cpp)：纯 C/C++ 实现的 LLM 推理，支持 CPU 运行

### 推理框架

| 名称 | 说明 | 资源 |
| ---- | ---- | ---- |
| vLLM | 高性能 LLM 推理引擎，PagedAttention + Continuous Batching，显存利用率 95%，企业级高并发首选 | [vLLM GitHub](https://github.com/vllm-project/vllm) |
| SGLang | RadixAttention 实现 KV 缓存复用，多轮对话吞吐量比 vLLM 高 5 倍，Agent 场景理想选择 | [SGLang GitHub](https://github.com/sgl-project/sglang) |
| TensorRT-LLM | NVIDIA 官方推理优化，FP8/INT4 量化，Blackwell 架构原生支持 | [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| LMDeploy | 上海 AI 实验室出品，支持量化部署和推理加速 | [LMDeploy GitHub](https://github.com/InternLM/lmdeploy) |

### 推理优化教程

- [vLLM vs SGLang vs TensorRT-LLM 综合对比（阿里云）](https://developer.aliyun.com/article/1686693)：三大框架技术原理、性能基准与选型建议
- [主流大模型推理部署框架全解析（知乎）](https://zhuanlan.zhihu.com/p/1937266323156607848)：一文梳理主流框架
- [大模型推理加速全景技术指南（博客园）](https://www.cnblogs.com/SVicen/p/19281160)：全面梳理量化、剪枝、蒸馏等优化技术栈
- [八大推理引擎横向拆解](https://quant67.com/post/llm-infra/13-vllm-sglang/13-vllm-sglang.html)：vLLM/SGLang/TensorRT-LLM/TGI/LMDeploy/MindIE 从架构到性能全面对比
- [InfiniTensor CUDA 训练营](https://github.com/InfiniTensor/Learning-CUDA)：启元实验室 + 清华大学，大模型推理相关 CUDA 编程实战

---

## AI 应用开发平台

| 名称 | GitHub | 说明 |
| ---- | ---- | ---- |
| Dify | [langgenius/dify](https://github.com/langgenius/dify) ⭐145k | LLM 应用开发平台，可视化编排工作流，支持 RAG 和 Agent，全球最高 Star 的 LLMOps 平台 |
| Coze Studio（扣子） | [coze-dev/coze-studio](https://github.com/coze-dev/coze-studio) ⭐21k | 字节跳动 2025 年 7 月开源，全栈 AI Agent 开发工具，Apache 2.0，Docker 一键部署 |
| RAGFlow | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) ⭐82k | 基于深度文档理解的开源 RAG 引擎 |
| FastGPT | [labring/FastGPT](https://github.com/labring/FastGPT) ⭐28k | 开源知识库问答系统，可视化工作流编排 |
| MaxKB | [1Panel-dev/MaxKB](https://github.com/1Panel-dev/MaxKB) ⭐20.6k | 企业级智能体平台，支持 DeepSeek 私有化部署和企业级权限治理 |
| DB-GPT | [eosphoros-ai/DB-GPT](https://github.com/eosphoros-ai/DB-GPT) ⭐17k | AI 数据应用框架，Text-to-SQL、数据分析和多 Agent 协作 |

---

## RAG 资源

### RAG 框架

| 名称 | GitHub | 说明 |
| ---- | ---- | ---- |
| RAGFlow | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) ⭐82k | 深度文档理解 RAG 引擎，支持多模态 PDF 解析，内置 Agent 能力 |
| LlamaIndex | [run-llama/llama_index](https://github.com/run-llama/llama_index) ⭐50k | 数据框架，支持结构化/非结构化数据索引与检索 |
| Langchain-Chatchat | [chatchat-space/Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) ⭐37.5k | 中文 RAG 与 Agent 应用，支持 ChatGLM/Qwen/Llama 等国产模型 |
| GraphRAG | [microsoft/graphrag](https://github.com/microsoft/graphrag) ⭐33.6k | 微软模块化图 RAG，基于知识图谱做全局摘要和社区检测 |
| LightRAG | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) ⭐28k | 轻量级图谱 RAG（EMNLP 2025），适合法律/医疗/金融领域 |
| QAnything | [netease-youdao/QAnything](https://github.com/netease-youdao/QAnything) ⭐14k | 网易有道，支持任意格式文件的本地知识库问答 |

### 2026 RAG 技术趋势

2026 年 RAG 已进入"Agentic RAG"时代——检索不再是单次流程，而是"思考→检索→再思考→再检索→行动"的循环。三大创新范式：

| 方向 | 说明 | 代表项目 |
| ---- | ---- | ---- |
| GraphRAG | 通过知识图谱实现多跳推理 | [microsoft/graphrag](https://github.com/microsoft/graphrag)、[QA-GraphRAG（VLDB 2026）](https://hub.baai.ac.cn/view/54807) |
| Agentic RAG | 将检索集成到 Agent 决策循环中 | RAGFlow + Agent、Dify 工作流 |
| Memory-Augmented RAG | 建立长期记忆系统，上下文 + 检索 + 推理统一 | Letta、WeKnora |

### RAG 学习教程

- [All-in-RAG（RAG 技术全栈指南）](https://github.com/datawhalechina/all-in-rag)：⭐2.5k，Datawhale 出品，系统化 RAG 全栈教程，从理论到生产级系统
- [Agentic RAG 3.0 核心升级点全解析（腾讯云）](https://cloud.tencent.com/developer/article/2650548)：百万上下文 + 多模态 GraphRAG + Agentic 检索
- [2026 年 RAG 技术最新进展与落地实践](https://segmentfault.com/a/1190000047621497)：从向量检索到 GraphRAG 与 Agentic RAG 的演进
- [RAG 技术演进全解析（CSDN）](https://gitcode.csdn.net/69d1bbcf0a2f6a37c59d17da.html)：2026 年 RAG 从向量检索到 GraphRAG 与 Agentic RAG

### 向量数据库

| 名称 | GitHub | 说明 |
| ---- | ---- | ---- |
| Milvus | [milvus-io/milvus](https://github.com/milvus-io/milvus) ⭐43k | 中国团队开发的全球领先向量数据库，分布式架构，支持数亿级向量检索 |
| Qdrant | [qdrant/qdrant](https://github.com/qdrant/qdrant) ⭐30k | Rust 编写的高性能向量数据库，生产环境首选之一 |
| Chroma | [chroma-core/chroma](https://github.com/chroma-core/chroma) ⭐27k | 轻量级嵌入式向量数据库，适合原型开发 |

---

## AI Agent 资源

### 中国团队主导的 Agent 框架

| 名称 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| MetaGPT | 深度智慧 | [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) ⭐62k | 多 Agent 模拟软件公司（产品经理→架构师→程序员），有中文文档 |
| OpenManus | MetaGPT 团队 | [FoundationAgents/OpenManus](https://github.com/FoundationAgents/OpenManus) ⭐29k | 3 小时复刻 Manus 的开源替代品，支持代码执行/文件处理/网页搜索 |
| ChatDev 2.0 | OpenBMB / 清华 | [OpenBMB/ChatDev](https://github.com/OpenBMB/ChatDev) ⭐27k | 多 Agent 虚拟软件公司，2026 年发布零代码多 Agent 编排平台 |
| Coze Studio | 字节跳动 | [coze-dev/coze-studio](https://github.com/coze-dev/coze-studio) ⭐21k | 全栈 Agent 开发工具，Golang + React，Apache 2.0 |
| CAMEL / OWL | CAMEL-AI | [camel-ai/camel](https://github.com/camel-ai/camel) ⭐13k / [camel-ai/owl](https://github.com/camel-ai/owl) ⭐11k | 角色扮演多 Agent 框架；OWL 在 GAIA 基准排名开源第一 |
| Qwen-Agent | 阿里巴巴 | [QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) ⭐12k | 基于 Qwen 模型的智能体框架，支持 Function Calling、MCP、RAG |
| AgentScope | 阿里巴巴 | [modelscope/agentscope](https://github.com/modelscope/agentscope) ⭐12k | 多 Agent 协作平台，支持数十个 Agent 交互 |
| MindSearch | 上海 AI 实验室 | [InternLM/MindSearch](https://github.com/InternLM/MindSearch) | 多 Agent 搜索引擎（媲美 Perplexity.ai Pro），处理 300+ 网页 |

### 国际主流 Agent 框架

| 名称 | 开发者 | GitHub | 说明 |
| ---- | ---- | ---- | ---- |
| AutoGPT | Significant Gravitas | [Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) ⭐170k | AI Agent 运动先驱，含可视化工作流构建器 |
| LangChain | LangChain Inc. | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) ⭐123k | LLM 应用开发框架，生态最成熟，有[中文文档](https://python.langchain.com.cn/docs/) |
| OpenHands | All Hands AI | [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) ⭐78k | 开源 AI 编程 Agent（原 OpenDevin），SWE-Bench Verified 53%+，MIT 开源 |
| AutoGen | Microsoft | [microsoft/autogen](https://github.com/microsoft/autogen) ⭐53k | 多 Agent 对话式协作，已升级为 Microsoft Agent Framework |
| CrewAI | CrewAI Inc. | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) ⭐44k | 基于角色的多 Agent 团队协作，上手最快 |
| Agno | Agno | [agno-agi/agno](https://github.com/agno-agi/agno) ⭐29k | 多 Agent 框架，含运行时和控制面板 |
| LangGraph | LangChain Inc. | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) ⭐25k | 基于图的有状态 Agent 工作流，适合复杂多步骤任务 |
| Letta | Letta AI | [letta-ai/letta](https://github.com/letta-ai/letta) ⭐21k | 有状态 Agent 平台（原 MemGPT），持久记忆 + 自主学习进化 |
| smolagents | Hugging Face | [huggingface/smolagents](https://github.com/huggingface/smolagents) | Code-first Agent 框架，Agent 直接写 Python 而非 JSON 工具调用，轻量简洁 |
| PydanticAI | Pydantic | [pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai) ⭐16k | 类型安全 Agent 框架，FastAPI 风格 |

### Agent 学习教程

- [从零开始构建智能体（hello-agents）](https://github.com/datawhalechina/hello-agents)：⭐59.1k，Datawhale 出品，覆盖 ReAct 范式、Coze、LangGraph、多智能体协作、Agentic RL
- [Agentic AI 中文教程](https://github.com/datawhalechina/agentic-ai)：吴恩达 DeepLearning.AI Agentic AI 系列课程中文翻译与知识整理
- [AI Agent 速成指南](https://github.com/didilili/ai-agents-from-zero)：系统教程 + 实战项目，覆盖 LangChain / LangGraph / Coze / Dify / MCP，含 NL2SQL 和 DeepAgents 多 Agent 项目
- [Agency-Agents-ZH](https://github.com/jnMetaCode/agency-agents-zh)：266 个即插即用 AI 专家角色，支持 18 种工具，含 50 个中国市场原创智能体（小红书/抖音/微信/飞书/钉钉）
- [Agent-Learning-Hub](https://github.com/datawhalechina/Agent-Learning-Hub)：AI Agent 学习路线与资料库
- [AI Agent 开发指南（AgentGuide）](https://github.com/adongwanai/AgentGuide)：LangGraph 实战、高级 RAG、大模型面试题库
- [LangChain 中文入门教程](https://github.com/liaokongVFX/LangChain-Chinese-Getting-Started-Guide)：LangChain 中文入门指南
- [Agentic AI 框架终极指南（火山引擎）](https://adg.csdn.net/69533d4f5b9f5f31781bfe2b.html)：20 大框架超详细分析，含选型建议
- [LangChain 500 页中文文档](https://www.langchain.asia/)：跟随官方同步更新

---

## MCP 资源

[Model Context Protocol（MCP）](https://modelcontextprotocol.io/) 是 Anthropic 发起的开放协议，现由 Linux 基金会旗下 Agentic AI Foundation 托管。截至 2026 年中已有 19,800+ 活跃公共 MCP 服务器，SDK 月下载量超 9,700 万次。OpenAI、Microsoft、AWS、Google 先后接入。2026-07-28 RC 版本实现协议无状态核心，远程服务器可在普通负载均衡器后运行。

### MCP 规范演进

| 版本 | 时间 | 关键特性 |
| ---- | ---- | ---- |
| 2025-11-25 (稳定版) | 2025.11 | OpenID Connect 发现、工具/资源/Prompt 图标元数据 |
| 2026-07-28 (RC) | 2026.07 | 无状态协议核心、Extensions 框架、Tasks、MCP Apps、授权强化 |

### 学习资源

- [awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers)：⭐89k，全球最大 MCP 服务器合集，收录 9200+ 服务器
- [Awesome-MCP-ZH](https://github.com/yzfly/awesome-mcp-zh)：MCP 中文资源精选，含指南、Servers、Clients 汇总
- [MCP 中文文档](https://mcp.fleeto.us/)：MCP 规范完整中文翻译
- [MCP 中文站](https://mcpcn.com/)：MCP 入门介绍与生态导航
- [MCP 编程极速入门](https://github.com/liaokongVFX/MCP-Chinese-Getting-Started-Guide)：面向开发者的 MCP 快速上手指南
- [MCP 2026 路线图解读](https://chenguangliang.com/posts/blog088_mcp-2026-roadmap-analysis/)：从本地工具到生产级 Agent 基础设施的演进分析

---

## A2A 协议资源

[Agent2Agent（A2A）协议](https://google.github.io/A2A/) 是 Google 发起的开放标准，用于标准化 AI Agent 之间的通信与协作。MCP 解决"LLM 如何访问数据和工具"，A2A 解决"Agent 如何与其他 Agent 对话"。

### 核心概念

| 概念 | 说明 |
| ---- | ---- |
| AgentCard | Agent 能力声明与发现机制 |
| Task | Agent 间通信的基本工作单元 |
| Artifact | Agent 交互产生的输出制品 |
| Message | Agent 间通信的消息格式 |

### 学习资源

- [A2A 协议中文站](https://a2acn.com/docs/introduction/)：协议介绍、快速开始、API 参考
- [A2A 协议完全指南（2026）](https://pengjiyuan.github.io/articles/a2a-protocol-2026/)：核心概念、Agent Card 机制、与 MCP 关系、实战开发
- [A2A 协议技术细节分析（知乎）](https://zhuanlan.zhihu.com/p/1893578344324379306)：Google A2A 技术细节与 MCP 关系
- [使用 ADK 实现 A2A Agent 完整开发指南](https://www.cnblogs.com/sing1ee/p/18985838/adk-a2a)：环境搭建、项目结构、服务端/客户端 Agent 开发
- [DeepLearning.AI A2A 课程](https://www.deeplearning.ai/courses/a2a-the-agent2agent-protocol)：官方入门课程
- [Google Codelabs: Getting Started with MCP, ADK and A2A](https://codelabs.developers.google.com/codelabs/currency-agent)：动手实验

---

## 上下文工程（Context Engineering）

上下文工程是 2026 年 LLM 应用开发的核心范式，通过系统化设计和管理 LLM 的输入上下文来最大化模型能力输出，是 Prompt 工程的全面升级。

### 五大核心策略

| 策略 | 说明 |
| ---- | ---- |
| Offload（转移） | 将复杂任务分解转移到多个上下文窗口 |
| Reduce（压缩） | 压缩冗余信息，保留关键语义 |
| Retrieve（检索） | 动态检索相关上下文注入 |
| Isolate（隔离） | 隔离不同类型上下文防止干扰 |
| Cache（缓存） | 缓存常用上下文片段提升效率 |

### 学习资源

- [大模型上下文工程权威指南](https://yeasy.gitbook.io/context_engineering_guide)：15 章全面教程，涵盖核心技术、高级架构和工程实践
- [Context Engineering 完整中文教程](https://github.com/xjthy001/Context-Engineering-CN)：750K+ 字，涵盖 RAG 系统/记忆架构/多智能体/认知工具/神经场论，含 12 周学习路径和即用模板
- [Awesome Context Engineering](https://github.com/yzfly/awesome-context-engineering)：上下文工程论文、工具和最佳实践精选合集
- [上下文工程实战指南](https://github.com/WakeUp-Jin/Practical-Guide-to-Context-Engineering)：大模型应用开发骨架思路，上下文工程为设计原则、Agent Harness 为构建目标
- [AWS 上下文工程实践](https://aws.amazon.com/cn/blogs/china/agentic-ai-infrastructure-practice-series-nine-context-engineering/)：Agentic AI 基础设施系列，结合 AWS Bedrock 的企业级实践
- [上下文工程：重塑大模型智能系统的技术革命](https://www.gogoai.com/blog/context-engineering/)：一文读懂技术全景
- [上下文工程解决 Agent 性能瓶颈](https://modelscope.csdn.net/68f6fe60a6dc56200e95f911.html)：大模型开发者必看实践指南
- [Harness Engineering（驾驭工程）](https://www.runoob.com/ai-agent/harness-engineering.html)：2026 年新范式，设计约束、反馈环和控制系统

---

## AI 编程工具

| 名称 | 开发者 | 链接 | 说明 |
| ---- | ---- | ---- | ---- |
| Claude Code | Anthropic | [claude.ai/code](https://claude.ai/code) | 终端 Agent，支持文件系统/终端/MCP 全访问，FrontierCode 基准领先 |
| Cursor | Anysphere | [cursor.com](https://www.cursor.com/) | AI 代码编辑器，3.3 版新增 Bugbot 自动修复和 Cloud Agent 后台运行 |
| TRAE | 字节跳动 | [Trae-AI/TRAE](https://github.com/Trae-AI/TRAE) | AI 原生 IDE，600 万+ 注册用户，SOLO 模式端到端自动编程 |
| OpenAI Codex | OpenAI | [openai.com/codex](https://openai.com/codex) | 400 万周活跃用户，GPT-5.5 驱动，400K 上下文，90+ 第一方插件 |
| Qoder | 阿里巴巴 | [qoder.alibaba.com](https://qoder.alibaba.com) | 智能体编程平台，500 万+ 用户，Quest 模式支持 Agent 自主开发 |
| 通义灵码 | 阿里巴巴 | [lingma.aliyun.com](https://lingma.aliyun.com) | 基于 Qwen 的 AI 编程助手，VS Code + JetBrains，百万级上下文 |
| 文心快码 (Comate) | 百度 | [comate.baidu.com](https://comate.baidu.com) | 100+ 编程语言支持，SPEC 规范驱动模式 |
| OpenHands | All Hands AI | [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands) ⭐78k | 开源自主 AI 编程 Agent（原 OpenDevin），可写代码/跑终端/提 PR，MIT 开源 |

### 2026 AI 编程趋势

2026 年 AI 编程的核心变化：Cloud Agent 后台自主运行（关掉笔记本仍在工作），团队常见模式为 Cursor 做主动开发 + Claude Code 做委托任务（重构/迁移/测试回填）。

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
| 【李宏毅】2026 春季机器学习 | 最新版，涵盖生成式 AI 时代下的 ML，含 Context Engineering | [课程主页](https://speech.ee.ntu.edu.tw/~hylee/ml/2026-spring.php)
| 【李宏毅】2025 生成式 AI 导论 | 42 集全，涵盖 LLM、AI Agent、神经网络等 | [B站](https://www.bilibili.com/video/BV1mXpuz7E9v/)
| 【吴恩达】LLM 大模型教程（2025） | 大模型入门到进阶，附课件代码 | [B站](https://www.bilibili.com/video/BV1sMEyzhEM3/)
| 【吴恩达】Agentic AI 智能体课程（2026） | 最新 Agent 开发课程，含 A2A 协议实践 | [DeepLearning.AI](https://www.deeplearning.ai/courses/a2a-the-agent2agent-protocol)
| 【吴恩达】Agent Skills with Anthropic（2026） | Agent 技能开发课程，含课件代码 | [B站](https://www.bilibili.com/video/BV1qv6eBZErD/)
| 【清华 NLP】刘知远大模型公开课 | 20 小时从入门到精通，含交叉领域应用 | [B站](https://www.bilibili.com/video/BV1UG411p7zv/)
| 【黑马程序员】大模型 RAG 与 Agent 实战 | 基于 LangChain 从提示词到 RAG+Agent 项目实战 | [B站](https://www.bilibili.com/video/BV1yjz5BLEoY/)
| Stanford CS336 | 2026 春季，从零构建语言模型，注重动手实践 | [课程主页](https://cs336.stanford.edu/)
| CMU 11-766: LLM Applications | 2026 春季研究生课程，LLM 核心技术落地 | [课程主页](https://cmu-llms.org/)
| 【李宏毅】ChatGPT 是怎么炼成的 | GPT社会化的过程 | [B站](https://www.bilibili.com/video/BV1U84y167i3/?p=1&vd_source=71b548de6de953e10b96b6547ada83f2) 
| 【李沐】InstructGPT 背后的技术 | InstructGPT 论文精读| [B站](https://www.bilibili.com/video/BV1hd4y187CR/?spm_id_from=333.788&vd_source=71b548de6de953e10b96b6547ada83f2)
|【Aston Zhang】Chain of Thought | 结合in-context fewshot 改善推理能力 | [B站](https://www.bilibili.com/video/BV1t8411e7Ug/?spm_id_from=333.788&vd_source=1e55c5426b48b37e901ff0f78992e33f)
| LLM CS324 | 斯坦福大语言模型课程，涵盖建模、伦理和系统方面 |[链接](https://stanford-cs324.github.io/winter2022/)
| 吴恩达大模型课程中文版（llm-cookbook） | ⭐24.2k，11 门课程翻译 + 可运行代码 | [GitHub](https://github.com/datawhalechina/llm-cookbook)
| 动手学大模型应用开发（llm-universe） | 面向小白的 LLM 应用开发教程 | [GitHub](https://github.com/datawhalechina/llm-universe)
| 大模型基础（so-large-lm） | 从数据准备到训练策略的全链路知识 | [GitHub](https://github.com/datawhalechina/so-large-lm)
| 大模型技术实战（llm-action） | ⭐18.1k，大模型工程化与应用落地实战经验，含量化/剪枝/分布式训练 | [GitHub](https://github.com/liguodongiot/llm-action)
| 大模型白盒子构建指南（tiny-universe） | 手搓 Tiny-LLM Universe，理解 LLM 原理 | [GitHub](https://github.com/datawhalechina/tiny-universe)
| AI 新知：2026 LLM 系列教程 | 面向 2026 年的 AI、大模型、智能体、LLMOps 系列 | [知乎专栏](https://zhuanlan.zhihu.com/p/1993642974563812198)
| InfiniTensor 2025 夏季大模型训练营 | 免费线上课程，涵盖 AI 编译器、推理系统、并行计算 | [训练营主页](https://opencamp.ai/InfiniTensor/camp/2025summer)

---

## 书籍与教材

| 名称 | 作者 / 出版方 | 链接 | 说明 |
| ---- | ---- | ---- | ---- |
| 《大语言模型》 | 赵鑫、李军毅、周昆等（人大高瓴） | [在线阅读](https://llmbook-zh.github.io/) / [PDF](https://llmbook-zh.github.io/LLMBook.pdf) | 第一本中文 LLM 系统教材，391 页，开源免费 |
| 《从零构建大模型》中文版 | Sebastian Raschka | [GitHub](https://github.com/MLNLP-World/LLMs-from-scratch-CN) | 全球爆款 Build a LLM From Scratch 中文版，含 DeepSeek 深度解析 |
| 《大语言模型：基础与前沿》 | 多位作者 | [知乎](https://zhuanlan.zhihu.com/p/18687323985) | 系统讲解 LLM 技术原理与应用 |
| 《大模型原理与架构》在线书 | yeasy | [在线阅读](https://yeasy.gitbook.io/llm_internals/) | 涵盖对齐技术（RLHF/DPO）、推理优化等，持续更新 |

---

## 评测与基准

| 名称 | 开发者 | 链接 | 说明 |
| ---- | ---- | ---- | ---- |
| OpenCompass | 上海 AI 实验室 | [opencompass.org.cn](https://opencompass.org.cn/) / [GitHub](https://github.com/open-compass/opencompass) ⭐6.6k | 通用大模型评测平台，支持 100+ 数据集 |
| C-Eval | 清华 / 上交 / 爱丁堡 | [cevalbenchmark.com](https://cevalbenchmark.com/) | 中文多选题评测基准，52 个学科 13,948 题 |
| SuperCLUE | 独立第三方 | [superclueai.com](https://www.superclueai.com/) | 中文通用大模型综合评测，含匿名对战 |
| FlagEval（天秤） | 智源研究院 (BAAI) | [flageval.baai.ac.cn](https://flageval.baai.ac.cn) | 30+ 能力、5 种任务、600+ 维度全面评测 |
| EvalScope | 阿里 ModelScope | [文档](https://evalscope.readthedocs.io/zh-cn/latest/) | 开源评测框架，内置 20+ 主流 benchmark |
| SafetyBench | 清华 COAI | [GitHub](https://github.com/thu-coai/SafetyBench) | 11,435 题中英双语 LLM 安全评测 |

---

## LLM 安全与对齐

### 安全研究

- [LLM Safety 最新论文推介系列（知乎）](https://zhuanlan.zhihu.com/p/1995509051832943275)：持续更新的中文 LLM 安全论文精读
- [SafetyBench](https://github.com/thu-coai/SafetyBench)：清华 COAI 出品，11,435 题覆盖 7 类安全问题
- [M3-SafetyBench](http://scis.scichina.com/cn/2025/SSI-2025-0254.pdf)：17 万+ 条目中文大模型内容安全评估数据集
- [AIRTBench](https://www.secrss.com/articles/80190)：衡量大语言模型的自主 AI 红队能力

### 对齐技术

- [多模态大模型 RLHF/DPO 全面解读（知乎）](https://zhuanlan.zhihu.com/p/20311517814)：SFT/RLHF/RLAIF/DPO 在多模态场景的应用
- [大模型对齐技术综述（CSDN）](https://blog.csdn.net/pythonhy/article/details/141192664)：一文看尽 RLHF/RLAIF/PPO/DPO 全家桶

### 越狱攻防

- [Weak-to-Strong Jailbreaking（ICML 2025）](https://blog.csdn.net/qq_33583069/article/details/156804484)：利用小模型引导大模型生成有害内容，ASR 超 99%
- [ALERT 零样本越狱检测框架](https://zhuanlan.zhihu.com/p/1992965841180959576)：从层级/模块级/token 级三粒度实现早期无侵入检测
- [自动化红队测试基准](https://adg.csdn.net/6952477f5b9f5f31781b56bd.html)：Garak/AdvGLUE 等主流 LLM 安全评估框架解析

### 2026 前沿安全研究

- [安全向量提取框架 GOSV（2026.2）](https://zhuanlan.zhihu.com/p/2001287271400362256)：全局优化识别安全关键注意力头，约 30% 头被重贴时安全性完全崩溃
- [跨模态越狱迁移（2026.3）](https://zhuanlan.zhihu.com/p/2013670172221792368)：发现"对齐诅咒"——强对齐会将文本漏洞传播到音频模态
- [GT-HARMBENCH 多智能体安全基准（2026.3）](https://zhuanlan.zhihu.com/p/2013709848571815059)：2,009 个高风险场景，15 个前沿模型仅 62% 选择有益行动
- [CC-Delta 稀疏自编码器防御（2026.3）](https://zhuanlan.zhihu.com/p/2013709848571815059)：比较有无越狱上下文下的 token 表示差异，识别越狱特征

---

## 综合资源索引

- [Awesome-Chinese-LLM](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)：⭐22.6k，最全面的中文 LLM 资源索引，涵盖模型、数据集、微调框架、应用等
- [awesome-chatgpt-zh](https://github.com/yzfly/awesome-chatgpt-zh)：⭐11.5k，ChatGPT 中文指南，含工具、应用、开发资源
- [大模型技术实战（llm-action）](https://github.com/liguodongiot/llm-action)：⭐18.1k，大模型工程化实战，涵盖训练/微调/量化/推理/部署全链路
- [LLM 全栈优质资源汇总（llm-resource）](https://github.com/liguodongiot/llm-resource)：大模型全栈学习资料索引
- [Awesome-LLM-Resources](https://github.com/WangRongsheng/awesome-LLM-resources)：全球 LLM 资源总结，含多模态/Agent/编程/数据处理/MCP 等
- [中国大模型列表（awesome-LLMs-In-China）](https://github.com/wgwang/awesome-LLMs-In-China)：国产大模型全景图
- [通往 AGI 之路](https://wiki.waytoagi.com/)：中文 AI 知识社区，含 LLM 开源模型及数据集集合

### 中文 AI 资讯平台

| 名称 | 链接 | 说明 |
| ---- | ---- | ---- |
| 机器之心 | [jiqizhixin.com](https://www.jiqizhixin.com/) | 国内首家系统关注 AI 的科技媒体，论文解读、产业分析 |
| 量子位 | [qbitai.com](https://www.qbitai.com/) | 专注 AI 及前沿科技，技术研究趋势和企业动态 |
| 新智元 | 微信公众号搜索"新智元" | AI 研究进展、行业动态和技术趋势，日更 |
| 阿里云开发者社区 | [developer.aliyun.com](https://developer.aliyun.com/) | ModelScope 生态、推理优化等深度技术内容 |
| 腾讯云开发者社区 | [cloud.tencent.com/developer](https://cloud.tencent.com/developer/) | AI 大模型技术文章、实测对比 |
