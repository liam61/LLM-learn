# Agent

## 昂贵的 Pre-Training

- 显卡成本
- 数据成本
- 存储成本
- 数据中心成本
- 人力成本

| 策略                | 难度 | 数据要求 |
|--------------------|------|---------|
| Prompt Engineering | 低   | 无       |
| Self-Reflection    | 低   | 无       |
| RAG                | 中   | 少量     |
| Agent              | 中   | 少量     |
| Fine-tuning        | 高   | 中等     |

## Prompt Engineering

Prompt Engineering 是优化 prompts 以获得有效输出的艺术和科学

它涉及设计、编写和修改 prompts，以引导 AI 模型生成高质量、相关且有用的响应

## Self-Reflection

## RAG

检索增强生成（Retrieval-Augmented Generation，简称 RAG）

通过结合大型语言模型（LLM）和信息检索系统来提高生成文本的准确性和相关性

优点：允许模型在生成回答之前，先从权威知识库中检索相关信息，从而确保输出内容的时效性和专业性，无需对模型本身进行重新训练

解决问题：例如虚假信息的提供、过时信息的生成、非权威来源的依赖以及由于术语混淆导致的不准确响应

个人看法：助力更好的回答问题，更实时、更专业

![alt text](https://luhengshiwo.github.io/LLMForEverybody/07-%E7%AC%AC%E4%B8%83%E7%AB%A0-Agent/assest/%E5%BC%80%E5%8F%91%E5%A4%A7%E6%A8%A1%E5%9E%8Bor%E4%BD%BF%E7%94%A8%E5%A4%A7%E6%A8%A1%E5%9E%8B/8.PNG)

## Agent

Agent 指的是一个能够感知其环境并根据感知到的信息做出决策以实现特定目标的系统，通过大模型的加持，Agent比以往任何时候都要更加引人注目

以 Langchain 为代表的 Agent 框架

an agent is something that can act or make decisions on its own. It’s like a person or a robot that can take actions based on the information it receives

In more technical terms, an agent here refers to a running large language model (LLM) model with its respective prompt. Each agent operates based on a prompt, which serves as its initial input or instruction. The prompt helps define the context, goals, and constraints for the agent's responses. It acts like a starting point or directive that guides the agent’s behavior and decision-making process.

个人看法：AI助手，帮助实施一系列动作，有自己的思考和行动

## Fine-tuning

相较于基础大模型动辄万卡的代价，微调可能是普通个人或者企业少数能够接受的后训练大模型(post-training)的方式

微调是指在一个预训练模型(pre-training)的基础上，通过少量的数据和计算资源，对模型进行进一步训练，以适应特定的任务或者数据集

# Agent 设计范式与常见框架

## Agent 设计范式

Agent的本质还是 prompt engineering

智能代理是指能够在环境中感知、推理并采取行动以完成特定任务的系统

## 常见范式

### 1. Reflection

Reflection 是指 Agent 能够对自己的行为和决策进行推理和分析的能力

![alt text](https://luhengshiwo.github.io/LLMForEverybody/07-%E7%AC%AC%E4%B8%83%E7%AB%A0-Agent/assest/Agent%E8%AE%BE%E8%AE%A1%E8%8C%83%E5%BC%8F%E4%B8%8E%E5%B8%B8%E8%A7%81%E6%A1%86%E6%9E%B6/1.PNG)

### 2. Tool use

Agent 也可以使用工具来帮助完成任务。这种 Agent 范式涉及到 Agent 如何利用外部工具和资源来提升自己的决策和执行能力

![alt text](https://luhengshiwo.github.io/LLMForEverybody/07-%E7%AC%AC%E4%B8%83%E7%AB%A0-Agent/assest/Agent%E8%AE%BE%E8%AE%A1%E8%8C%83%E5%BC%8F%E4%B8%8E%E5%B8%B8%E8%A7%81%E6%A1%86%E6%9E%B6/2.PNG)


```python
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(temperature=0, model=llm_model)
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
```

### 3. Planning

规划是 Agent AI 的一个关键设计模式，我们使用大型语言模型自主决定执行哪些步骤来完成更大的任务

### 4. Multi-agent collaboration

多智能体协作是四种关键人工智能智能体设计模式中的最后一种。对于编写软件这样的复杂任务，多智能体方法会将任务分解为由不同角色

![alt text](https://luhengshiwo.github.io/LLMForEverybody/07-%E7%AC%AC%E4%B8%83%E7%AB%A0-Agent/assest/Agent%E8%AE%BE%E8%AE%A1%E8%8C%83%E5%BC%8F%E4%B8%8E%E5%B8%B8%E8%A7%81%E6%A1%86%E6%9E%B6/4.PNG)

## Agent 框架

1. Langgraph

以Langchain为代表的Agent框架，是目前在国内最被广泛使用的开源框架

[text](https://img.36krcdn.com/hsossms/20250108/v2_236ed379dc174d729f16f4b6de8b77bd%40000000_oswg382766oswg600oswg728_img_000?x-oss-process%3Dimage%2Fformat%2Cjpg%2Finterlace%2C1)

https://36kr.com/p/3113897985658368


![alt text](https://luhengshiwo.github.io/LLMForEverybody/07-%E7%AC%AC%E4%B8%83%E7%AB%A0-Agent/assest/langchain%E5%90%91%E5%B7%A6coze%E5%90%91%E5%8F%B3/2.webp)

2. 扣子

扣子是新一代大模型 AI 应用开发平台。无论你是否有编程基础，都可以快速搭建出各种 Bot，并一键发布到各大社交平台

![alt text](https://luhengshiwo.github.io/LLMForEverybody/07-%E7%AC%AC%E4%B8%83%E7%AB%A0-Agent/assest/langchain%E5%90%91%E5%B7%A6coze%E5%90%91%E5%8F%B3/3.webp)

# AI Agents: Autonomy and Adaptability

- https://www.digitalocean.com/community/conceptual-articles/rag-ai-agents-agentic-rag-comparative-analysis

![alt text](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/2024/Shaoni/Adrien/Image_4.png)

## How Model-Based Reflex Agents Work

- https://www.digitalocean.com/community/conceptual-articles/rag-ai-agents-agentic-rag-comparative-analysis#how-model-based-reflex-agents-work

- https://dev.to/tal7aouy/rag-vs-agents-a-comparison-and-when-to-use-each-gn

For example, a robot vacuum cleaner represents a model-based reflex agent. It uses sensors to identify its position and detect obstacles while keeping an internal room map. This map helps the vacuum recall areas it has already cleaned and navigate obstacles more effectively. This way, the agent prevents unnecessary actions and enhances performance compared to a simple reflex system.

example2:

Let's say you have a personal assistant agent. You ask it to "Schedule a meeting with Sarah for tomorrow at 3 PM." The agent would:
Check your calendar.
Verify Sarah's availability.
Schedule the meeting.
Confirm it with you and Sarah.

![alt text](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/2024/Shaoni/Adrien/Image_5.png)

1. Input
2. Processing: They analyze the input and determine a course of action, often involving multiple steps, context management, or interactions.
3. Output

## Agent types

1. Standard Agent

An Agent can bind a knowledge base you provided

2. RAG Agent

It will find related document based on the human message
