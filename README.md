# 中文版本：
# RAG + FAISS 增强的情感支持 Agent

#### 本情感恢复AI助手基于RAG（检索增强生成）架构，通过FAISS向量检索为LLM提供心理学知识支撑，有效消除模型在情感支持场景中的“幻觉”问题。具体而言，系统使用 `paraphrase-multilingual-MiniLM-L12-v2` 多语言嵌入模型将本地数据集（JSON/CSV/TXT/PDF等）中的每条知识转化为多维向量，并采用FAISS的 `IndexFlatL2`（精确欧氏距离索引）构建离线向量库。当用户输入查询时，系统将其编码为相同维度的向量，通过FAISS进行相似度检索，召回Top-K相关心理学知识片段，与用户问题拼接后送入LLM，从而引导模型生成基于事实、专业且符合情境的回复。同时，系统支持基于内容哈希和来源URL的增量去重机制，确保知识库在不断扩充时不会产生冗余。

## 整体页面展示（中文模式）
![整体页面](./images/1.png)

## 功能特点

- **四个专业 AI Agent**
  - 情感支持 Agent：识别情绪，提供共情回应
  - 认知重构 Agent：帮助识别和挑战负面思维模式
  - 行为支持 Agent：提供个性化的应对策略和行动计划
  - 激励 Agent：增强自我效能感，保持积极进展

- **多语言支持**
  - 中文 / English 界面切换
  - Agent 回复自动匹配所选语言

- **RAG 知识库**
  - FAISS 向量索引，高效检索
  - 支持多种数据格式（JSON、CSV、TXT、PDF）
  - 自动去重，增量更新

- **历史记录**
  - 自动保存对话历史
  - 支持导出 JSON / CSV 格式

## 技术栈

| 组件 | 技术 |
|------|------|
| GUI 框架 | PyQt5 |
| 大语言模型 | Gemini / OpenAI / Claude / DeepSeek |
| 向量索引 | FAISS |
| 嵌入模型 | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |

## 安装依赖
pip install -r requirements.txt 


## 使用指南
## 1. 第一步：构建知识库（首次运行）
```bash
python scripts/build_knowledge_base.py
```
### 运行完后选择操作:
- [1]爬虫采集-从网络爬取心理学知识并更新到数据库
![爬虫](./images/3.png)
- [2]本地数据集 - 加载 data/datasets/目录下的文件并更新到数据库

### 其中[2]有两种方式：
- 方式一、使用内置知识库快速启动：
将下载好的文档保存到./data/datasets下，直接选择 2，系统自动保存到知识库。 
![直接加载本地](./images/4.png)

- 方式二、下载真实情感数据集：
```bash
python scripts/download_datasets.py
```
系统会将下载好的文档自动保存到./data/datasets下，然后选择 2，保存到知识库。
![下载数据集到本地](./images/5.png)

### 查看知识库统计：
```bash
python scripts/show_knowledge_stats.py
```
![查看知识库](./images/6.png)

## 2. 第二步：启动程序
```bash
python main.py
```

## 3. 第三步：配置模型
- 在右侧面板选择语言（中文 / English）
- 选择 AI 模型（Gemini / OpenAI / Claude / DeepSeek）
- 输入对应的 API Key
- 点击「初始化 Agent」


## 4. 第四步：开始对话

在输入框分享你的感受（文字/图片），AI 助手会从四个维度给出回应。


## 5. 第五步：历史记录
### 查看历史记录：
![查看历史记录](./images/7.png)

### 导出历史记录：
![导出历史记录](./images/8.png)

### 导出结果展示：
![结果展示](./images/9.png)




# English Version:
# RAG + FAISS Enhanced Emotional Support Agent

#### This Emotional Recovery AI Assistant is built on the RAG (Retrieval-Augmented Generation) architecture, using FAISS vector retrieval to provide psychological knowledge support for LLMs, effectively eliminating the "hallucination" problem in emotional support scenarios. Specifically, the system uses the `paraphrase-multilingual-MiniLM-L12-v2` multilingual embedding model to convert each piece of knowledge from local datasets (JSON/CSV/TXT/PDF, etc.) into multi-dimensional vectors, and employs FAISS's `IndexFlatL2` (exact Euclidean distance index) to build an offline vector database. When a user inputs a query, the system encodes it into a vector of the same dimension, performs similarity search through FAISS, retrieves Top-K relevant psychology knowledge fragments, and concatenates them with the user's question before sending to the LLM, guiding the model to generate factual, professional, and context-appropriate responses. Additionally, the system supports incremental deduplication based on content hashing and source URLs, ensuring no redundancy as the knowledge base expands.

## Overall Interface (English Mode)
![Overall Interface](./images/2.png)

## Features

- **Four Professional AI Agents**
  - Emotional Support Agent: Recognizes emotions and provides empathetic responses
  - Cognitive Restructuring Agent: Helps identify and challenge negative thought patterns
  - Behavioral Support Agent: Provides personalized coping strategies and action plans
  - Motivational Agent: Enhances self-efficacy and maintains positive progress

- **Multi-language Support**
  - Chinese / English interface switching
  - Agent responses automatically match the selected language

- **RAG Knowledge Base**
  - FAISS vector index for efficient retrieval
  - Supports multiple data formats (JSON, CSV, TXT, PDF)
  - Automatic deduplication and incremental updates

- **History Records**
  - Automatically saves conversation history
  - Supports export in JSON / CSV format

## Tech Stack

| Component | Technology |
|-----------|------------|
| GUI Framework | PyQt5 |
| LLM | Gemini / OpenAI / Claude / DeepSeek |
| Vector Index | FAISS |
| Embedding Model | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |

## Install Dependencies
```bash
pip install -r requirements.txt 
```

## User Guide
## Step 1: Build Knowledge Base (First Run)
```bash
python scripts/build_knowledge_base.py
```
### After running, you will see the following options:
- [1]Web Crawler - Crawl psychology knowledge from the internet
![crawl](./images/3.png)
- [2]Local Dataset - Load files from data/datasets/ directory

### Option [2] has two methods:
- Method 1: Quick Start with Built-in Knowledge Base
Save your document files to ./data/datasets/，select option 2，The system will automatically load and save to knowledge base. 
![loader](./images/4.png)

- Method 2: Download Real Emotion Datasets
```bash
python scripts/download_datasets.py
```
The system automatically saves downloaded files to ./data/datasets/，Select option 2 to save to knowledge base.
![download](./images/5.png)

### View Knowledge Base Statistics:
```bash
python scripts/show_knowledge_stats.py
```
![view](./images/6.png)

## Step 2: Launch the Program
```bash
python main.py
```

## Step 3: Configure the Model
- Select language (Chinese / English) in the right panel
- Select AI model (Gemini / OpenAI / Claude / DeepSeek)
- Enter your API Key
- Click 「Initialize Agent」


## Step 4: Start the Conversation

Type your feelings or upload images in the input box. The AI assistant will respond from four dimensions.


## Step 5: History Records
### View History:
![view](./images/10.png)

### Export historical records:
![export](./images/8.png)

### Exported results presentation:
![result](./images/9.png)
