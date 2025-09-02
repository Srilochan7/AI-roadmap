# 🚀This cheatsheet is all you need 
This cheatsheet serves as a practical roadmap and resource guide for anyone looking to get into GenAI or Agentic AI.

> *I'm actively exploring more resources and refining this roadmap to make it more detailed and genuinely helpful — so ⭐ it if you find it valuable!*

---

## 📋 Table of Contents

- [Math Foundations](#0-math-foundations)
- [Python Basics](#1-python-basics)
- [Streamlit](#2-streamlit)
- [Machine Learning — Core Basics](#3-machine-learning--core-basics)
- [Machine Learning — Deep Dive](#4-machine-learning--deep-dive)
- [ML for NLP](#5-ml-for-nlp)
- [Deep Learning Basics](#6-dl-basics)
- [Core Deep Learning](#7-core-dl)
- [DL Frameworks](#8-dl-frameworks)
- [Transformers](#9-transformers)
- [Introduction to Gen AI](#10-introduction-to-gen-ai)
- [Large Language Models (LLMs) - Advanced](#11-large-language-models-llms---advanced)
- [Introduction to LangChain](#12-introduction-to-langchain)
- [RAG (Retrieval Augmented Generation)](#13-rag-retrieval-augmented-generation)
- [Vector Databases](#14-vector-databases)
- [Agentic AI](#15-agentic-ai)
- [LangGraph & Advanced Agents](#16-langgraph--advanced-agents)
- [FastAPI (Backend for AI)](#17-fastapi-backend-for-ai)
- [Resources](#-resources)

---

## **0. Math Foundations**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 0 | **Math for ML/DL** | Linear Algebra, Probability, Statistics, Calculus | [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · [CampusX](https://youtube.com/playlist?list=PLKnIA16_RmvbYFaaeLY28cWeqV-3vADST) |

---

## **1. Python Basics**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 1 | **Python Fundamentals** | Basics, data structures, file handling, exception handling, OOP | [FreeCodeCamp](https://youtu.be/eWRfhZUzrAc) |

---

## **2. Streamlit**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 2 | **Streamlit Basics** | UI building, web apps for ML | [Chai aur Code](https://youtu.be/yKTEC1Y5bEQ) |

---

## **3. Machine Learning — Core Basics**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 3 | **ML Fundamentals** | Classification, Regression, Pipelines, Feature Engineering | [CampusX](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) · [Stanford CS229](https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) |
| 4 | **ML Evaluation** | Accuracy, Precision, Recall, Confusion Matrix, ROC-AUC | [StatQuest](https://www.youtube.com/@statquest) |
| 5 | **Feature Scaling** | Normalization, Standardization, MinMax, Robust Scaling | [Scikit-learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html) |
| 6 | **Data Labeling** | Manual annotation, Label Studio, Roboflow | [Label Studio](https://labelstud.io/) · [Roboflow](https://roboflow.com/) |

### 🛠 **P1: Core ML Projects**

| Project | Description | Datasets | Tech Stack |
|---------|-------------|----------|------------|
| **ML Classification App** | Build a classification app using sklearn + Streamlit | Iris, Titanic, MNIST | sklearn, Streamlit, pandas |
| **Regression Price Predictor** | Housing price prediction with feature engineering | Boston Housing, California Housing | scikit-learn, seaborn, matplotlib |

---

## **4. Machine Learning — Deep Dive**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 7 | **Unsupervised ML** | Clustering (K-Means, DBSCAN, Hierarchical), Dimensionality Reduction (PCA, t-SNE, UMAP) | [StatQuest](https://www.youtube.com/@statquest) |
| 8 | **Ensemble Methods** | Bagging, Boosting (XGBoost, LightGBM), Stacking | [Krish Naik](https://www.youtube.com/@krishnaik06) |
| 9 | **Hyperparameter Tuning** | GridSearchCV, RandomSearch, Optuna, Bayesian Optimization | [Optuna Docs](https://optuna.org/) |
| 10 | **Core ML Concepts** | Bias-variance tradeoff, Underfitting/Overfitting, Regularization (L1/L2) | [Andrew Ng ML](https://www.coursera.org/learn/machine-learning) |

---

## **5. ML for NLP**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 11 | **Traditional NLP** | Text preprocessing, One-Hot Encoding, Bag of Words, TF-IDF, Word2Vec | [Krish Naik](https://youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn) |

### 🛠 **P2: NLP Projects**

| Project | Description | Datasets | Tech Stack |
|---------|-------------|----------|------------|
| **Text Classifier** | Spam detection or sentiment analysis using BoW/TF-IDF | SMS Spam, IMDb Reviews | sklearn, NLTK, pandas |
| **Word2Vec Explorer** | Visualize similarity between words using Word2Vec | Google News Word2Vec | Gensim, matplotlib, seaborn |

---

## **6. DL Basics**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 12 | **Deep Learning Fundamentals** | Neural Networks, Loss Functions, Optimizers, Activation Functions | [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) · [MIT 6.S191](http://introtodeeplearning.com/) |

---

## **7. Core DL**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 13 | **Neural Networks & ANN** | Feedforward networks, backpropagation, gradient descent | [MIT 6.S191](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [3Blue1Brown](https://www.youtube.com/@3blue1brown) |
| 14 | **CNN** | Convolutional Neural Networks for computer vision | [MIT 6.S191](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [CS231n](https://youtu.be/iOdFUJiB0Zc) |
| 15 | **RNN & LSTM** | Sequential data modeling, time series | [MIT 6.S191](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |

---

## **8. DL Frameworks**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 16 | **PyTorch/TensorFlow** | Tensors, model building, training loops | [PyTorch Docs](https://pytorch.org) · [TensorFlow Docs](https://www.tensorflow.org) · [PyTorch Tutorial](https://youtu.be/Z_ikDlimN6A) |

### 🛠 **P3: Deep Learning Projects**

| Project | Description | Datasets | Tech Stack |
|---------|-------------|----------|------------|
| **Image Classifier** | Build CNN to classify cats vs dogs | Dogs vs Cats (Kaggle) | TensorFlow/Keras, PyTorch |
| **Sentiment with LSTM** | Sentiment prediction using LSTM networks | IMDb, Twitter Sentiment | Keras, PyTorch, torchtext |

---

## **9. Transformers**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 17 | **Transformer Architecture** | Self-attention, Multi-head attention, Positional Encoding, Encoder-Decoder | [3Blue1Brown](https://youtu.be/wjZofJX0v4M) · [Campus X](https://youtube.com/playlist?list=PLkBMe2eZMRQ2VKEtoL0GVUrNzEiXfgj07) |
| 18 | **Tokenization** | BPE, SentencePiece, GPT-2 tokenizer, Hugging Face tokenizers | [Andrej Karpathy](https://youtu.be/ZhAz268Hdpw) · [Original Paper](https://arxiv.org/pdf/1706.03762) |

---

## **10. Introduction to Gen AI**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 19 | **GenAI Fundamentals** | AI vs ML vs DL vs GenAI, How GPT/LLMs are trained, LLM evolution | [Fireship](https://youtu.be/X7Zd4VyUgL0) · [Two Minute Papers](https://youtu.be/d4yCWBGFCEs) |
| 20 | **LLM Evaluation** | BLEU, ROUGE, Perplexity, Human Evaluation, Benchmarks | [Hugging Face Evaluation](https://huggingface.co/docs/evaluate/index) |
| 21 | **Ethics & AI Safety** | Hallucination, bias, responsible deployment, alignment | [AI Safety Course](https://course.aisafetyfundamentals.com/) |

---



## **11. Introduction to LangChain**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 27 | **LangChain Fundamentals** | Components, Chains, Agents, Memory | [LangChain Docs](https://python.langchain.com/docs/introduction/) · [LangChain Tutorial](https://youtu.be/X0btK9X0Xnk) |
| 28 | **LLM Integration** | OpenAI, Ollama, Hugging Face, Groq integration | [Ollama Setup](https://ollama.ai/) · [Groq API](https://groq.com/) |
| 29 | **Prompt Engineering** | Zero-shot, few-shot, chain-of-thought, prompt optimization | [OpenAI Cookbook](https://github.com/openai/openai-cookbook) · [Prompt Engineering Guide](https://www.promptingguide.ai/) |

### 🛠 **P4: LangChain Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **Chatbot with LangChain** | Build intelligent chatbot using LangChain + LLM + Streamlit | LangChain, Streamlit, Ollama/OpenAI |
| **Document Summarizer** | Summarize PDF/Text documents with LLMs | LangChain, PyPDF, Hugging Face Transformers |

---

## **12. RAG (Retrieval Augmented Generation)**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 30 | **RAG Fundamentals** | Retrieval pipeline, embedding models, vector similarity | [RAG Tutorial](https://youtu.be/X0btK9X0Xnk) · [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) |
| 31 | **Advanced RAG** | Multi-query retrieval, re-ranking, hybrid search | [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) |

### 🛠 **P5: RAG Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **PDF Q&A with RAG** | Upload PDF → extract → chunk → embed → query via LLM | LangChain, FAISS, OpenAI/Groq, Streamlit |
| **Multi-Document RAG** | Query across multiple documents with source attribution | ChromaDB, LangChain, sentence-transformers |

---

## **13. Vector Databases**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 32 | **Vector DB Fundamentals** | FAISS, ChromaDB, Pinecone, Weaviate, similarity search | [Pinecone Docs](https://docs.pinecone.io/) · [ChromaDB](https://docs.trychroma.com/) |
| 33 | **Embedding Models** | sentence-transformers, OpenAI embeddings, custom embeddings | [Sentence Transformers](https://www.sbert.net/) |

---
## **14. Large Language Models (LLMs) - Advanced**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 22 | **PEFT (Parameter Efficient Fine-Tuning)** | LoRA, QLoRA, AdaLoRA, Prefix Tuning, P-Tuning | [Hugging Face PEFT](https://huggingface.co/docs/peft/index) · [LoRA Paper](https://arxiv.org/abs/2106.09685) |
| 23 | **LoRA & QLoRA** | Low-Rank Adaptation, Quantized LoRA for efficient fine-tuning | [QLoRA Paper](https://arxiv.org/abs/2305.14314) · [Practical LoRA](https://youtu.be/PXWYUTMt-AU) |
| 24 | **Quantization Techniques** | INT8, INT4, GPTQ, AWQ, GGML/GGUF formats | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) · [GPTQ](https://arxiv.org/abs/2210.17323) |
| 25 | **Model Compression** | Pruning, Distillation, Quantization-Aware Training | [Neural Compression](https://youtu.be/DQQsNBzp-oI) |
| 26 | **Advanced Fine-tuning** | Full fine-tuning vs PEFT, Instruction tuning, RLHF basics | [Hugging Face Fine-tuning](https://huggingface.co/docs/transformers/training) |

---

## **15. Agentic AI**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 34 | **AI Agent Fundamentals** | Agent architecture, planning, tool use, memory systems | [Lilian Weng's Blog](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| 35 | **Tool-Using Agents** | Function calling, external APIs, code execution | [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) |
| 36 | **Multi-Agent Systems** | Agent collaboration, communication protocols | [AutoGen](https://github.com/microsoft/autogen) · [CrewAI](https://github.com/joaomdmoura/crewAI) |
| 37 | **ReAct & Planning** | Reasoning + Acting, chain-of-thought for agents | [ReAct Paper](https://arxiv.org/abs/2210.03629) |

### 🛠 **P6: Agentic AI Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **Research Assistant Agent** | AI agent that can search web, summarize, and synthesize information | LangChain, Tavily/SerpAPI, OpenAI |
| **Code Review Agent** | Agent that reviews code, suggests improvements, runs tests | GitHub API, LangChain, code execution tools |

---

## **16. LangGraph & Advanced Agents**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 38 | **LangGraph Fundamentals** | State machines, graph-based workflows for agents | [LangGraph Docs](https://langchain-ai.github.io/langgraph/) · [LangGraph Tutorial](https://youtu.be/VaAlSpe2B30) |
| 39 | **Complex Agent Workflows** | Multi-step reasoning, conditional flows, human-in-the-loop | [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples) |
| 40 | **Agent Orchestration** | Managing multiple agents, workflow optimization | [LangGraph Advanced](https://youtu.be/9lWe6K5OMuY) |

### 🛠 **P7: LangGraph Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **Multi-Step Research Agent** | Agent that plans research, gathers info, and creates reports | LangGraph, multiple LLMs, web search APIs |
| **Customer Service Agent** | Complex customer service with escalation and human handoff | LangGraph, FastAPI, database integration |

---

## **17. Model Context Protocol (MCP)**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 41 | **MCP Fundamentals** | Protocol for connecting AI assistants to external data sources and tools | [Anthropic MCP Docs](https://modelcontextprotocol.io/) · [MCP GitHub](https://github.com/modelcontextprotocol) |
| 42 | **MCP Implementation** | Building MCP servers, client integration, tool development | [MCP Quickstart](https://modelcontextprotocol.io/quickstart) |

---

## **18. FastAPI (Backend for AI)**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 43 | **FastAPI Fundamentals** | REST APIs, async programming, model deployment | [FastAPI Docs](https://fastapi.tiangolo.com/) · [FastAPI Course](https://youtu.be/0sOvCWFmrtA) |
| 44 | **AI Model Deployment** | Serving ML/DL models, batch processing, monitoring | [MLOps Best Practices](https://ml-ops.org/) |

---

## 📚 **Resources**

### 🎥 **YouTube Channels**
- [Krish Naik](https://www.youtube.com/@krishnaik06) - Comprehensive ML/AI tutorials
- [CampusX](https://www.youtube.com/@campusx-official) - Indian ML education
- [IBM Technology](https://www.youtube.com/@IBMTechnology) - Enterprise AI concepts
- [Codebasics](https://www.youtube.com/@codebasics) - Programming + ML
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy) - Deep learning from scratch
- [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) - Latest AI research

### 📖 **Essential Books**
- **Hands-On Machine Learning** by Aurélien Géron
- **Hands-On Large Language Models** by Jay Alammar & Maarten Grootendorst
- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Pattern Recognition and Machine Learning** by Christopher Bishop

### 📄 **Key Research Papers**
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

### 🗃️ **Datasets**
- [Kaggle](https://www.kaggle.com) - Competition datasets
- [Hugging Face Datasets](https://huggingface.co/datasets) - NLP and multimodal datasets
- [Papers With Code](https://paperswithcode.com/datasets) - Research datasets
- [Google Dataset Search](https://datasetsearch.research.google.com/) - Dataset discovery

### 🛠️ **Tools & Platforms**
- **Development**: Jupyter, VS Code, Google Colab
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM
- **DL Frameworks**: PyTorch, TensorFlow, JAX
- **LLM Tools**: Hugging Face Transformers, LangChain, LlamaIndex
- **Vector DBs**: Pinecone, ChromaDB, Weaviate, FAISS
- **Deployment**: FastAPI, Streamlit, Gradio, Docker

---

## 🚀 **Getting Started**

1. **Prerequisites**: Ensure you have Python 3.8+ and basic programming knowledge
2. **Environment Setup**: Use conda/venv for package management
3. **Start Small**: Begin with traditional ML before moving to deep learning
4. **Build Projects**: Apply concepts through hands-on projects
5. **Stay Updated**: Follow latest research and industry trends

---

## 🤝 **Contributing**

Feel free to contribute to this roadmap by:
- Adding new resources and tutorials
- Suggesting improvements to the learning path
- Sharing your project experiences
- Reporting broken links or outdated content

---

## ⭐ **Star this repository if you find it helpful!**

*This roadmap is continuously updated with the latest developments in Generative AI and Machine Learning.*
