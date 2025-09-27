# 🚀This repo is all you need for AIML 
This cheatsheet serves as a practical roadmap and resource guide for anyone looking to get into GenAI or Agentic AI.

> *I'm actively exploring more resources and refining this roadmap to make it more detailed and genuinely helpful — so ⭐ it if you find it valuable!*

---

## 📋 Table of Contents

- [Math Foundations](#0-math-foundations)
- [Python Basics](#1-python-basics)
- [Streamlit](#2-streamlit)
- [FastAPI](#3-fastapi)
- [Machine Learning — Core Basics](#4-machine-learning--core-basics)
- [Machine Learning — Deep Dive](#5-machine-learning--deep-dive)
- [ML for NLP](#6-ml-for-nlp)
- [Deep Learning Basics](#7-dl-basics)
- [Core Deep Learning](#8-core-dl)
- [DL Frameworks](#9-dl-frameworks)
- [MLOps](#10-mlops)
- [Transformers](#11-transformers)
- [Introduction to Gen AI](#12-introduction-to-gen-ai)
- [Large Language Models (LLMs) - Advanced](#13-large-language-models-llms---advanced)
- [Introduction to LangChain](#14-introduction-to-langchain)
- [RAG (Retrieval Augmented Generation)](#15-rag-retrieval-augmented-generation)
- [Vector Databases](#16-vector-databases)
- [Agentic AI](#17-agentic-ai)
- [LangGraph & Advanced Agents](#18-langgraph--advanced-agents)
- [Model Context Protocol (MCP)](#19-model-context-protocol-mcp)
- [FastAPI (Backend for AI)](#20-fastapi-backend-for-ai)
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

## **3. FastAPI**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 3 | **FastAPI Fundamentals** | REST APIs, async programming, model deployment | [FastAPI Docs](https://fastapi.tiangolo.com/) · [FastAPI Course](https://youtu.be/0sOvCWFmrtA) |

---

## **4. Machine Learning — Core Basics**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 4 | **ML Fundamentals** | Classification, Regression, Pipelines, Feature Engineering | [CampusX](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) · [Stanford CS229](https://youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) |
| 5 | **ML Evaluation** | Accuracy, Precision, Recall, Confusion Matrix, ROC-AUC | [StatQuest](https://www.youtube.com/@statquest) |
| 6 | **Feature Scaling** | Normalization, Standardization, MinMax, Robust Scaling | [Scikit-learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html) |
| 7 | **Data Labeling** | Manual annotation, Label Studio, Roboflow | [Label Studio](https://labelstud.io/) · [Roboflow](https://roboflow.com/) |

### 🛠 **P1: Core ML Projects**

| Project | Description | Datasets | Tech Stack |
|---------|-------------|----------|------------|
| **ML Classification App** | Build a classification app using sklearn + Streamlit | Iris, Titanic, MNIST | sklearn, Streamlit, pandas |
| **Regression Price Predictor** | Housing price prediction with feature engineering | Boston Housing, California Housing | scikit-learn, seaborn, matplotlib |

---

## **5. Machine Learning — Deep Dive**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 8 | **Unsupervised ML** | Clustering (K-Means, DBSCAN, Hierarchical), Dimensionality Reduction (PCA, t-SNE, UMAP) | [StatQuest](https://www.youtube.com/@statquest) |
| 9 | **Ensemble Methods** | Bagging, Boosting (XGBoost, LightGBM), Stacking | [Krish Naik](https://www.youtube.com/@krishnaik06) |
| 10 | **Hyperparameter Tuning** | GridSearchCV, RandomSearch, Optuna, Bayesian Optimization | [Optuna Docs](https://optuna.org/) |
| 11 | **Core ML Concepts** | Bias-variance tradeoff, Underfitting/Overfitting, Regularization (L1/L2) | [Andrew Ng ML](https://www.coursera.org/learn/machine-learning) |

---

## **6. ML for NLP**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 12 | **Traditional NLP** | Text preprocessing, One-Hot Encoding, Bag of Words, TF-IDF, Word2Vec | [Krish Naik](https://youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn) |

### 🛠 **P2: NLP Projects**

| Project | Description | Datasets | Tech Stack |
|---------|-------------|----------|------------|
| **Text Classifier** | Spam detection or sentiment analysis using BoW/TF-IDF | SMS Spam, IMDb Reviews | sklearn, NLTK, pandas |
| **Word2Vec Explorer** | Visualize similarity between words using Word2Vec | Google News Word2Vec | Gensim, matplotlib, seaborn |

---

## **7. DL Basics**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 13 | **Deep Learning Fundamentals** | Neural Networks, Loss Functions, Optimizers, Activation Functions | [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) · [MIT 6.S191](http://introtodeeplearning.com/) [Campus X](https://youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&si=VGTrEEzSt02wLgks)|

---

## **8. Core DL**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 14 | **Neural Networks & ANN** | Feedforward networks, backpropagation, gradient descent | [MIT 6.S191](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [3Blue1Brown](https://www.youtube.com/@3blue1brown) [Campus X](https://youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&si=VGTrEEzSt02wLgks)|
| 15 | **CNN** | Convolutional Neural Networks for computer vision |[Campus X](https://youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&si=VGTrEEzSt02wLgks) [MIT 6.S191](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [CS231n](https://youtu.be/iOdFUJiB0Zc) |
| 16 | **RNN & LSTM** | Sequential data modeling, time series |[Campus X](https://youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&si=VGTrEEzSt02wLgks) [MIT 6.S191](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) · [Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |

---

## **9. DL Frameworks**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 17 | **PyTorch/TensorFlow** | Tensors, model building, training loops | [PyTorch Docs](https://pytorch.org) · [TensorFlow Docs](https://www.tensorflow.org) · [PyTorch Tutorial](https://youtu.be/Z_ikDlimN6A) |

### 🛠 **P3: Deep Learning Projects**

| Project | Description | Datasets | Tech Stack |
|---------|-------------|----------|------------|
| **Image Classifier** | Build CNN to classify cats vs dogs | Dogs vs Cats (Kaggle) | TensorFlow/Keras, PyTorch |
| **Sentiment with LSTM** | Sentiment prediction using LSTM networks | IMDb, Twitter Sentiment | Keras, PyTorch, torchtext |

---

## **10. MLOps**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 18 | **MLOps Fundamentals** | Model versioning, experiment tracking, CI/CD for ML, monitoring | [MLOps Playlist](https://youtube.com/playlist?list=PLupK5DK91flV45dkPXyGViMLtHadRr6sp) · [MLOps Best Practices](https://ml-ops.org/) |
| 19 | **Model Deployment** | Docker, cloud deployment, model serving, A/B testing | [MLOps Playlist](https://youtube.com/playlist?list=PLupK5DK91flV45dkPXyGViMLtHadRr6sp) |
| 20 | **Experiment Tracking** | MLflow, Weights & Biases, model registry | [MLflow Docs](https://mlflow.org/) · [Weights & Biases](https://wandb.ai/) |

---

## **11. Transformers**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 21 | **Transformer Architecture** | Self-attention, Multi-head attention, Positional Encoding, Encoder-Decoder | [3Blue1Brown](https://youtu.be/wjZofJX0v4M) · [Campus X](https://youtube.com/playlist?list=PLkBMe2eZMRQ2VKEtoL0GVUrNzEiXfgj07) |
| 22 | **Tokenization** | BPE, SentencePiece, GPT-2 tokenizer, Hugging Face tokenizers | [Campus X](https://youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn&si=VGTrEEzSt02wLgks) [Andrej Karpathy](https://youtu.be/ZhAz268Hdpw) · [Original Paper](https://arxiv.org/pdf/1706.03762) |

---

## **12. Introduction to Gen AI**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 23 | **GenAI Fundamentals** | AI vs ML vs DL vs GenAI, How GPT/LLMs are trained, LLM evolution | [Fireship](https://youtu.be/X7Zd4VyUgL0) · [Two Minute Papers](https://youtu.be/d4yCWBGFCEs) |
| 24 | **LLM Evaluation** | BLEU, ROUGE, Perplexity, Human Evaluation, Benchmarks | [Hugging Face Evaluation](https://huggingface.co/docs/evaluate/index) |
| 25 | **Ethics & AI Safety** | Hallucination, bias, responsible deployment, alignment | [AI Safety Course](https://course.aisafetyfundamentals.com/) |

---

## **13. Large Language Models (LLMs) - Advanced**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 26 | **PEFT (Parameter Efficient Fine-Tuning)** | LoRA, QLoRA, AdaLoRA, Prefix Tuning, P-Tuning | [Hugging Face PEFT](https://huggingface.co/docs/peft/index) · [LoRA Paper](https://arxiv.org/abs/2106.09685) |
| 27 | **LoRA & QLoRA** | Low-Rank Adaptation, Quantized LoRA for efficient fine-tuning | [QLoRA Paper](https://arxiv.org/abs/2305.14314) · [Practical LoRA](https://youtu.be/PXWYUTMt-AU) |
| 28 | **Quantization Techniques** | INT8, INT4, GPTQ, AWQ, GGML/GGUF formats | [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) · [GPTQ](https://arxiv.org/abs/2210.17323) |
| 29 | **Model Compression** | Pruning, Distillation, Quantization-Aware Training | [Neural Compression](https://youtu.be/DQQsNBzp-oI) |
| 30 | **Advanced Fine-tuning** | Full fine-tuning vs PEFT, Instruction tuning, RLHF basics | [Hugging Face Fine-tuning](https://huggingface.co/docs/transformers/training) |

---

## **14. Introduction to LangChain**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 31 | **LangChain Fundamentals** | Components, Chains, Agents, Memory | [LangChain Docs](https://python.langchain.com/docs/introduction/) · [LangChain Tutorial](https://youtu.be/X0btK9X0Xnk) |
| 32 | **LLM Integration** | OpenAI, Ollama, Hugging Face, Groq integration | [Ollama Setup](https://ollama.ai/) · [Groq API](https://groq.com/) |
| 33 | **Prompt Engineering** | Zero-shot, few-shot, chain-of-thought, prompt optimization | [OpenAI Cookbook](https://github.com/openai/openai-cookbook) · [Prompt Engineering Guide](https://www.promptingguide.ai/) |

### 🛠 **P4: LangChain Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **Chatbot with LangChain** | Build intelligent chatbot using LangChain + LLM + Streamlit | LangChain, Streamlit, Ollama/OpenAI |
| **Document Summarizer** | Summarize PDF/Text documents with LLMs | LangChain, PyPDF, Hugging Face Transformers |

---

## **15. RAG (Retrieval Augmented Generation)**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 34 | **RAG Fundamentals** | Retrieval pipeline, embedding models, vector similarity | [RAG Tutorial](https://youtu.be/X0btK9X0Xnk) · [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) |
| 35 | **Advanced RAG** | Multi-query retrieval, re-ranking, hybrid search | [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) |

### 🛠 **P5: RAG Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **PDF Q&A with RAG** | Upload PDF → extract → chunk → embed → query via LLM | LangChain, FAISS, OpenAI/Groq, Streamlit |
| **Multi-Document RAG** | Query across multiple documents with source attribution | ChromaDB, LangChain, sentence-transformers |

---

## **16. Vector Databases**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 36 | **Vector DB Fundamentals** | FAISS, ChromaDB, Pinecone, Weaviate, similarity search | [Pinecone Docs](https://docs.pinecone.io/) · [ChromaDB](https://docs.trychroma.com/) |
| 37 | **Embedding Models** | sentence-transformers, OpenAI embeddings, custom embeddings | [Sentence Transformers](https://www.sbert.net/) |

---

## **17. Agentic AI**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 38 | **AI Agent Fundamentals** | Agent architecture, planning, tool use, memory systems | [Lilian Weng's Blog](https://lilianweng.github.io/posts/2023-06-23-agent/) |
| 39 | **Tool-Using Agents** | Function calling, external APIs, code execution | [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) |
| 40 | **Multi-Agent Systems** | Agent collaboration, communication protocols | [AutoGen](https://github.com/microsoft/autogen) · [CrewAI](https://github.com/joaomdmoura/crewAI) |
| 41 | **ReAct & Planning** | Reasoning + Acting, chain-of-thought for agents | [ReAct Paper](https://arxiv.org/abs/2210.03629) |

### 🛠 **P6: Agentic AI Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **Research Assistant Agent** | AI agent that can search web, summarize, and synthesize information | LangChain, Tavily/SerpAPI, OpenAI |
| **Code Review Agent** | Agent that reviews code, suggests improvements, runs tests | GitHub API, LangChain, code execution tools |

---

## **18. LangGraph & Advanced Agents**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 42 | **LangGraph Fundamentals** | State machines, graph-based workflows for agents | [LangGraph Docs](https://langchain-ai.github.io/langgraph/) · [LangGraph Tutorial](https://youtu.be/VaAlSpe2B30) |
| 43 | **Complex Agent Workflows** | Multi-step reasoning, conditional flows, human-in-the-loop | [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples) |
| 44 | **Agent Orchestration** | Managing multiple agents, workflow optimization | [LangGraph Advanced](https://youtu.be/9lWe6K5OMuY) |

### 🛠 **P7: LangGraph Projects**

| Project | Description | Tech Stack |
|---------|-------------|------------|
| **Multi-Step Research Agent** | Agent that plans research, gathers info, and creates reports | LangGraph, multiple LLMs, web search APIs |
| **Customer Service Agent** | Complex customer service with escalation and human handoff | LangGraph, FastAPI, database integration |

---

## **19. Model Context Protocol (MCP)**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 45 | **MCP Fundamentals** | Protocol for connecting AI assistants to external data sources and tools | [Campus X](https://youtube.com/playlist?list=PLKnIA16_Rmva_oZ9F4ayUu9qcWgF7Fyc0&si=xcrbYp8oaCFopSln) · [MCP GitHub](https://github.com/modelcontextprotocol) |
| 46 | **MCP Implementation** | Building MCP servers, client integration, tool development | [Krishnaik](https://youtu.be/MDBG2MOp4Go?si=pEbvdUGHK5S0euEH) |

---

## **20. FastAPI (Backend for AI)**

| S.No | Topic | Description | Resources |
|------|-------|-------------|-----------|
| 47 | **AI Model Deployment** | Serving ML/DL models, batch processing, monitoring | [MLOps Best Practices](https://ml-ops.org/) · [MLOps Playlist](https://youtube.com/playlist?list=PLupK5DK91flV45dkPXyGViMLtHadRr6sp) |

---

## 📚 **Resources**

### 🎥 **Popular YouTube Channels**
- [3Blue1Brown]([https://www.youtube.com/@krishnaik06](https://www.youtube.com/@3blue1brown)) - AIML indepth intuition
- [CampusX](https://www.youtube.com/@campusx-official) - Indian ML education
- [Krish Naik](https://www.youtube.com/@krishnaik06) - Comprehensive ML/AI tutorials
- [IBM Technology](https://www.youtube.com/@IBMTechnology) - Fast recap while interviews
- [Codebasics](https://www.youtube.com/@codebasics) - extras
- [FreeCodeCamp](https://www.youtube.com/@freecodecamp) - extras

- [Andrej Karpathy](http://www.youtube.com/channel/UCXUPKJO5MZQN11PqgIvyuvQ)  
- [Jeremy Howard](http://www.youtube.com/user/howardjeremyp)  
- [3Blue1Brown](http://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)  
- [Serrano Academy](http://www.youtube.com/channel/UCgBncpylJ1kiVaPyP-PZauQ)  
- [Lex Fridman](http://www.youtube.com/user/lexfridman)  
- [Hamel Husain](http://www.youtube.com/channel/UC__dUuqF5w4OnbW221JxmKg)  
- [Machine Learning Street Talk](http://www.youtube.com/channel/UCMLtBahI5DMrt0NPvDSoIRQ)  
- [Jason Liu](http://www.youtube.com/channel/UCNhyF7LnO54xA-RW3shkSxA)  
- [StatQuest with Josh Starmer](http://www.youtube.com/user/joshstarmer)  
- [Dave Ebbelaar](http://www.youtube.com/channel/UCn8ujwUInbJkBhffxqAPBVQ) 

### 📖 **Essential Books**
- [All Top AIML Books](https://drive.google.com/drive/folders/1uZJrGQpOCv6K17YHvqk83oeBEbzpDTtG?usp=drive_link) - collection in one place


### 📄 **Key Research Papers**
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)



## 🤝 **Contributing**

Feel free to contribute to this roadmap by:
- Adding new resources and tutorials
- Suggesting improvements to the learning path
- Sharing your project experiences
- Reporting broken links or outdated content

---

## ⭐ **Star this repository if you find it helpful!**

*This roadmap is continuously updated with the latest developments in Generative AI and Machine Learning.*
