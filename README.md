# 🚀 Gen AI CheatSheet

*(I'm actively exploring more resources and refining this roadmap to make it more detailed and genuinely helpful — so ⭐ it if you find it valuable!)*

---

## **0. Math Foundations**

| S.No | Title          | Side Topics                                   | Resources                                                                 |
| ---- | -------------- | --------------------------------------------- | ------------------------------------------------------------------------- |
| 0    | Math for ML/DL | Linear Algebra, Probability, Statistics, Calc | [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) · [CampusX](https://youtube.com/playlist?list=PLKnIA16_RmvbYFaaeLY28cWeqV-3vADST) · [Stanford CS229](https://see.stanford.edu/Course/CS229) |

---

## **1. Python Basics**

| S.No | Title         | Side Topics                                                     | Resources                                               |
| ---- | ------------- | --------------------------------------------------------------- | ------------------------------------------------------- |
| 1    | Python Basics | basics, data structures, file handling, exception handling, OOP | [FreeCodeCamp](https://youtu.be/eWRfhZUzrAc)            |

---

## **2. Streamlit**

| S.No | Title     | Side Topics                   | Resources                                                   |
| ---- | --------- | ----------------------------- | ----------------------------------------------------------- |
| 2    | Streamlit | Streamlit basics, UI building | [Data Professor](https://youtu.be/yKTEC1Y5bEQ)              |

---

## **3. Machine Learning — Core Basics**

| S.No | Title              | Side Topics                                                                 | Resources                                                                 |
| ---- | ------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 3    | ML Fundamentals    | Classification, Regression, Pipelines, Feature Engineering                  | [CampusX](https://youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH), [MIT 6.S191](http://introtodeeplearning.com/) |
| 4    | ML Evaluation      | Accuracy, Precision, Recall, Confusion Matrix, ROC-AUC                      | [StatQuest](https://www.youtube.com/@statquest)                           |
| 5    | Feature Scaling    | Normalization, Standardization, MinMax, Robust                              | [Scikit-learn Docs](https://scikit-learn.org/stable/modules/preprocessing.html) |
| 6    | Data Labeling      | Manual annotation, Label Studio, Roboflow                                   | [Label Studio](https://labelstud.io/), [Roboflow](https://roboflow.com)   |

---

### 🛠 **P1: Core ML Projects**

| S.No | Project Name               | Explanation                                          | Datasets                   | Resources                         |
| ---- | -------------------------- | ---------------------------------------------------- | -------------------------- | --------------------------------- |
| 1    | ML Classification App      | Build a classification app using sklearn + Streamlit | Iris, Titanic, MNIST       | Kaggle, sklearn docs, Streamlit   |
| 2    | Regression Price Predictor | Housing price prediction with feature engineering    | Boston Housing, California | scikit-learn, seaborn, matplotlib |

---

## **4. Machine Learning — Deep Dive**

| S.No | Title               | Side Topics                                                              | Resources                                                                 |
| ---- | ------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- |
| 7    | Unsupervised ML     | Clustering (K-Means, DBSCAN, Hierarchical), Dimensionality Reduction (PCA, t-SNE, UMAP) | [StatQuest](https://www.youtube.com/@statquest) |
| 8    | Ensemble Methods    | Bagging, Boosting (XGBoost, LightGBM), Stacking                         | [Krish Naik](https://www.youtube.com/@krishnaik06) |
| 9    | Hyperparameter Tuning | GridSearchCV, RandomSearch, Optuna, Bayesian Opt                       | [Optuna Docs](https://optuna.org/)                                       |
| 10   | Core ML Concepts    | Bias-variance tradeoff, Underfitting/Overfitting, Regularization (L1/L2) | [Andrew Ng ML](https://www.coursera.org/learn/machine-learning)          |

---

## **5. ML for NLP**

| S.No | Title      | Side Topics                                    | Resources                                                                 |
| ---- | ---------- | ---------------------------------------------- | ------------------------------------------------------------------------- |
| 11   | ML for NLP | Text preprocessing, OHE, BoW, TF-IDF, Word2Vec | [Krish Naik](https://youtube.com/playlist?list=PLZoTAELRMXVNNrHSKv36Lr3_156yCo6Nn) |

---

### 🛠 **P2: NLP Projects**

| S.No | Project Name      | Explanation                                           | Datasets               | Resources           |
| ---- | ----------------- | ----------------------------------------------------- | ---------------------- | ------------------- |
| 1    | Text Classifier   | Spam detection or sentiment analysis using BoW/TF-IDF | SMS Spam, IMDb Reviews | Kaggle, sklearn NLP |
| 2    | Word2Vec Explorer | Visualize similarity between words using Word2Vec     | Google News Word2Vec   | Gensim, seaborn     |

---

## **6. DL Basics**

| S.No | Title     | Side Topics                                         | Resources                                                                 |
| ---- | --------- | --------------------------------------------------- | ------------------------------------------------------------------------- |
| 12   | DL Basics | Neural Nets, Loss Functions, Optimizers, Activations | [3Blue1Brown](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi), [MIT 6.S191](http://introtodeeplearning.com/) |

---

## **7. Core DL**

| S.No | Title        | Side Topics               | Resources                                                                 |
| ---- | ------------ | ------------------------- | ------------------------------------------------------------------------- |
| 13   | NN + ANN     | Basics of neural nets     | [DeepLearning.ai](https://www.deeplearning.ai/), [3Blue1Brown](https://www.youtube.com/@3blue1brown) |
| 14   | CNN          | ConvNets for vision       | [Stanford CS231n](http://cs231n.stanford.edu/), [FastAI](https://course.fast.ai/) |
| 15   | RNN + LSTM   | Sequential data modeling  | [Stanford CS224N](http://web.stanford.edu/class/cs224n/), DeepLearning.ai |

---

## **8. DL Frameworks**

| S.No | Title              | Side Topics                       | Resources                                                                 |
| ---- | ------------------ | --------------------------------- | ------------------------------------------------------------------------- |
| 16   | PyTorch/TensorFlow | tensors, training, model building | [PyTorch Docs](https://pytorch.org), [TensorFlow Docs](https://www.tensorflow.org), [PyTorch YouTube](https://youtu.be/Z_ikDlimN6A?si=rzfpSwj8zffWDUXb) |

---

### 🛠 **P3: Projects**

| S.No | Project Name        | Explanation                        | Datasets              | Resources                 |
| ---- | ------------------- | ---------------------------------- | --------------------- | ------------------------- |
| 1    | Image Classifier    | Build CNN to classify cats vs dogs | Dogs vs Cats (Kaggle) | TensorFlow/Keras, PyTorch |
| 2    | Sentiment with LSTM | Sentiment prediction using LSTM    | IMDb, Twitter         | Keras, torchtext          |

---

## **9. Transformers**

| S.No | Title           | Side Topics                                                                                         | Resources                                                                                     |
| ---- | --------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 15   | Transformers    | Self-attention, Multi-head, Positional Encoding, Encoder-Decoder Arch, Layer Norm, Masked MHA, etc. | [YouTube](https://www.youtube.com/live/SMZQrJ_L1vo)                                           |
| 16   | 🆕 Tokenization | BPE, GPT-2 tokenizer, HF tokenizer                                                                  | [YouTube](https://youtu.be/ZhAz268Hdpw), [Paper](https://arxiv.org/pdf/1706.03762)            |

---

## **10. Introduction to Gen AI**

| S.No | Title              | Side Topics                                                     | Resources                                                                 |
| ---- | ------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 17   | Intro to Gen AI    | AI vs ML vs DL vs GenAI, How GPT/LLM are trained, LLM evolution | [YouTube](https://youtu.be/X7Zd4VyUgL0), [YouTube](https://youtu.be/d4yCWBGFCEs) |
| 18   | 🆕 LLM Evaluation  | BLEU, ROUGE, Perplexity, Human Evaluation                       |                                                                           |
| 19   | 🆕 Ethics & Safety | Hallucination, bias, responsible deployment                     |                                                                           |

---

## **11. Introduction to LangChain**

| S.No | Title        | Side Topics                                           | Resources                                                                 |
| ---- | ------------ | ----------------------------------------------------- | ------------------------------------------------------------------------- |
| 20   | LangChain    | Components, Ollama, Huggingface, Groq                 | [LangChain Docs](https://python.langchain.com/docs/introduction/), [YouTube - LangChain](https://youtu.be/X0btK9X0Xnk) |
| 21   | 🆕 Prompting | Zero-shot, few-shot, chain-of-thought, prompt hacking | OpenAI Cookbook, LangChain Docs                                           |

---

### 🛠 **P4: Projects**

| S.No | Project Name           | Explanation                                     | Datasets | Resources                                  |
| ---- | ---------------------- | ----------------------------------------------- | -------- | ------------------------------------------ |
| 1    | Chatbot with LangChain | Build chatbot using LangChain + LLM + Streamlit | -        | LangChain Docs, Ollama                     |
| 2    | Document Summarizer    | Summarize PDF/Text with LLM                     | -        | LangChain, PyPDF, Huggingface Transformers |

---

## **12. RAG (Retrieval Augmented Generation)**

| S.No | Title | Side Topics                       | Resources                                                                                           |
| ---- | ----- | --------------------------------- | --------------------------------------------------------------------------------------------------- |
| 22   | RAG   | Retrieval pipeline, vector search | [YouTube](https://youtu.be/X0btK9X0Xnk), LangChain RAG Docs, Pinecone Blog                          |

---

### 🛠 **P5: Projects**

| S.No | Project Name            | Explanation                                               | Datasets    | Resources                     |
| ---- | ----------------------- | --------------------------------------------------------- | ----------- | ----------------------------- |
| 1    | PDF Summarizer with RAG | Upload PDF → extract text → chunk → embed → query via LLM | Custom PDFs | LangChain, FAISS, OpenAI/Groq |

---

## **13. Vector Databases**

| S.No | Title               | Side Topics                                  | Resources                                                                  |
| ---- | ------------------- | -------------------------------------------- | -------------------------------------------------------------------------- |
| 23   | Vector Databases    | FAISS, ChromaDB, Pinecone, similarity search | Pinecone Docs, Weaviate, FAISS GitHub                                     |
| 24   | 🆕 Fine-tuning LLMs | PEFT, LoRA, prompt-tuning intro              | Huggingface PEFT, LoRA Explained (YouTube)                                |

---

## **14. FastAPI (Backend for AI)**

| S.No | Title    | Side Topics                           | Resources                                                                 |
| ---- | -------- | ------------------------------------- | ------------------------------------------------------------------------- |
| 25   | FastAPI  | APIs for ML/DL models, deployment     | [FastAPI Docs](https://fastapi.tiangolo.com/), YouTube (FastAPI CrashCourse) |

---

## 🎥 YouTube Channels

- [KrishNaik](https://www.youtube.com/@krishnaik06)  
- [CampusX](https://www.youtube.com/@campusx-official)  
- [IBM Technology](https://www.youtube.com/@IBMTechnology)  
- [Codebasics](https://www.youtube.com/@codebasics)

---

## 📚 Books

- *Hands-On Machine Learning*  
- *Hands-On LLMs*  
- *Hands-On Deep Learning*  
- Other “Hands-On” series books  

---

## 📄 Good Research Papers

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)  
- [LangChain Docs](https://python.langchain.com/docs/introduction/)

---

## 📊 Datasets

- [Kaggle](https://www.kaggle.com)
