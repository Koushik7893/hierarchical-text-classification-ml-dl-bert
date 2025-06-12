# 🧠 Hierarchical Text Classification (HTC) with ML, DL & Transformers

This repository showcases multiple approaches to **Hierarchical Text Classification (HTC)** using traditional machine learning, deep learning (LSTM, HAN), and transformers (DistilBERT). HTC models not only predict labels but also respect the underlying **label hierarchy**, which is essential in domains like document organization, product categorization, and topic modeling.

---

## 📚 Project Overview

We explore five key strategies:

| Notebook | Method | Model Type | Description |
|----------|--------|------------|-------------|
| `01_flat_classifier_logreg_tfidf.ipynb` | Flat Classification | Logistic Regression + TF-IDF | Ignores label hierarchy — treats problem as flat multiclass |
| `02_lcpn_tree_per_node_models.ipynb` | Local Classifier Per Node (LCPN) | Decision Trees / Random Forest | Trains one model at each node in the hierarchy |
| `03_bilstm_amazon_product_hierarchy.ipynb` | Multi-level Output | BiLSTM | One model with separate outputs for each level of the hierarchy |
| `04_hierarchical_attention_network.ipynb` | Hierarchical Attention | HAN (GRU + Attention) | Learns attention at both word and sentence levels |
| `05_transformer_multihead_htc.ipynb` | Transformers with Multi-Heads | DistilBERT + Multi-Dense Heads | Fine-tunes a transformer with multiple heads for level-wise outputs |

---

## 🗂️ Folder Structure

```

hierarchical-text-classification-ml-dl/
│
├── notebooks/
│   ├── 01\_flat\_classifier\_logreg\_tfidf.ipynb
│   ├── 02\_lcpn\_tree\_per\_node\_models.ipynb
│   ├── 03\_bilstm\_amazon\_product\_hierarchy.ipynb
│   ├── 04\_hierarchical\_attention\_network.ipynb
│   └── 05\_transformer\_multihead\_htc.ipynb
│
├── requirements.txt
└── README.md

````

---

## 🧪 Datasets Used

### 🔹 `fetch_20newsgroups`  
Used for flat, LCPN, HAN, and transformer models. Simulated 3-level hierarchy with mapped categories.

### 🔹 Amazon Product Categories (Custom Split)  
Used for LSTM-based hierarchical multi-output learning. Products are labeled with hierarchical category paths (e.g., Electronics → Cameras).

---

## 🔧 Model Highlights

### ✅ Flat Classifier
- TF-IDF + Logistic Regression
- Baseline multiclass classification ignoring label hierarchy

### ✅ LCPN (Local Classifier Per Node)
- Decision Trees or Random Forests trained at each node
- Top-down prediction using node-wise classifiers

### ✅ BiLSTM
- Embedding + BiLSTM + Multi-head Softmax
- Predicts all levels in one forward pass

### ✅ HAN (Hierarchical Attention Network)
- Two-stage attention: word-level → sentence-level
- Handles document-level inputs

### ✅ Transformer (DistilBERT)
- Shared encoder
- Separate output heads for level-wise classification

---

## 📊 Evaluation Metrics

- Accuracy at each level
- Complete label path accuracy
- Macro / Micro F1-scores

---

## 📦 Installation

```bash
git clone https://github.com/Koushim/hierarchical-text-classification-ml-dl.git
cd hierarchical-text-classification-ml-dl
pip install -r requirements.txt
````

---

## ⚙️ Requirements

```txt
scikit-learn
pandas
numpy
matplotlib
seaborn
tensorflow
torch
transformers
```

---

## 📌 Applications

* Document Classification (e.g., legal, medical reports)
* E-commerce Product Categorization
* Web Content Topic Modeling
* News Article Classification
* Academic Paper Taxonomy

---

## 📚 References

* [Hierarchical Attention Networks (Yang et al., 2016)](https://www.aclweb.org/anthology/N16-1174/)
* [A Survey on Hierarchical Text Classification](https://arxiv.org/abs/1905.01646)
* Hugging Face Transformers
* Scikit-learn documentation

---

## 👨‍💻 Author

**Koushik Reddy**
🔗 [Hugging Face](https://huggingface.co/Koushim) 
🔗 [LinkedIn](https://www.linkedin.com/in/koushik-reddy-k-790938257)

---

## 📌 License

This project is open source and available under the [Apache License](LICENSE).
