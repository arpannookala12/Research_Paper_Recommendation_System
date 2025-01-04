# 📚 Research Paper Recommendation System

## 🌟 Overview
This repository contains the code and resources for the **Research Paper Recommendation System**. The system is designed to provide precise academic paper suggestions based on abstract similarity. It leverages advanced Natural Language Processing (NLP) techniques to measure semantic similarity and deliver context-aware recommendations across diverse research categories. 

Key features include:
- **TF-IDF** for initial similarity measurements.
- **Sentence-BERT (SBERT)** models for dense semantic embeddings.
- **AllenAI-Specter** for citation-based contextual relevance.
- Integration of **Retrieval-Augmented Generation (RAG)** for explainability.

---

## 📋 Dataset and Preprocessing

### Dataset Source and Composition
- **Source**: The dataset was sourced from the **[ArXiv Metadata Snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)**, containing metadata for **2.6 million research papers** across various scientific disciplines.
- **Fields Included**: Titles, authors, categories, comments, abstracts, and publication dates.
- **Initial Sampling**: To ensure balanced representation across research categories, a **stratified sample** of **100,000 papers** was selected.

### Stratification and Filtering
- **Stratified Sampling**: Categories with fewer than 20 papers were excluded to maintain balance.
- **Final Dataset**: The filtering process resulted in a refined dataset of **99,942 papers**, distributed evenly across **149 unique categories**.

### Preprocessing Steps
- **Enhanced Text Field**: Titles, authors, categories, comments, and abstracts were combined to provide richer contextual information for similarity analysis.
- **Text Cleaning**:
  - Removal of stopwords.
  - Conversion to lowercase.
  - Lemmatization to standardize word forms.
- **TF-IDF Vector Generation**: Processed abstracts were transformed into sparse vectors for the baseline model.
- **Embedding Preparation**: Abstract embeddings were generated using SBERT and AllenAI-Specter for advanced models.

---

## 🔑 Key Notebooks and Their Functions

### 1. 🛠️ **TF-IDF Approach (Notebook 1)**
- **Purpose**: Baseline model for abstract similarity using sparse vectors.
- **Key Steps**:
  - Preprocessing abstracts.
  - Generating and storing TF-IDF vectors in LanceDB.
  - Evaluating similarity using cosine similarity and hybrid scoring.

### 2. 🧠 **SBERT Pre-trained Model: all-MiniLM-L6-v2 (Notebook 2)**
- **Purpose**: Dense semantic embeddings for improved recommendation accuracy.
- **Key Steps**:
  - Embedding generation using stratified data.
  - Efficient storage in LanceDB.
  - Evaluation with hybrid scoring (category relevance, clustering similarity, temporal relevance).

### 3. 📊 **SBERT Pre-trained Model: allenai-specter (Notebook 3)**
- **Purpose**: High-dimensional embeddings optimized for citation-based tasks.
- **Key Steps**:
  - Generating embeddings for academic abstracts.
  - Batch testing and relevance evaluation using hybrid scoring.

### 4. 🔍 **SBERT Fine-Tuning: all-MiniLM-L6-v2 (Notebook 4)**
- **Purpose**: Domain-specific fine-tuning of SBERT for enhanced performance.
- **Key Steps**:
  - Training with abstract pairs (positive/negative labels).
  - Evaluating using metrics like Precision@k, Recall@k, and MRR.

### 5. 🌐 **Streamlit Application (Notebook 5)**
- **Purpose**: A user-friendly interface for abstract-based recommendations.
- **Key Steps**:
  - Integration of embeddings with RAG for explainability.
  - Real-time recommendations with metadata filtering.

---

## 🧪 Results

- **TF-IDF Baseline**: Delivered acceptable results but lacked semantic depth.
- **SBERT Pre-trained (all-MiniLM-L6-v2)**: Showed significant improvement in capturing semantic relationships.
- **SBERT Fine-Tuned**: Achieved an accuracy of **78.9%** and F1 Score of **79.64%** for domain-specific tasks.
- **AllenAI-Specter**: Consistently delivered the most contextually accurate recommendations, excelling in research categories like reinforcement learning and traffic signal control.

🚀 *"AllenAI-Specter embeddings provided the most contextually rich and accurate recommendations."*

---

## 🛠️ Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/arpannookala12/Research-Paper-Recommendation-based-on-Abstract-Similarity.git
