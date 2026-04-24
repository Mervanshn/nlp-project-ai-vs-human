# 🤖 AI vs. Human Text Classification

**BIM432 Natural Language Processing - Term Project**

This repository contains the source code, datasets, and final report for the BIM432 NLP project. The primary goal of this project is to develop a robust machine learning pipeline capable of distinguishing between human-written text and AI-generated text with high accuracy.

## 📌 Project Overview
With the rapid advancement of Large Language Models (LLMs), detecting AI-generated content has become a critical challenge in NLP. In this project, we compared a traditional statistical machine learning approach (**Logistic Regression** with TF-IDF) against a deep learning architecture (**Bidirectional LSTM**). 

By utilizing sequence-aware modeling and cost-sensitive learning, our final deep learning model achieved an overall accuracy of **98.45%**.

## 📊 Dataset
* **Source:** [AI vs Human Text Dataset (Kaggle)](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)
* **Size:** 30,000 balanced samples (15,000 Human, 15,000 AI-Generated) to prevent majority-class bias.
* **Note:** Due to GitHub's file size limits, only `dataset_sample.csv` is included in this repository.

## 🏗️ Repository Structure
```text
nlp-project-ai-vs-human/
│
├── data/                  # Contains the sample dataset
├── notebooks/             # Jupyter notebooks for initial data exploration
├── src/                   # Source code files
│   ├── preprocessing.py   # Text cleaning and tokenization
│   ├── feature_extraction.py # TF-IDF and sequence padding
│   ├── main_deep.py       # Bi-LSTM model training
│   └── evaluate.py        # Evaluation metrics and plotting
├── results/               # Saved models and exported figures (Accuracy/Loss/Confusion Matrix)
├── report/                # Final project report (PDF)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation