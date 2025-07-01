# Anomaly Detection in Persian Text Data Using GAN and Logistic Regression (Lasso & Ridge)

This project focuses on detecting anomalous comments in user-generated Persian text, specifically within product reviews from online platforms. It combines classic logistic regression models (Lasso and Ridge) with data preprocessing techniques and sets the foundation for incorporating data augmentation through Generative Adversarial Networks (GAN).
## 🎯 Project Objectives
- Detect anomalous or non-genuine comments in user reviews
- Improve model accuracy by generating synthetic text data using GAN
- Compare performance between L1 (Lasso) and L2 (Ridge) regularized logistic regression
- Provide a reproducible NLP pipeline for Persian anomaly detection
  
  ## 🛠 Methods and Techniques
- **Data Preprocessing**: 
  - Removing punctuation and digits
  - Stopword removal using a custom Persian stopword list
  - Word stemming with [`parsivar`](https://github.com/ICTRC/Parsivar)
- **Feature Extraction**:
  - Bag-of-Words (BoW) representation
- **Modeling**:
  - `Logistic Regression` with `L1` (Lasso)
  - `Logistic Regression` with `L2` (Ridge)
- **Evaluation Metrics**:
  - Confusion Matrix
  - Class-wise Accuracy (Recommended / Not Recommended)
