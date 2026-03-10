# Decision Trees & Advanced Ensemble Methods

## 📌 Overview

This repository explores the power of Decision Trees and advanced gradient boosting frameworks across different classification tasks. It features end-to-end machine learning pipelines, including complex dimensionality reduction, advanced feature engineering, model selection, and hyperparameter tuning to achieve high-performance metrics.

## ✨ Projects Included

### 1. Credit Card Fraud Detection (`Credit_card_fraud.ipynb`)

This notebook focuses on identifying fraudulent credit card transactions using advanced data preprocessing and the **CatBoost Classifier**.

* **Dimensionality Reduction & Mixed Data Handling:** Utilized **PCA** (Principal Component Analysis) for continuous variables, **MCA** (Multiple Correspondence Analysis) for categorical variables, and **FAMD** (Factor Analysis of Mixed Data) to handle datasets containing both.
* **Feature Engineering:** Crafted new, highly predictive features from the existing dataset to improve model distinction between anomalous and normal behavior.
* **Results:** The final CatBoost model achieved a robust **F1-Score of 0.9153**, demonstrating high precision and recall on an inherently imbalanced dataset.

### 2. Glass Identification (`Glass_identification.ipynb`)

This notebook is a comparative study of various tree-based ensemble methods for multi-class classification, aiming to identify different types of glass based on their chemical properties.

* **Models Evaluated:** * Random Forest
* Gradient Boosting Classifier
* CatBoost Classifier


* **Optimization:** Conducted rigorous hyperparameter tuning (e.g., adjusting tree depth, learning rate, and the number of estimators) across all models to systematically improve accuracy and generalization.

## 🧮 The Mathematics (Trees & Dimensionality Reduction)

**Information Gain in Decision Trees**
At the core of these models is the decision tree, which splits data to maximize purity. The splitting criterion often relies on maximizing Information Gain ($IG$), which reduces Entropy ($H$):

$$IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$

Where $D$ is the dataset, $A$ is the feature to split on, and $D_v$ is the subset of $D$ where feature $A$ has value $v$.

**Principal Component Analysis (PCA)**
To handle high-dimensional continuous data in the fraud dataset, PCA finds the orthogonal axes (principal components) that maximize the variance of the projected data. We achieve this by computing the eigenvectors and eigenvalues of the data's covariance matrix $\Sigma$:

$$\Sigma \mathbf{v} = \lambda \mathbf{v}$$

Where $\mathbf{v}$ represents the principal components and $\lambda$ represents the amount of variance captured by each component.

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* Jupyter Notebook or JupyterLab
* Standard Data Science Stack (NumPy, Pandas, Scikit-Learn)
* Specialized Libraries: `catboost`, `prince` (or equivalent for MCA/FAMD)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/decision-trees-ensembles.git
cd decision-trees-ensembles

```


2. Install the required dependencies:
```bash
pip install numpy pandas scikit-learn catboost prince jupyter

```



## 💻 Usage

To view the analyses and model training processes, launch Jupyter Notebook:

```bash
jupyter notebook

```

From there, you can open either `Credit_card_fraud.ipynb` or `Glass_identification.ipynb` and run the cells sequentially.

## 📂 Project Structure

* `Credit_card_fraud.ipynb` - Pipeline for fraud detection using FAMD/PCA and CatBoost.
* `Glass_identification.ipynb` - Comparative analysis and hyperparameter tuning of Random Forest, Gradient Boosting, and CatBoost.
* *(Include your data folder or CSV files here if they are part of the repository)*.
