# 🎬 Movie Review Sentiment Classifiers

This repository presents two approaches to classifying movie reviews as positive or negative:

- **Classical Machine Learning** using traditional NLP techniques.
- **Deep Learning (DL)** using neural networks and word embeddings.

## 📁 Project Structure

- `Movie sentiment classifier Classical.ipynb`: Implements sentiment classification using classical ML models.
- `Movie sentiment classifier DL.ipynb`: Implements sentiment classification using a Deep Learning model (LSTM-based).

---

## 📌 Problem Statement

The goal is to **build a binary sentiment classifier** that can automatically predict whether a given movie review is positive or negative using the IMDB dataset.

---

## 📊 Dataset

- Source: IMDB Movie Review dataset via TensorFlow Datasets (`tensorflow_datasets`)
- Each review is labeled as `0` (negative) or `1` (positive).
- Dataset split:
  - Training: 25,000 samples
  - Test: 25,000 samples

---

## 🧠 Classical ML Approach (`Movie sentiment classifier Classical.ipynb`)

### ✅ Workflow

1. **Data Preprocessing**
   - Tokenization and padding using TensorFlow's `Tokenizer`
   - Convert text into sequences
2. **Feature Extraction**
   - Text converted into Bag-of-Words or TF-IDF representation
3. **Model Training**
   - Logistic Regression (`sklearn.linear_model.LogisticRegression`)
   - Random Forest (`sklearn.ensemble.RandomForestClassifier`)
4. **Evaluation**
   - Accuracy, Confusion Matrix, Classification Report

### 🔍 Observations

- Logistic Regression performed surprisingly well with minimal preprocessing.
- Random Forest offered competitive results but slower training.

---

## 🤖 Deep Learning Approach (`Movie sentiment classifier DL.ipynb`)

### ✅ Workflow

1. **Text Vectorization**
   - Use of Keras `TextVectorization` layer
   - Vocabulary size: 10,000
2. **Model Architecture**
   - Embedding Layer
   - Bidirectional LSTM
   - Dense layers with Dropout
3. **Training**
   - Epochs: 10
   - Optimizer: Adam
   - Loss: Binary Crossentropy
4. **Evaluation**
   - Accuracy and validation loss tracked
   - Plotting training history

### 📈 Performance

- The LSTM-based DL model outperformed classical methods in accuracy.
- Handles long-term dependencies in review texts better.

---

## 📦 Requirements

```bash
pip install tensorflow scikit-learn matplotlib
```
## 🚀 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/Auwal007/Movie-Review-Sentiment-Classifier.git
    cd movie-sentiment-classifier
    ```

2. Open either notebook:
    - **Classical**: `Movie sentiment classifier Classical.ipynb`
    - **Deep Learning**: `Movie sentiment classifier DL.ipynb`

3. Run all cells in a Jupyter Notebook environment.

---

## 🧠 Key Learnings

- Classical models are fast and interpretable but limited in handling context.
- DL models require more data and compute but offer better performance especially in NLP tasks.

