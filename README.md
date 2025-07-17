

## SUICIDAL IDEATION DETECTION 

## 📊 Libraries Used

- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `nltk`, `re` for text cleaning
- `tensorflow.keras` for deep learning
- `scikeras.wrappers.KerasClassifier` for model tuning

  
### 1. Business Understanding
The goal of this project is to detect suicide ideation through reddit posts

### 2. Data Understanding
The dataset contains labeled text data with the following columns:

- `text`: The original text input.
- `class`: The target class (suicide or not suicide).
- Additional engineered features like:
  - `char_count`
  - `word_count`
  - `sentiment`

An Exploratory analysis was conducted to understand class distribution, text lengths, and sentiment patterns.


### 3. Data Preparation
Key steps in preprocessing:

- **Text Cleaning**: Lowercasing, removing punctuation, stopwords, special characters.
- **Tokenization**: Splitting text into word tokens.
- **TF-IDF Vectorization**: Capturing important terms from the text.
- **Padding Sequences**: Ensuring uniform input length for neural networks.
- **Meta-Features**: Added `char_count`, `word_count`, and `sentiment` to boost model accuracy.


### 4. Modeling

Three modeling strategies were used:

#### > Machine Learning Models
- **Logistic Regression** with TF-IDF and hyperparameter tuning via GridSearchCV.
- **Support Vector Machine (SVM)** using LinearSVC and TF-IDF features.

#### > Deep Learning Models
- **Multi-Layer Perceptron (MLP)** with word embeddings and dropout regularization.
- **LSTM-style Neural Network** built with `KerasClassifier` and tuned using `GridSearchCV`.

Each model was trained  using  train/test splits.


### 5. Evaluation

Evaluation metrics included:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

The best-performing models were selected based on **recall**, which is critical in tasks such as suicidal ideation , where false negatives are costly.


### 6. Deployment 

- Export trained models using joblib.
- Develop a simple web app or API using FastAPI for inference.


##  Future Improvements

- Integrate LSTM or Bidirectional LSTM layers for better context understanding.
- Use pretrained embeddings like **GloVe** or **BERT**.
- Implement early stopping and learning rate schedules.


