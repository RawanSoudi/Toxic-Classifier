# Toxic Comments Classification 

A multi-approach toxic comment classifier combining traditional NLP techniques with transformer-based models for comparative analysis using this dataset: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification

## Features

### 1. Advanced Text Preprocessing Pipeline
- **Text Cleaning**:
  - Non-ASCII character removal
  - Number replacement
  - Whitespace normalization
- **Token Processing**:
  - Custom tokenization with NLTK
  - Stopword removal (English)
  - Dual-lemmatization (noun + verb forms)
  - Case normalization
  - Punctuation stripping
- **Pipeline Architecture**:
  ```python
  def TextNormalization(text):
      # Integrated cleaning and token processing
      return processed_words
### 2. Vectorization Approaches
  TF-IDF Vectorization:
  
  Scikit-learn implementation
  
  Custom preprocessing integration

### 3. Model Architectures
  Logistic Regression Classifier

### 4. Evaluation:

  5-fold cross-validation
  
  ROC-AUC scoring

### 5. DistilBERT Transformer Model
  Pretrained model: distilbert-base-uncased-finetuned-sst-2-english
  
  GPU-accelerated inference
  
  Probability extraction for toxicity scoring
