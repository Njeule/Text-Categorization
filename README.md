
# Text Categorization on the 20 Newsgroups Dataset

This repository contains code for a machine learning and deep learning project focused on multi-class text classification using the 20 Newsgroups dataset. The workflow includes preprocessing, descriptive analysis, feature extraction, model training, evaluation, and result visualization.

## 📊 Descriptive Analysis and Dataset Statistics

### Key Code Sections:
- **Data Loading & Extraction:** Extracts the dataset from a `.tar.gz` archive and reads files into a structured format.
- **Initial Overview:** Generates statistics including number of documents, category distribution, vocabulary size, and average document length.
- **Text Cleaning:** Uses regular expressions, stopword removal, and lemmatization with NLTK to clean text.
- **Visualization:** Includes bar plots, word clouds, and a Venn diagram to show vocabulary overlap and class distributions.

These statistics and visualizations support the understanding of the dataset structure and the effectiveness of preprocessing.

## 🤖 Model Training and Evaluation

### Preprocessing Steps:
- Clean text (remove punctuation, digits, special characters, etc.)
- Tokenize and lemmatize words using NLTK
- Remove stopwords and metadata
- Transform labels using LabelEncoder
- Feature extraction:
  - TF-IDF for classical models
  - Tokenization and sequence padding for deep learning models

### Model Training:
- **Classical Models:** SVM, Naïve Bayes, Random Forest, and XGBoost using scikit-learn
- **Deep Learning Models:** CNN and RCNN using TensorFlow/Keras

### Evaluation:
- Train/test split with validation for tuning
- Performance metrics: accuracy, precision, recall, F1-score
- Confusion matrices and training/validation curve plots
- Ensemble Learning (Stacking) with optimized parameters for robust classification

Each model's training and performance is logged, compared, and visualized in the notebook.

## 🚀 Getting Started

This notebook runs on Google Colab. To run it:
1. Upload the `20news-bydate.tar.gz` dataset
2. Follow each code block sequentially
3. Install required packages (`nltk`, `scikit-learn`, `tensorflow`, etc.)
4. Use GPU runtime for training deep learning models

---

**Dataset:** [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)  
