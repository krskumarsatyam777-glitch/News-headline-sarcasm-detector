# News Headline Sarcasm Detection using DistilBERT

A Natural Language Processing (NLP) project that classifies news headlines as **Satirical** or **Genuine** using a fine-tuned **DistilBERT** transformer model. The project also includes a **TF-IDF + Logistic Regression** baseline for performance comparison and an interactive **Gradio** web application for real-time inference.

**Hugging Face Model:** https://huggingface.co/satyame639291/sarcasm-distilbert

**Live Demo:** https://huggingface.co/spaces/satyame639291/news_headline_scarcasm_detector

---
## Project Overview

Sarcasm detection is a challenging NLP task because sarcasm often depends on context rather than individual words. Traditional machine learning models struggle to capture these contextual relationships.

This project compares:

- **TF-IDF + Logistic Regression** (Classical NLP Baseline)
- **Fine-tuned DistilBERT** (Transformer-based Deep Learning)

The fine-tuned DistilBERT model is deployed through a Gradio application on Hugging Face Spaces, allowing users to classify custom news headlines in real time.

---

## Dataset

**News Headlines Dataset for Sarcasm Detection** (Misra, 2019)

The dataset consists of news headlines collected from **The Onion** (satirical) and **HuffPost** (genuine).

- **Total Headlines:** 28,503
- **Satirical Headlines:** The Onion
- **Genuine Headlines:** HuffPost

Each headline is labeled as:

- **1** → Satirical
- **0** → Genuine

---

## Exploratory Data Analysis

The notebook includes:

- Class distribution analysis
- Headline length and character count distribution
- Word frequency analysis
- Bigram analysis
- Word clouds
- Punctuation usage analysis
- Source domain analysis

All EDA plots are available in [`images/eda/`](images/eda/), with full explanations in the [notebook](notebook/news%20headline%20sarcasm%20detector.ipynb).

---

## Methodology

### Baseline Model

- TF-IDF Vectorization (`max_features=5000`)
- Logistic Regression

### Deep Learning Model

- Pretrained DistilBERT (`distilbert-base-uncased`)
- Fine-tuned using the Hugging Face Transformers Trainer API
- Binary sequence classification (Satirical / Genuine)
- Training configuration:
  - Epochs: 3
  - Learning Rate: 2e-5
  - Batch Size: 32

---

## Project Workflow

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
TF-IDF + Logistic Regression (Baseline)
   │
   ▼
Fine-tune DistilBERT
   │
   ▼
Model Evaluation
   │
   ▼
Deploy with Gradio on Hugging Face Spaces
```

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|:--------:|:---------:|:------:|:--------:|
| TF-IDF + Logistic Regression | **77.34%** | **0.7736** | **0.7734** | **0.7729** |
| Fine-tuned DistilBERT | **92.10%** | **0.9210** | **0.9210** | **0.9209** |

### Key Observations

- Fine-tuned DistilBERT significantly outperformed the TF-IDF + Logistic Regression baseline.
- Contextual language understanding enabled DistilBERT to better identify satirical writing patterns.
- DistilBERT was selected over the full BERT model because it offers a substantially smaller model size with minimal performance loss, making it more suitable for deployment.

---

## Limitations

- The model was trained specifically on **The Onion** and **HuffPost** headlines, so performance may decrease on sarcasm found in conversations, social media posts, reviews, or other domains.
- Labels are derived from the publication source rather than human annotation, meaning the model may partially learn publication-specific writing style.
- Subtle or deadpan sarcasm without obvious contextual cues remains challenging to detect.

---

## Application

The application allows users to:

- Enter a custom news headline
- Classify it as **Satirical** or **Genuine**
- View the prediction confidence
- Test the model using a set of predefined sample headlines

App screenshots are available in [`images/ui_demo/`](images/ui_demo/).

---

## Technologies Used

- Python
- Gradio
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- DistilBERT
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## Project Structure

```text
news-headline-sarcasm-detector/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── SarcasmDetect.json
│
├── notebook/
│   └── news headline sarcasm detector.ipynb
│
└── images/
    ├── eda/
    └── ui_demo/
```

> Note: the fine-tuned DistilBERT model itself is not stored in this repo — `app.py` loads it directly from the Hugging Face Hub (`satyame639291/sarcasm-distilbert`) at runtime.

---

## Installation

Clone the repository

```bash
git clone https://github.com/krskumarsatyam777-glitch/news-headline-sarcasm-detector.git
```

Navigate to the project directory

```bash
cd news-headline-sarcasm-detector
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Run the Gradio application

```bash
python app.py
```

---

## Future Improvements

- Train the model on a larger and more diverse sarcasm dataset to improve generalization across different writing styles and domains.
- Extend the model to detect sarcasm in social media posts, product reviews, and conversational text.
- Evaluate other transformer architectures such as RoBERTa and DeBERTa for performance comparison.
- Add attention visualization or explainability techniques to better interpret model predictions.

---

## Author

Satyam — [GitHub](https://github.com/krskumarsatyam777-glitch) | [Hugging Face](https://huggingface.co/satyame639291)
