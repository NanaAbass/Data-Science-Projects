# 🏥 Symptom-Based Medical Test Recommender System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end NLP project to recommend medical lab tests based on a user's natural language description of their symptoms. This project explores and compares several recommendation models, from simple keyword-based approaches to advanced semantic search models.

## 🌟 Features

-   **Natural Language Input:** Users can describe their symptoms in plain English.
-   **Multiple Recommendation Models:** The project implements and compares three different models:
    -   TF-IDF Cosine Similarity (Keyword-based)
    -   Word2Vec Sentence Embeddings (Meaning-based)
    -   Sentence-BERT (State-of-the-art semantic search)
-   **Ensemble Recommender:** A hybrid model that combines the strengths of TF-IDF and Word2Vec for more robust recommendations.
-   **Evaluation Framework:** A comprehensive framework to evaluate and compare the performance of the different models using Precision, Recall, and F1-Score.
-   **Interactive Demo:** A command-line interface to interact with the recommender system and see the results in real-time.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

-   Python 3.6+
-   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/medical-test-recommender.git
    cd medical-test-recommender
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn gensim sentence-transformers matplotlib seaborn nltk
    ```

### Running the Interactive Demo

To run the interactive demo, open the `medical_test_recommender.ipynb` notebook in a Jupyter environment and run the last cell, which contains the `interactive_demo()` function.

Alternatively, you can run the following Python code:
```python
from medical_test_recommender import interactive_demo

interactive_demo()
```

## 🛠️ Usage

Once the interactive demo is running, you will be prompted to describe your symptoms. Enter a description of your symptoms in plain English and press Enter. The system will then provide a ranked list of recommended medical tests, along with the likely condition and a confidence score.

```
=================================================================
🏥  MEDICAL TEST RECOMMENDER — Interactive Demo
=================================================================
Describe your symptoms in plain English.
Type "quit" to exit.

⚠️  DISCLAIMER: For educational purposes only.
    Always consult a qualified medical professional.


🔬 Using: Ensemble Recommender (TF-IDF + Word2Vec)
=================================================================
🔍 Query: "Headache, fever, tiredness"
   (TF-IDF weight: 0.6 | Word2Vec weight: 0.4)
=================================================================

📊 Rank 1 | Score: 0.9748 (tfidf=0.3176, w2v=0.8061)
   🦠 Possible condition: Leukemia
   🧪 Recommended tests:
      • ...

...
```

## 📊 Model Evaluation

The different models were evaluated using Precision@3, Recall, and F1-Score on a held-out test set. The results are as follows:

| Model    | Precision@3 | Recall | F1-Score |
| :------- | :---------- | :----- | :------- |
| TF-IDF   | 1.0000      | 1.0000 | 1.000    |
| Word2Vec | 0.9984      | 0.9996 | 0.999    |

*Note: The Sentence-BERT model was not included in the final evaluation due to its computational complexity, but it is available in the notebook for experimentation.*

## 🔮 Future Work

-   **Web Application:** Deploy the recommender system as a web application using Flask or FastAPI.
-   **Fine-tune BioBERT:** Use a pre-trained BERT model that has been fine-tuned on biomedical text to improve accuracy.
-   **Knowledge Graph:** Build a knowledge graph to represent the relationships between symptoms, tests, and diseases, which could lead to more insightful recommendations.
-   **Multilingual Support:** Use a multilingual BERT model to support symptom descriptions in different languages.

## ⚖️ Ethical Considerations

This project is for educational purposes only and is not a substitute for professional medical advice. It is crucial to always consult a qualified medical professional for any health concerns. The system should be used as a tool to assist, not to replace, the expertise of a doctor.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
