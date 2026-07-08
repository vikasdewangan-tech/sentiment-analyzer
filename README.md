# 🧠 AI-Based Sentiment Analyzer

A web application built with **Flask** and **HuggingFace Transformers** that analyzes the sentiment (Positive/Negative) of any text — reviews, comments, or feedback — with a confidence score, visual charts, and analysis history.

Built as part of a Python Developer internship project assignment.

---

## 🚀 Features

- 🤖 Sentiment analysis using a pretrained **DistilBERT** transformer model — classifies text as **Positive, Negative, or Neutral** (Neutral is inferred when the model's confidence falls below a threshold, since the base model is binary)
- 📊 Confidence score with animated visual bar
- 📈 Live doughnut chart showing Positive vs Negative distribution
- 📜 Analysis history stored in a local SQLite database
- 🎨 Modern dark-themed, responsive UI (no page reloads — AJAX-based)
- ⚠️ Input validation and friendly error handling

## 🛠️ Tech Stack

| Component        | Technology                                          |
|-------------------|------------------------------------------------------|
| Language          | Python 3                                              |
| Web Framework     | Flask                                                 |
| NLP Model         | HuggingFace Transformers (DistilBERT SST-2)           |
| Database          | SQLite                                                |
| Frontend          | HTML, CSS, JavaScript, Chart.js                       |

## 📸 Screenshot

*(Add a screenshot here after running the app — drag the image into this README on GitHub's web editor)*

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/sentiment-analyzer.git
   cd sentiment-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > First install may take a few minutes since it downloads PyTorch.

3. **Run the app**
   ```bash
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000`

   > On first run, the model (~260MB) downloads automatically from HuggingFace — this only happens once and needs an internet connection.

## 📁 Project Structure

```
sentiment-analyzer/
├── app.py                 # Flask backend + model inference
├── templates/
│   └── index.html         # Frontend UI (dark theme, Chart.js)
├── requirements.txt       # Python dependencies
└── README.md               # Project documentation
```

## ⚠️ Disclaimer

This project uses a general-purpose pretrained sentiment model for educational/demo purposes. Accuracy may vary for sarcasm, mixed sentiment, or domain-specific text (e.g. medical, legal).

## 👤 Author

**Vikas Dewangan**
Python Developer Intern

---

*Built as part of a Python Developer Internship project assignment.*
