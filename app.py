from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "sentiment.db")

# ---------------- LOAD MODEL (once, at startup) ----------------
print("Loading sentiment analysis model... (first run may take a minute)")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Model loaded successfully.")


# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_result(text, sentiment, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (text, sentiment, confidence, created_at) VALUES (?, ?, ?, ?)",
        (text, sentiment, confidence, datetime.now().strftime("%d %b %Y, %I:%M %p"))
    )
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT text, sentiment, confidence, created_at FROM history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"text": r[0], "sentiment": r[1], "confidence": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment, COUNT(*) FROM history GROUP BY sentiment")
    rows = cursor.fetchall()
    conn.close()
    stats = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for sentiment, count in rows:
        stats[sentiment] = count
    return stats


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    if len(text) > 2000:
        return jsonify({"error": "Text is too long. Please limit to 2000 characters."}), 400

    try:
        result = sentiment_pipeline(text)[0]
        raw_label = result["label"]  # POSITIVE or NEGATIVE (from model)
        raw_score = result["score"]  # confidence for that raw label, 0-1

        # The base model only outputs Positive/Negative. When it isn't
        # confident either way, we treat the text as Neutral rather than
        # forcing a strong label — this covers plain/factual statements
        # (e.g. "my name is Vikas") that aren't really opinions.
        NEUTRAL_THRESHOLD = 0.60
        if raw_score < NEUTRAL_THRESHOLD:
            sentiment = "NEUTRAL"
            confidence = round(raw_score * 100, 2)
        else:
            sentiment = raw_label
            confidence = round(raw_score * 100, 2)

        save_result(text, sentiment, confidence)

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "text": text
        })
    except Exception as e:
        return jsonify({"error": f"Something went wrong while analyzing: {str(e)}"}), 500


@app.route("/history")
def history():
    return jsonify({
        "history": get_history(),
        "stats": get_stats()
    })


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
