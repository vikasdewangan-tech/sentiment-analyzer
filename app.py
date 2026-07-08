from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "sentiment.db")

# ---------------- LOAD MODELS (once, at startup) ----------------
print("Loading sentiment analysis model... (first run may take a minute)")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Loading emotion detection model...")
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)
print("Models loaded successfully.")

EMOTION_EMOJI = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐",
}

# Which emotions are allowed to be shown for each sentiment bucket. This
# keeps the emoji consistent with the sentiment badge — e.g. we never show
# a "Joy" emoji next to a "Negative" result, even if the emotion model
# leans that way on its own (the two models look at text differently).
POSITIVE_EMOTIONS = {"joy", "surprise"}
NEGATIVE_EMOTIONS = {"anger", "sadness", "fear", "disgust"}


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
    # Safe migration: add emotion/emoji columns if this is an older db
    # that was created before this feature existed.
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(history)")]
    if "emotion" not in existing_cols:
        cursor.execute("ALTER TABLE history ADD COLUMN emotion TEXT DEFAULT ''")
    if "emoji" not in existing_cols:
        cursor.execute("ALTER TABLE history ADD COLUMN emoji TEXT DEFAULT ''")
    conn.commit()
    conn.close()


def save_result(text, sentiment, confidence, emotion, emoji):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (text, sentiment, confidence, created_at, emotion, emoji) VALUES (?, ?, ?, ?, ?, ?)",
        (text, sentiment, confidence, datetime.now().strftime("%d %b %Y, %I:%M %p"), emotion, emoji)
    )
    conn.commit()
    conn.close()


def get_history(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT text, sentiment, confidence, created_at, emotion, emoji FROM history ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "text": r[0], "sentiment": r[1], "confidence": r[2], "created_at": r[3],
            "emotion": r[4] or "", "emoji": r[5] or "🙂"
        }
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

        # The base model only outputs Positive/Negative, and it tends to be
        # very confident even on plain/factual statements (e.g. "my name is
        # Vikas" can score 99%+ Positive). A high threshold is needed to
        # actually catch neutral-leaning text as Neutral.
        NEUTRAL_THRESHOLD = 0.90
        if raw_score < NEUTRAL_THRESHOLD:
            sentiment = "NEUTRAL"
            confidence = round(raw_score * 100, 2)
        else:
            sentiment = raw_label
            confidence = round(raw_score * 100, 2)

        # Granular emotion detection (joy/sadness/anger/fear/surprise/disgust/neutral).
        # We constrain the chosen emotion to match the final sentiment bucket so the
        # emoji never contradicts the sentiment badge (e.g. no "Joy" on a Negative result).
        all_emotions = emotion_pipeline(text)[0]  # list of {label, score} for every emotion
        all_emotions.sort(key=lambda e: e["score"], reverse=True)

        if sentiment == "POSITIVE":
            allowed = POSITIVE_EMOTIONS
        elif sentiment == "NEGATIVE":
            allowed = NEGATIVE_EMOTIONS
        else:
            allowed = None  # Neutral: allow the model's top pick as-is

        if allowed:
            match = next((e for e in all_emotions if e["label"].lower() in allowed), None)
            if match:
                emotion_label = match["label"].lower()
            else:
                # Fallback if none of the allowed emotions scored highly
                emotion_label = "joy" if sentiment == "POSITIVE" else "sadness"
        else:
            emotion_label = all_emotions[0]["label"].lower()

        emoji = EMOTION_EMOJI.get(emotion_label, "🙂")

        save_result(text, sentiment, confidence, emotion_label, emoji)

        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "emotion": emotion_label,
            "emoji": emoji,
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
