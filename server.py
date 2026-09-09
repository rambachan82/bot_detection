"""
server.py
---------------------------------------
Flask backend that receives the behavior payload from collector.js,
combines it with rule-based checks (honeypot, webdriver flag, time-trap),
runs the ML model for a probability score, and returns a verdict.

Run:
    python train_model.py      # first, to produce model.pkl
    python server.py
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

from train_model import FEATURE_COLUMNS

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # Allow the demo.html frontend (different origin/file) to call this API.
    # In production, replace "*" with your actual frontend origin.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response

MODEL = joblib.load("model.pkl")

# Tunable thresholds
BOT_PROB_THRESHOLD = 0.7      # above this -> reject / show fallback CAPTCHA
BORDERLINE_LOW = 0.4          # between LOW and THRESHOLD -> "uncertain" zone
MIN_TIME_ON_PAGE_MS = 800     # submissions faster than this are auto-suspicious


@app.route("/verify", methods=["POST"])
def verify():
    payload = request.get_json(force=True)

    # ---- 1. Hard rule-based checks (fast, cheap, catch obvious bots) ----
    if payload.get("honeypot_triggered"):
        return jsonify(verdict="bot", reason="honeypot_triggered", bot_probability=1.0)

    if payload.get("webdriver_flag"):
        return jsonify(verdict="bot", reason="webdriver_detected", bot_probability=1.0)

    if payload.get("time_on_page_ms", 0) < MIN_TIME_ON_PAGE_MS:
        return jsonify(verdict="bot", reason="submitted_too_fast", bot_probability=0.95)

    # ---- 2. ML model scoring on engineered features ----
    row = {col: payload.get(col, 0) for col in FEATURE_COLUMNS}
    # Rule-based booleans above are already handled, but keep as numeric features too
    row["honeypot_triggered"] = int(bool(row["honeypot_triggered"]))
    row["webdriver_flag"] = int(bool(row["webdriver_flag"]))

    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    bot_probability = float(MODEL.predict_proba(X)[0, 1])

    # ---- 3. Decision logic ----
    if bot_probability >= BOT_PROB_THRESHOLD:
        verdict = "bot"
    elif bot_probability >= BORDERLINE_LOW:
        verdict = "uncertain"  # frontend should show fallback CAPTCHA here
    else:
        verdict = "human"

    return jsonify(verdict=verdict, bot_probability=bot_probability)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
