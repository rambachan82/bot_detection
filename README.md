# Behavioral Bot Detection — Starter Project

An ML-based "are you human?" verification system that scores users passively
based on how they interact with a page (mouse movement, typing rhythm,
timing), layered with fast rule-based checks.

## Structure

```
bot_detection/
├── frontend/
│   ├── collector.js   # Tracks mouse/keyboard/scroll, computes features
│   └── demo.html      # Sample signup form wired to the collector
└── backend/
    ├── train_model.py # Generates synthetic data + trains RandomForest
    ├── server.py      # Flask API: /verify endpoint
    ├── requirements.txt
    └── model.pkl      # (generated after running train_model.py)
```

## How it works

1. **collector.js** silently records mouse movement, keystroke timing,
   scrolling, and time-on-page while the user fills out a form. It also
   injects a hidden **honeypot** field — invisible to humans, but bots that
   auto-fill every field will trigger it.
2. On submit, the browser computes engineered features (path straightness,
   speed variance, keystroke interval variance, etc.) and sends them to the
   backend.
3. **server.py** first runs cheap rule-based checks (honeypot triggered,
   `navigator.webdriver` flag, submitted too fast), then — if nothing obvious
   fires — runs the ML model for a bot-probability score.
4. Based on the score: `human` (let through), `bot` (block), or `uncertain`
   (show a fallback CAPTCHA — best of both worlds: low friction for real
   users, still catches ambiguous cases).

## Running it

```bash
cd backend
pip install -r requirements.txt

# 1. Train the model (uses synthetic data — swap in real logged sessions later)
python train_model.py

# 2. Start the API
python server.py
```

Then open `frontend/demo.html` in a browser (the fetch call points at
`http://localhost:5000/verify`).

## Next steps to extend this into a full project

- **Replace synthetic data** with real sessions: log `collector.js` payloads
  from real traffic, label them (known bots via rate-limiting/WAF logs,
  humans via completed+verified signups), and retrain.
- **Sequence models**: instead of only summary statistics, feed the raw
  mouse (x, y, t) and keystroke timing sequences into an LSTM/GRU or a
  1D-CNN for richer pattern detection.
- **Anomaly detection variant**: train an autoencoder only on human sessions;
  flag high reconstruction error as bot-like (useful when bot examples are
  scarce or evolve quickly).
- **Add more signals**: TLS/JA3 fingerprint, canvas/WebGL fingerprinting,
  IP reputation lookups, device orientation (mobile), touch pressure.
- **A/B test thresholds**: tune `BOT_PROB_THRESHOLD` / `BORDERLINE_LOW` in
  `server.py` against your false-positive tolerance.
- **Retraining loop**: periodically retrain on freshly labeled data since
  bot behavior evolves (adversarial drift).
- **Evaluate rigorously**: track Precision/Recall/F1/ROC-AUC, but weight
  **false positive rate** heavily — blocking real users is costly.
