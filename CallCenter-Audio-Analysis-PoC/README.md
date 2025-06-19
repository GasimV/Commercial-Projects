# 📞 CallCenter-Audio-Analysis-PoC

**Client**: Pasha Bank  
**Type**: Proof of Concept  
**Goal**: Analyze recorded call center conversations after each call and generate insights.

---

## 🚀 Features

This project performs end-to-end audio analysis:

- 🎙️ **Speaker Diarization** – Segment and attribute speech by speaker.
- 🧠 **Speaker Identification** – Identify customer vs. operator.
- ✍️ **Summarization** – Generate concise summaries of calls.
- 📊 **Quality Classification** – Rate operator's response quality using BERT + KNN.
- 🖥️ **Real-time UI** – Live dashboard using FastAPI + WebSocket to monitor processing stages.

---

## 🗂️ Project Structure

```
├── main.py                         # FastAPI app + file watcher + WebSocket
├── models_upd.py                   # Core models: diarizer, identifier, summarizer, classifier
├── knn_model/                      # Pretrained KNN classifier
├── mt5-summarize-callcenter-az-final/  # Summarization model
├── speaker-identification/         # Speaker role classification BERT model
├── records/                        # Source audio files
├── watch_folder/                   # Processed or monitored files
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

> Requires Python 3.8+

1. Clone this repo inside the **Commercial-Projects** repo:
```bash
git clone https://github.com/GasimV/Commercial-Projects/CallCenter-Audio-Analysis-PoC.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
---

🧩 PyTorch Installation

Since the GPU version is recommended, but CPU fallback should also work:

⚡ Recommended (GPU - CUDA 12.1+)

```bash
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

💡 Use [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to pick the correct version for your GPU and CUDA driver.

🐢 Optional (CPU only - for testing or development)

```bash
pip install torch torchaudio
```

⚠️ Warning: running on CPU is significantly slower and may not support full real-time performance.

---

4. Make sure ffmpeg is installed for audio processing:
```bash
# Windows: download from https://ffmpeg.org/download.html
# Linux/macOS: use package manager (e.g., brew install ffmpeg)
```

---

## 🧪 Running the App

```bash
python main.py
```

- Server starts at: `http://localhost:8001`
- WebSocket endpoint: `ws://localhost:8001/ws`
- Monitors folder: `...\records\1005`

---

## 🧠 Models Used

- 🤖 Whisper (via `stable-whisper`) – Speech recognition
- 🗣️ `pyannote.audio` – Diarization
- 🔍 Custom BERT model – Speaker role classification
- 📝 MT5-small – Summarization
- 🎯 KNN classifier – Operator response rating

---

## 📌 Notes

- You may need a Hugging Face token (`hf_...`) to access certain models.
- Avoid committing sensitive tokens, models, or large `.wav` files.

---

## 📷 Demo UI

> Automatically opens and updates in browser — shows steps like:
1. File detected
2. Diarization
3. Speaker Identification
4. Summary
5. Classification Result (Good/Poor)

---

## 👤 Author

This project was authored by **Gasym A. Valiyev** as part of a private proof-of-concept developed for **Pasha Bank**.

---
