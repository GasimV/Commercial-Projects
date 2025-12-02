# Call Center Audio Analysis – Proof of Concept

*Demo project by **Gasym A. Valiyev***
*For the Food Safety Agency of the Republic of Azerbaijan (Azərbaycan Respublikasının Qida Təhlükəsizliyi Agentliyi – AQTA) Call Center*

---

## Overview

This Proof-of-Concept demonstrates a **real-time audio monitoring and analysis platform** designed for the AQTA Call Center. The system ingests recorded or live audio calls, processes them through speech analysis pipelines, and provides **transparent, structured insights** into the interaction quality.

It integrates **signal processing, AI transcription, and NLP models** to ensure that customer–operator conversations are not only stored but also evaluated for service quality, topic relevance, and customer sentiment.

---

## Key Features

### 🔊 Audio Ingestion

* **Automatic monitoring** of new audio files in a source directory.
* **Web UI live recording** support via browser microphone.
* **Real-time WebRTC streaming** through OpenAI Realtime API for direct speech-to-speech (STS) interaction.

### 🎙️ Pre-Processing

* **Denoising** of raw recordings with a pretrained DNS64 model.
* Normalization to **16 kHz mono PCM WAV** for consistency.

### 🤖 AI-Powered Analysis (Google Gemini)

* **Diarization & Transcription**: Speaker separation into *müştəri* (customer) and *operator*, with full turn-by-turn transcript.
* **Emotion Analysis**: Emotion labeling per speaker turn (`Xoşbəxt`, `Neytral`, `Kədərli`, `Qəzəbli`).
* **Topic Classification**: Automatic assignment to categories such as `Hesab Məlumatları`, `Kredit Müraciəti`, `Kart Sifarişi`, `Texniki Dəstək`, `Şikayət`, `Ümumi Məlumat`.
* **Summary Generation**: Concise, one-sentence description of the call’s purpose and outcome.
* **Quality Assessment**: Service quality rating (1–5 stars) with “Yaxşı cavab” or “Pis cavab” classification.

### 📊 Web Dashboard

* **Step-by-step visualization** of the pipeline (file detected → transcription → emotion trend → summary → classification).
* **Interactive transcript view** with expandable sections.
* **Parallel playback** of original and denoised audio.
* **Live status monitoring** of connections and processing steps.

### 🛠️ Realtime Demo Mode

* Secure ephemeral sessions with **OpenAI Realtime API**.
* Operator instructions tailored for AQTA use case:

  * Speak politely, clearly, and concisely.
  * Ask clarifying questions if information is missing.
  * Do **not** provide legal or medical advice—redirect to official resources (`aqta.gov.az` or 1003 Call Center).

---

## Technology Stack

* **Backend:** FastAPI (Python), WebSockets, Watchdog file observer.
* **Audio Processing:** PyTorch, DNS64 Denoiser, pydub, soundfile.
* **Analysis:** Google Gemini API (`gemini-2.5-flash`).
* **Realtime STS:** OpenAI Realtime API with WebRTC.
* **Frontend:** Modern HTML/CSS/JS interface with live WebSocket updates.
* **Deployment:** Uvicorn server with CORS enabled.

---

## Demo Workflow

1. **Audio Input**

   * Upload or record live call audio.
   * Files automatically normalized and denoised.

2. **AI Pipeline Execution**

   * Gemini model produces transcript, emotions, summary, topic, and quality rating.
   * Intermediate results streamed to frontend in real-time.

3. **Dashboard Visualization**

   * Transcript displayed with emotion indicators.
   * Customer emotion trend summarized.
   * Service quality scored with star rating and classification.

---

## Purpose

This PoC demonstrates how **AI-driven call analysis** can support:

* **Monitoring service quality** in real-time.
* **Identifying recurring topics and complaints.**
* **Improving training of call center operators.**
* **Ensuring transparency and accountability** in citizen interactions with AQTA.

---

## Disclaimer

This project is a **demo prototype** and not a production-ready solution.

* The analysis relies on third-party AI services (Google Gemini, OpenAI Realtime).
* Audio files may be transmitted to external APIs for processing.
* For official deployments, further steps must be taken regarding **data privacy, security, and compliance**.