# 📞 CallCenter-Audio-Analysis-PoC

**Client**: Pasha Bank   
**Type**: Proof of Concept  
**Author**: Gasym A. Valiyev  
**Status**: Completed

## Executive Summary

This project is a successful Proof of Concept (PoC) for a real-time call center analytics platform developed for Pasha Bank. The system automatically ingests call recordings and performs a comprehensive, multi-stage analysis to generate actionable business insights. The pipeline includes speaker diarization, speaker role identification (customer vs. operator), transcription, emotion trend analysis, abstractive summarization, hierarchical topic modeling, and a final quality assessment. The results are streamed in real-time to a live web dashboard, providing a complete and intuitive overview of each call.

---

## 🚀 Features & Pipeline

This project performs an end-to-end audio analysis through a sophisticated pipeline of machine learning models. The output of one stage serves as the input for the next, creating a comprehensive analytical view of each call.

- 📂 **File Monitoring**: The `watchdog` library monitors a directory for new audio files (`.wav`, `.mp3`), automatically triggering the analysis pipeline.

- 🎙️ **Speech-to-Text & Diarization**:
    - **Transcription**: A fine-tuned `whisper-az-small-finetuned` model provides highly accurate Azerbaijani speech-to-text.
    - **Diarization**: `pyannote/speaker-diarization-3.1` segments the audio to determine *who* spoke and *when*.

- 🧠 **Speaker Role Identification**:
    - A custom BERT-based model (`speakar-idenfication`) analyzes the dialogue to label generic speakers as **"customer"** or **"operator"**.

- 😊 **Emotion Trend Analysis**:
    - The `superb/wav2vec2-large-superb-er` model classifies the emotion of each dialogue segment (Happy, Neutral, Sad, Angry).
    - This generates a summary of the customer's emotional journey throughout the call (e.g., "Customer started Neutral and finished Happy").

- ✍️ **Abstractive Summarization**:
    - The `mt5-summarize-callcenter-az-final` model generates a concise, descriptive summary of the entire call's content.

- 🏷️ **Hierarchical Topic Modeling**:
    - A two-step classification process using `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` first identifies a general topic and then a more specific sub-topic from a predefined `topics.csv`.

- 📊 **Quality Assessment**:
    - The `nlptown/bert-base-multilingual-uncased-sentiment` model analyzes the generated summary to rate the operator's service quality.
    - The output includes a "Good" or "Poor" label and a 1-5 star rating for a more granular, visual metric.

- 🖥️ **Real-time UI Dashboard**:
    - A web interface built with **FastAPI** and **WebSockets** provides a live view of the processing pipeline.
    - For each call, the dashboard displays the file details, an interactive audio player, the status of each analysis stage, and collapsible sections for detailed results.

---

## 🗂️ Project Structure

```
├── main.py                         # FastAPI app, WebSocket manager, and file watcher
├── models_upd.py                   # Core CallProcessor class managing the entire ML pipeline
├── speakar-idenfication/           # Speaker role classification BERT model
├── mt5-summarize-callcenter-az-final/  # Abstractive summarization model
├── records/1003/                   # Monitored folder for incoming audio files
├── topics.csv                      # Predefined topics for hierarchical classification
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

> Requires Python 3.8+

1.  **Clone this repo:**
    ```bash
    git clone <repository_url>
    cd CallCenter-Audio-Analysis-PoC
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **PyTorch Installation:**
    A GPU with CUDA is strongly recommended for performance.

    ⚡ **Recommended (GPU - CUDA 12.1+)**
    ```bash
    pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    ```
    *Use the official [PyTorch website](https://pytorch.org/get-started/locally/) to find the correct command for your specific CUDA version.*

    🐢 **Optional (CPU only)**
    ```bash
    pip install torch torchaudio
    ```
    ⚠️ **Warning**: Running on CPU is significantly slower and not suitable for real-time performance.

5.  **Install ffmpeg:**
    This is a crucial dependency for audio processing.
    -   **Windows**: Download the binaries from the [ffmpeg official website](https://ffmpeg.org/download.html) and add them to your system's PATH.
    -   **Linux/macOS**: Use a package manager (`sudo apt-get install ffmpeg` or `brew install ffmpeg`).

6.  **Hugging Face Credentials:**
    You will likely need a Hugging Face token to download certain models. It is recommended to log in via the CLI:
    ```bash
    huggingface-cli login
    ```
    Alternatively, you can set your token as an environment variable named `HF_TOKEN`.

---

## 🧪 Running the App

1.  Place your audio files (e.g., `.wav`, `.mp3`) into the `records/1003` directory.
2.  Run the main application from the project root:
    ```bash
    python main.py
    ```
3.  The application will start, and the console will display the following information:
    -   Watching folder: `...\records\1003`
    -   Server starts at: `http://localhost:8001`
    -   WebSocket endpoint: `ws://localhost:8001/ws`

4.  Open your web browser and navigate to **`http://localhost:8001`**. The real-time dashboard will open and wait for new audio files to be added to the monitored folder.

---

## 🧠 Models Used

| Stage | Model Name | Hugging Face / Source |
| :--- | :--- | :--- |
| **Transcription** | `whisper-az-small-finetuned` | Local fine-tuned model |
| **Diarization** | `speaker-diarization-3.1` | `pyannote/speaker-diarization-3.1` |
| **Speaker ID** | `speakar-idenfication` | Local custom BERT model |
| **Emotion Analysis** | `wav2vec2-large-superb-er`| `superb/wav2vec2-large-superb-er` |
| **Summarization** | `mt5-summarize-callcenter-az-final` | Local fine-tuned MT5 model |
| **Topic Modeling** | `mDeBERTa-v3-base-mnli-xnli`| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| **Quality Assessment**| `bert-base-multilingual-uncased-sentiment` | `nlptown/bert-base-multilingual-uncased-sentiment` |

---

## 📌 Technical Challenges & Key Findings

-   **Model Compatibility**: A significant challenge was a `TorchScript` bug that caused the `mDeBERTa-v3` topic model to crash on the GPU. This was resolved by forcing this specific model to run on the CPU, which stabilized the application with a minimal performance impact.
-   **Topic Modeling Accuracy**: Initial topic classification results were poor. Accuracy was drastically improved by implementing a hierarchical (Class -> Sub-class) approach and discovering that the model performed best with an English hypothesis template ("This text is about {}") despite the Azerbaijani input.
-   **Quality Assessment Model**: An initial custom KNN classifier proved unreliable. It was replaced with the `nlptown` multilingual sentiment model, which delivered far superior and more consistent accuracy in assessing service quality based on the call summary.
-   **Real-time Architecture**: The system uses a thread-safe `queue.Queue` to manage communication between the background file processing thread and the main FastAPI WebSocket event loop. This ensures a non-blocking, stable, and responsive real-time experience for all connected clients.

---

## 👤 Author

This Proof of Concept was developed by **Gasym A. Valiyev** for **Pasha Bank**.