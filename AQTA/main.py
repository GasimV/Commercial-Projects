# --- START OF FILE main.py ---

import torch
import torch.nn.functional as F
from typing import Tuple, List
from io import BytesIO
from pydub import AudioSegment
from models_upd import CallProcessor
import os
import time
import json
import asyncio
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import logging
import queue
import concurrent.futures
from fastapi.staticfiles import StaticFiles
import shutil

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Real-time Audio Monitor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WATCH_FOLDER = r"C:\Pasha-PoC\records\1003"
os.makedirs(WATCH_FOLDER, exist_ok=True)
DEBUG_FOLDER = "./debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)
STATIC_FOLDER = "./static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
app.mount("/audio", StaticFiles(directory=WATCH_FOLDER), name="audio")
app.mount("/denoised", StaticFiles(directory=DEBUG_FOLDER), name="denoised")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = CallProcessor(
    device=device
)

# Store active WebSocket connections and message queue
active_connections: List[WebSocket] = []
message_queue = queue.Queue()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_message(self, message: dict):
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket: {e}")
                    disconnected.append(connection)
            for conn in disconnected:
                self.disconnect(conn)

    def queue_message(self, message: dict):
        message_queue.put(message)


manager = ConnectionManager()


async def process_message_queue():
    while True:
        try:
            try:
                message = message_queue.get_nowait()
                await manager.send_message(message)
            except queue.Empty:
                pass
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing message queue: {e}")
            await asyncio.sleep(1)


class NewFileHandler(FileSystemEventHandler):
    def process_file(self, filepath: str):
        if not filepath.endswith((".wav", ".mp3", ".flac")):
            return
        print(f"[INFO] Processing triggered for: {filepath}")
        manager.queue_message({
            "type": "file_detected",
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "timestamp": datetime.now().isoformat()
        })
        try:
            manager.queue_message({
                "type": "processing_started",
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "timestamp": datetime.now().isoformat()
            })
            from pydub import AudioSegment
            try:
                seg = AudioSegment.from_file(filepath)  # auto-detect
                seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                seg.export(filepath, format="wav")  # 16-kHz mono PCM WAV
            except Exception as e:
                logger.error(f"Normalize/re-encode failed: {e}")
            for output in processor.process_call(filepath):
                print("*" * 100)
                print(f"Stage: {output['stage']}")
                result_data = output['result']
                if isinstance(result_data, list):
                    print("\n".join(map(str, result_data)))
                else:
                    print(result_data)
                manager.queue_message({
                    "type": "processing_update",
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "stage": output['stage'],
                    "result": output['result'],
                    "timestamp": datetime.now().isoformat()
                })
            manager.queue_message({
                "type": "processing_completed",
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[ERROR] Failed to process {filepath}: {e}")
            manager.queue_message({
                "type": "processing_error",
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    def on_created(self, event):
        pass


@app.post("/upload-audio")
async def upload_live_audio(file: UploadFile):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Always re-encode to WAV for the pipeline
        filename = f"live_recording_{timestamp}.wav"
        save_path = os.path.join(MOVE_SOURCE_FOLDER, filename)

        contents = await file.read()

        # Detect input format
        content_type = (file.content_type or "").lower()
        inferred_fmt = None
        if "webm" in content_type:
            inferred_fmt = "webm"
        elif "ogg" in content_type or "opus" in content_type:
            inferred_fmt = "ogg"
        elif "wav" in content_type:
            inferred_fmt = "wav"

        # Fallback: try probing without format; if it fails, try webm
        try:
            if inferred_fmt:
                audio_segment = AudioSegment.from_file(BytesIO(contents), format=inferred_fmt)
            else:
                audio_segment = AudioSegment.from_file(BytesIO(contents))
        except Exception:
            # last resort: attempt webm
            audio_segment = AudioSegment.from_file(BytesIO(contents), format="webm")

        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_segment.export(save_path, format="wav")

        logger.info(f"Live recording saved to source folder: {save_path}. It will now be moved and processed.")
        return {"status": "success", "filename": filename}
    except Exception as e:
        logger.error(f"Failed to save live recording to source folder: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_message_queue())
    logger.info("Message queue processor started")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": f"Connected to folder monitor: {WATCH_FOLDER}",
            "timestamp": datetime.now().isoformat()
        }))
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.get("/")
async def get():
    return HTMLResponse(r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Processing Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }

        .header-logo {
            height: 80px;
            margin-bottom: 1rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }

        .recording-panel {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;
        }

        .recording-panel h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: #1e293b;
        }

        .recording-controls button {
            font-size: 1rem;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 0 0.5rem;
        }

        #recordButton {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        }
        #recordButton:hover {
             transform: scale(1.05);
        }
        #recordButton:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        #stopButton {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        #stopButton:hover {
            transform: scale(1.05);
        }
        #stopButton:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        #recordingStatus {
            margin-top: 1rem;
            font-weight: 500;
            color: #475569;
        }

        .status-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-weight: 600;
            font-size: 1.1rem;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .status-dot.connected {
            background: #10b981;
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.3);
        }

        .status-dot.disconnected {
            background: #ef4444;
            box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.3);
        }

        .folder-path {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0,0,0,0.05);
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-size: 0.9rem;
        }

        .processing-grid {
            display: grid;
            gap: 2rem;
            flex: 1;
        }

        .file-card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            transform: translateY(10px);
            opacity: 0;
            animation: slideIn 0.5s ease forwards;
            margin-bottom: 1rem;
        }

        @keyframes slideIn {
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        .file-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #f1f5f9;
        }

        .file-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
        }

        .file-info {
            flex: 1;
        }

        .file-info h3 {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.25rem;
        }

        .file-info .timestamp {
            color: #64748b;
            font-size: 0.9rem;
        }

        .audio-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-left: auto;
        }

        .play-button {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            padding-top: 4px;
        }

        .play-button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }

        .play-button.denoised {
            background: linear-gradient(135deg, #fb923c, #f97316);
            box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
        }
        .play-button.denoised:hover {
             transform: scale(1.05);
             box-shadow: 0 6px 20px rgba(249, 115, 22, 0.4);
        }

        .play-button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .play-button .play-icon {
            font-size: 1.2rem;
            line-height: 1;
        }
        .play-button .play-text {
            font-size: 0.7rem;
            font-weight: 700;
            margin-top: 2px;
            line-height: 1;
            text-transform: uppercase;
        }

        .play-button .play-icon::before {
            content: '▶️';
        }
        .play-button.playing .play-icon::before {
            content: '⏸️';
        }

        .audio-player {
            display: none;
            margin-top: 1rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background: rgba(0,0,0,0.05);
            border-radius: 12px;
        }

        .audio-player audio {
            width: 100%;
        }

        .steps-container {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .step {
            text-align: center;
            padding: 1.5rem 1rem;
            border-radius: 16px;
            transition: all 0.3s ease;
            position: relative;
            border: 2px solid transparent;
        }

        .step::after {
            content: '';
            position: absolute;
            right: -0.5rem;
            top: 50%;
            transform: translateY(-50%);
            width: 0;
            height: 0;
            border-left: 8px solid;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
            z-index: 1;
        }

        .step:last-child::after {
            display: none;
        }

        .step.pending {
            background: #f8fafc;
            color: #64748b;
            border-color: #e2e8f0;
        }

        .step.pending::after {
            border-left-color: #e2e8f0;
        }

        .step.processing {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
            border-color: #3b82f6;
        }

        .step.processing::after {
            border-left-color: #3b82f6;
        }

        .step.processing .step-icon {
            animation: pulse 2s infinite;
        }

        .step.completed {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border-color: #10b981;
        }

        .step.completed::after {
            border-left-color: #10b981;
        }

        .step.error {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            border-color: #ef4444;
        }

        .step.error::after {
            border-left-color: #ef4444;
        }

        .step-icon {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            display: block;
        }

        .step-title {
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }

        .step-subtitle {
            font-size: 0.75rem;
            opacity: 0.8;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }

        .results-section {
            background: #f8fafc;
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1rem;
            border: 1px solid #e2e8f0;
        }

        .result-item {
            background: white;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }

        .result-item:last-child {
            margin-bottom: 0;
        }

        .result-item.classification {
            border-left: 4px solid #8b5cf6;
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.05), rgba(139, 92, 246, 0.02));
        }

        .result-item.classification .result-title {
            background: rgba(139, 92, 246, 0.1);
            color: #6b21a8;
        }

        .star-rating {
            font-size: 1.5rem;
            color: #f59e0b;
            margin-bottom: 0.5rem;
        }
        .star-rating .empty {
            color: #d1d5db;
        }

        .result-title {
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .result-content {
            color: #475569;
            line-height: 1.6;
            font-size: 0.9rem;
            max-height: 350px;
            overflow-y: auto;
            padding-right: 0.5rem;
        }

        .classification-result {
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            text-align: center;
            margin-top: 0.5rem;
        }

        .classification-good {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
        }

        .classification-bad {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
        }

        .collapsible-header {
            cursor: pointer;
            user-select: none;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .collapsible-header:hover {
            color: #3b82f6;
        }

        .toggle-icon {
            font-size: 0.8rem;
            transition: transform 0.3s ease;
            color: #64748b;
        }

        .toggle-icon.expanded {
            transform: rotate(180deg);
        }

        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .collapsible-content.expanded {
            max-height: 1000px;
        }

        .collapsible-item {
            border-left: 4px solid #f59e0b;
        }

        .collapsible-item .result-title {
            background: rgba(245, 158, 11, 0.1);
            margin: -1rem -1rem 0.5rem -1rem;
            padding: 1rem;
            border-radius: 8px 8px 0 0;
        }

        @media (max-width: 1200px) {
            .steps-container {
                grid-template-columns: 1fr;
                gap: 0.5rem;
            }

            .step {
                display: flex;
                align-items: center;
                text-align: left;
                padding: 1rem 1.5rem;
            }

            .step::after {
                display: none;
            }

            .step-icon {
                margin-right: 1rem;
                margin-bottom: 0;
            }

            .file-header {
                flex-direction: column;
                align-items: flex-start;
            }

            .audio-controls {
                margin-left: 0;
                margin-top: 1rem;
                width: 100%;
                justify-content: flex-start;
            }
        }

        .no-files {
            text-align: center;
            padding: 4rem 2rem;
            color: rgba(255,255,255,0.8);
        }

        .no-files h2 {
            font-size: 2rem;
            margin-bottom: 1rem;
            opacity: 0.7;
        }

        .no-files p {
            font-size: 1.1rem;
            opacity: 0.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="/static/aqta_logo.svg" alt="AQTA Logo" class="header-logo">
            <h1>CALL CENTER AUDIO ANALYSIS</h1>
        </div>

        <div class="recording-panel">
            <h2>Live Recording Demo</h2>
            <div class="recording-controls">
                <button id="recordButton">🔴 Record</button>
                <button id="stopButton" disabled>⏹️ Stop</button>
            </div>
            <div id="recordingStatus">Click 'Record' to start</div>
            <!-- ==== [ADD START] Realtime STS panel ==== -->
            <div class="recording-panel" style="margin-top:1rem">
              <h2>Realtime (Mic → Speech-to-Speech)</h2>
              <div class="recording-controls">
                <button id="rtStartBtn">🎤 Start Realtime</button>
                <button id="rtStopBtn" disabled>⏹️ Stop</button>
              </div>
              <div id="rtStatus">Idle</div>
              <audio id="rtRemoteAudio" autoplay playsinline></audio>
            </div>
            <!-- ==== [ADD END] Realtime STS panel ==== -->
        </div>

        <div class="status-bar">
            <div class="connection-status">
                <div id="status-dot" class="status-dot disconnected"></div>
                <span id="status-text">Disconnected</span>
            </div>
            <div class="folder-path">Monitoring audio files...</div>
        </div>

        <div class="processing-grid" id="processing-grid">
            <div class="no-files">
                <h2>Waiting for new call...</h2>
                <p>When new call happens, you will see real-time processing</p>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket("ws://localhost:8001/ws");
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const processingGrid = document.getElementById('processing-grid');

        let currentFiles = {};

        // <<< MODIFICATION: The UI steps are now a cleaner 6-step process >>>
        const stepConfig = [
            { key: 'file_detected', title: 'NEW FILE', subtitle: 'DETECTED', icon: '📥' },
            { key: 'identified_dialogue', title: 'TRANSCRIPTION', subtitle: 'DIARIZATION + STT', icon: '🎙️' },
            { key: 'emotion_analysis', title: 'EMOTION TREND', subtitle: 'CUSTOMER ANALYSIS', icon: '😊' },
            { key: 'summary', title: 'SUMMARY', subtitle: 'GENERATION', icon: '📝' },
            { key: 'topic_classification', title: 'TOPIC', subtitle: 'CLASSIFICATION', icon: '🏷️' },
            { key: 'classification', title: 'QUALITY', subtitle: 'ASSESSMENT', icon: '⭐' }
        ];

        ws.onopen = function(event) {
            statusText.textContent = 'Connected';
            statusDot.className = 'status-dot connected';
        };

        ws.onclose = function(event) {
            statusText.textContent = 'Disconnected';
            statusDot.className = 'status-dot disconnected';
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            handleMessage(data);
        };

        const recordButton = document.getElementById('recordButton');
        const stopButton = document.getElementById('stopButton');
        const recordingStatus = document.getElementById('recordingStatus');

        let mediaRecorder;
        let audioChunks = [];

        recordButton.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('file', audioBlob, 'live_recording.wav');

                    recordingStatus.textContent = 'Uploading and processing...';

                    fetch('/upload-audio', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        console.log('Upload successful:', data);
                        recordingStatus.textContent = `File ${data.filename} uploaded. Processing will start shortly.`;
                        setTimeout(() => {
                           recordingStatus.textContent = "Click 'Record' to start";
                        }, 5000);
                    })
                    .catch(error => {
                        console.error('Upload failed:', error);
                        recordingStatus.textContent = 'Upload failed. Please try again.';
                    });

                    audioChunks = [];
                };

                mediaRecorder.start();
                recordButton.disabled = true;
                stopButton.disabled = false;
                recordingStatus.textContent = 'Recording... Speak into the microphone.';

            } catch (err) {
                console.error("Error accessing microphone:", err);
                recordingStatus.textContent = 'Microphone access denied. Please allow microphone access in your browser settings.';
            }
        });

        stopButton.addEventListener('click', () => {
            mediaRecorder.stop();
            recordButton.disabled = false;
            stopButton.disabled = true;
            recordingStatus.textContent = 'Recording stopped. Preparing for upload...';
        });

        function handleMessage(data) {
            if (data.type === 'heartbeat' || data.type === 'connection_established') return;

            const filename = data.filename;

            if (data.type === 'file_detected') {
                createFileCard(filename, data.timestamp);
                updateStep(filename, 'file_detected', 'completed');
                // <<< MODIFICATION: Directly set the final dialogue step to processing >>>
                updateStep(filename, 'identified_dialogue', 'processing');
            } else if (data.type === 'processing_started') {
                // This state is less important now but can be kept
            } else if (data.type === 'processing_update') {
                handleProcessingUpdate(filename, data);
            } else if (data.type === 'processing_completed') {
                // All processing completed
            } else if (data.type === 'processing_error') {
                markAllStepsError(filename);
                addResult(filename, 'error', `Error: ${data.error}`);
            }
        }

        function createFileCard(filename, timestamp) {
            const noFiles = document.querySelector('.no-files');
            if (noFiles) noFiles.remove();

            const card = document.createElement('div');
            card.className = 'file-card';
            card.id = `file-${filename}`;

            card.innerHTML = `
                <div class="file-header">
                    <div class="file-icon">🎵</div>
                    <div class="file-info">
                        <h3>${filename}</h3>
                        <div class="timestamp">${new Date(timestamp).toLocaleString()}</div>
                    </div>
                    <div class="audio-controls" id="audio-controls-${filename}">
                         <button class="play-button" onclick="toggleAudio('${filename}', 'original')" id="play-btn-original-${filename}">
                            <span class="play-icon"></span>
                            <span class="play-text">RAW</span>
                         </button>
                    </div>
                </div>
                <div class="audio-player" id="audio-player-original-${filename}">
                    <audio controls id="audio-original-${filename}">
                        <source src="/audio/${filename}" type="audio/mpeg">
                        <source src="/audio/${filename}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                </div>
                <div class="steps-container">
                    ${stepConfig.map(step => `
                        <div class="step pending" data-step="${step.key}">
                            <span class="step-icon">${step.icon}</span>
                            <div class="step-title">${step.title}</div>
                            <div class="step-subtitle">${step.subtitle}</div>
                        </div>
                    `).join('')}
                </div>
                <div class="results-section" id="results-${filename}" style="display: none;"></div>
            `;

            processingGrid.prepend(card);
            currentFiles[filename] = { card, results: [] };
        }

        function addDenoisedPlayer(filename, denoisedFilename) {
            const card = document.getElementById(`file-${filename}`);
            if (!card) return;

            const audioControls = card.querySelector(`#audio-controls-${filename}`);

            const denoisedButton = document.createElement('button');
            denoisedButton.className = 'play-button denoised';
            denoisedButton.id = `play-btn-denoised-${filename}`;
            denoisedButton.onclick = () => toggleAudio(filename, 'denoised');
            denoisedButton.innerHTML = `
                <span class="play-icon"></span>
                <span class="play-text">CLEAN</span>
            `;
            audioControls.appendChild(denoisedButton);

            const originalPlayer = document.getElementById(`audio-player-original-${filename}`);
            const denoisedPlayerDiv = document.createElement('div');
            denoisedPlayerDiv.className = 'audio-player';
            denoisedPlayerDiv.id = `audio-player-denoised-${filename}`;
            denoisedPlayerDiv.innerHTML = `
                <audio controls id="audio-denoised-${filename}">
                    <source src="/denoised/${denoisedFilename}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            `;
            originalPlayer.parentNode.insertBefore(denoisedPlayerDiv, originalPlayer.nextSibling);
        }

        function toggleAudio(filename, type) {
            const audioPlayer = document.getElementById(`audio-player-${type}-${filename}`);
            const audio = document.getElementById(`audio-${type}-${filename}`);
            const playBtn = document.getElementById(`play-btn-${type}-${filename}`);
            if (!audioPlayer || !audio || !playBtn) return;

            const otherType = type === 'original' ? 'denoised' : 'original';
            const otherAudio = document.getElementById(`audio-${otherType}-${filename}`);
            const otherPlayBtn = document.getElementById(`play-btn-${otherType}-${filename}`);
            if (otherAudio && otherPlayBtn && !otherAudio.paused) {
                otherAudio.pause();
                otherPlayBtn.classList.remove('playing');
            }

            const isPlaying = playBtn.classList.contains('playing');

            if (isPlaying) {
                audio.pause();
                playBtn.classList.remove('playing');
            } else {
                if (audioPlayer.style.display === 'none' || audioPlayer.style.display === '') {
                    audioPlayer.style.display = 'block';
                }
                audio.play();
                playBtn.classList.add('playing');
            }

            audio.onended = () => playBtn.classList.remove('playing');
        }

        function updateStep(filename, stepKey, status) {
            const fileData = currentFiles[filename];
            if (!fileData) return;

            const step = fileData.card.querySelector(`[data-step="${stepKey}"]`);
            if (step) {
                step.className = `step ${status}`;
            }
        }

        function handleProcessingUpdate(filename, data) {
            const stage = data.stage.toLowerCase();

            if (stage === 'denoised_audio_ready') {
                addDenoisedPlayer(filename, data.result.denoised_filename);
                return;
            }

            // <<< MODIFICATION: Simplified process flow >>>
            if (stage === 'identified_dialogue') {
                updateStep(filename, 'identified_dialogue', 'completed');
                updateStep(filename, 'emotion_analysis', 'processing');
            } else if (stage === 'emotion_analysis') {
                updateStep(filename, 'emotion_analysis', 'completed');
                updateStep(filename, 'summary', 'processing');
            } else if (stage === 'summary') {
                updateStep(filename, 'summary', 'completed');
                updateStep(filename, 'topic_classification', 'processing');
            } else if (stage === 'topic_classification') {
                updateStep(filename, 'topic_classification', 'completed');
                updateStep(filename, 'classification', 'processing');
            } else if (stage === 'classification') {
                updateStep(filename, 'classification', 'completed');
            }

            // <<< MODIFICATION: Only add result if it's not the old 'dialogue' stage >>>
            if (stage !== 'dialogue') {
                addResult(filename, stage, data.result);
            }
        }

        function addResult(filename, stage, result) {
            const fileData = currentFiles[filename];
            if (!fileData) return;

            const resultsSection = document.getElementById(`results-${filename}`);
            resultsSection.style.display = 'block';

            const resultItem = document.createElement('div');
            const isCollapsible = stage.toLowerCase().includes('dialogue');
            const isClassification = stage.toLowerCase() === 'classification' && typeof result === 'object';
            const isEmotionAnalysis = stage.toLowerCase() === 'emotion_analysis';
            const isTopicClassification = stage.toLowerCase() === 'topic_classification';

            if (isCollapsible) {
                resultItem.className = 'result-item collapsible-item';
                const resultContent = Array.isArray(result) ? result.map(line => {
                    let coloredLine = line;
                    coloredLine = coloredLine.replace(/\(Happy\)/g, '<span>(😊 Happy)</span>');
                    coloredLine = coloredLine.replace(/\(Neutral\)/g, '<span>(😐 Neutral)</span>');
                    coloredLine = coloredLine.replace(/\(Sad\)/g, '<span>(😢 Sad)</span>');
                    coloredLine = coloredLine.replace(/\(Angry\)/g, '<span>(😠 Angry)</span>');
                    return coloredLine;
                }).join('<br>') : result;
                const collapsibleId = `collapsible-${filename}-${Date.now()}`;

                resultItem.innerHTML = `
                    <div class="result-title">
                        <div class="collapsible-header" onclick="toggleCollapsible('${collapsibleId}')">
                            <!-- <<< MODIFICATION: Title is now just TRANSCRIPT -->
                            <span>TRANSCRIPT</span>
                            <span class="toggle-icon">▼</span>
                        </div>
                    </div>
                    <div class="collapsible-content" id="${collapsibleId}">
                        <div class="result-content">${resultContent}</div>
                    </div>
                `;
            } else if (isClassification) {
                resultItem.className = 'result-item classification';
                const { label, score } = result;

                const isGoodService = score >= 4;
                const classificationClass = isGoodService ? 'classification-good' : 'classification-bad';

                const renderStars = (rating) => {
                    let stars = '';
                    for (let i = 1; i <= 5; i++) {
                        stars += i <= rating ? '★' : '<span class="empty">☆</span>';
                    }
                    return stars;
                };

                resultItem.innerHTML = `
                    <div class="result-title">QUALITY ASSESSMENT</div>
                    <div class="result-content" style="text-align: center;">
                        <div class="star-rating">${renderStars(score)}</div>
                        <div class="classification-result ${classificationClass}">
                            ${label}
                        </div>
                    </div>
                `;
            } else if (isEmotionAnalysis) {
                resultItem.className = 'result-item';
                resultItem.style.borderLeft = '4px solid #f97316';
                resultItem.innerHTML = `
                    <div class="result-title" style="color: #c2410c;">EMOTION TREND ANALYSIS</div>
                    <div class="result-content" style="font-weight: 500; font-size: 1.05rem;">${result}</div>
                `;
            } else if (isTopicClassification) {
                resultItem.className = 'result-item';
                resultItem.style.borderLeft = '4px solid #d946ef';
                resultItem.innerHTML = `
                    <div class="result-title" style="color: #86198f;">TOPIC CLASSIFICATION</div>
                    <div class="result-content" style="font-weight: 500; font-size: 1.05rem;">${result}</div>
                `;
            }
            else {
                resultItem.className = 'result-item';
                const resultContent = Array.isArray(result) ? result.join('<br>') : result;

                resultItem.innerHTML = `
                    <div class="result-title">${stage.toUpperCase()}</div>
                    <div class="result-content">${resultContent}</div>
                `;
            }

            resultsSection.appendChild(resultItem);
        }

        function toggleCollapsible(id) {
            const content = document.getElementById(id);
            const icon = content.previousElementSibling.querySelector('.toggle-icon');

            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                icon.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                icon.classList.add('expanded');
            }
        }

        function markAllStepsError(filename) {
            const fileData = currentFiles[filename];
            if (!fileData) return;

            const steps = fileData.card.querySelectorAll('.step');
            steps.forEach(step => {
                if (!step.classList.contains('completed')) {
                    step.className = 'step error';
                }
            });
        }

        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);

        /* ==== Realtime STS WebRTC (UNIFIED) ==== */
        const rtStartBtn = document.getElementById('rtStartBtn');
        const rtStopBtn  = document.getElementById('rtStopBtn');
        const rtStatus   = document.getElementById('rtStatus');
        const rtRemoteAudio = document.getElementById('rtRemoteAudio');
        
        let rtcPeer = null;
        let micStream = null;
        let botStream = null;      // remote audio
        let dataChannel = null;
        
        // mixed recorder (user + assistant)
        let rtMixedRecorder = null;
        let rtMixedChunks = [];
        let audioCtx = null;
        let mixDest = null;
        let recorderStarted = false;
        
        // No-op stubs because the transcript UI was removed
        function appendTranscript(){ /* noop */ }
        function updatePartial(){ /* noop */ }
        function finalizePartial(){ /* noop */ }
        
        function requestTextAndAudio() {
          if (!dataChannel || dataChannel.readyState !== "open") return;
          dataChannel.send(JSON.stringify({
            type: "response.create",
            response: { modalities: ["audio","text"], conversation: "default" }
          }));
        }
        
        function wireEventsChannel(dc) {
          dataChannel = dc;
        
          dc.onopen = () => {
            console.log("oai-events open");
            requestTextAndAudio();              // first turn: ask for audio+text
          };
        
          dc.onmessage = (e) => {
            // console.log("oai-events:", e.data);
            let evt; try { evt = JSON.parse(e.data); } catch { return; }
        
            // USER transcript (your speech)
            if (evt.type === "response.input_text.delta" && typeof evt.delta === "string") {
              updatePartial("user", evt.delta);
            }
            if (evt.type === "response.input_text.done") {
              finalizePartial("user");
            }
        
            // ASSISTANT transcript
            if (evt.type === "response.output_text.delta" && typeof evt.delta === "string") {
              updatePartial("assistant", evt.delta);
            }
            if (evt.type === "response.output_text.done") {
              finalizePartial("assistant");
              requestTextAndAudio();            // keep getting text next turns
            }
        
            // Optional speaking markers
            if (evt.type === "input_audio_buffer.speech_started") {
              appendTranscript("user", "(speaking…)", true);
            }
            if (evt.type === "input_audio_buffer.speech_stopped") {
              finalizePartial("user");
            }
          };
        }
        
        // Wait for ICE to finish so our SDP has candidates
        function waitForIceGathering(pc) {
          if (pc.iceGatheringState === 'complete') return Promise.resolve();
          return new Promise((resolve) => {
            function check() {
              if (pc.iceGatheringState === 'complete') {
                pc.removeEventListener('icegatheringstatechange', check);
                resolve();
              }
            }
            pc.addEventListener('icegatheringstatechange', check);
          });
        }
        
        async function startRealtime() {
          try {
            rtStatus.textContent = 'Requesting microphone...';
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
            rtStatus.textContent = 'Creating OpenAI session...';
            const sessRes = await fetch('/realtime/session', { method: 'POST' });
            if (!sessRes.ok) throw new Error('Failed to create ephemeral session');
            const sess = await sessRes.json();
            const EPHEMERAL_KEY = sess?.client_secret?.value;
            if (!EPHEMERAL_KEY) throw new Error('No ephemeral key in response');
        
            // Peer connection
            rtcPeer = new RTCPeerConnection();
            micStream.getTracks().forEach(t => rtcPeer.addTrack(t, micStream));
        
            // Remote audio track
            rtcPeer.ontrack = (evt) => {
              [botStream] = evt.streams;
              rtRemoteAudio.srcObject = botStream;
        
              // If we haven't started the mixed recorder yet, do it now (we have both ends).
              if (!recorderStarted) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                mixDest = audioCtx.createMediaStreamDestination();
        
                const micSource = audioCtx.createMediaStreamSource(micStream);
                micSource.connect(mixDest);
        
                const botSource = audioCtx.createMediaStreamSource(botStream);
                botSource.connect(mixDest);
        
                rtMixedChunks = [];
                rtMixedRecorder = new MediaRecorder(mixDest.stream, { mimeType: 'audio/webm' });
                rtMixedRecorder.ondataavailable = (ev) => { if (ev.data.size > 0) rtMixedChunks.push(ev.data); };
                rtMixedRecorder.start();
                recorderStarted = true;
              }
            };
        
            // Events channel
            rtcPeer.ondatachannel = (e) => {
              if (e.channel.label === "oai-events") wireEventsChannel(e.channel);
            };
            // Open proactively too
            wireEventsChannel(rtcPeer.createDataChannel("oai-events"));
        
            // Offer/Answer
            rtStatus.textContent = 'Negotiating session...';
            const offer = await rtcPeer.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: false });
            await rtcPeer.setLocalDescription(offer);
            await waitForIceGathering(rtcPeer);
        
            const baseUrl = 'https://api.openai.com/v1/realtime';
            const model = encodeURIComponent('gpt-realtime');
            const sdpRes = await fetch(`${baseUrl}?model=${model}`, {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${EPHEMERAL_KEY}`,
                'Content-Type': 'application/sdp',
                'OpenAI-Beta': 'realtime=v1'
              },
              body: rtcPeer.localDescription.sdp
            });
            if (!sdpRes.ok) throw new Error('Realtime SDP exchange failed');
            const answerSdp = await sdpRes.text();
            await rtcPeer.setRemoteDescription({ type: 'answer', sdp: answerSdp });
        
            rtStatus.textContent = 'Realtime connected. Speak now 🎙️';
            rtStartBtn.disabled = true;
            rtStopBtn.disabled  = false;
          } catch (err) {
            console.error(err);
            rtStatus.textContent = `Error: ${err.message}`;
            cleanupRealtime();
          }
        }
        
        async function stopRealtime() {
          rtStatus.textContent = 'Stopping...';
        
          // Stop the mixed recorder and get a Blob
          let mixedBlob = null;
          try {
            if (rtMixedRecorder && rtMixedRecorder.state !== 'inactive') {
              mixedBlob = await new Promise((resolve) => {
                rtMixedRecorder.onstop = () => resolve(new Blob(rtMixedChunks, { type: 'audio/webm' }));
                rtMixedRecorder.stop();
              });
            }
          } catch {}
        
          cleanupRealtime();
        
          // Upload the MIX of user+assistant to reuse your existing analysis pipeline
          if (mixedBlob) {
            const formData = new FormData();
            formData.append('file', mixedBlob, `realtime_mixed_${Date.now()}.webm`);
            rtStatus.textContent = 'Uploading conversation for analysis...';
            try {
              const r = await fetch('/upload-audio', { method: 'POST', body: formData });
              const data = await r.json();
              rtStatus.textContent = `Uploaded: ${data.filename}. Processing…`;
            } catch (e) {
              rtStatus.textContent = 'Upload failed. (Analysis skipped)';
            }
          }
        
          rtStartBtn.disabled = false;
          rtStopBtn.disabled  = true;
          rtStatus.textContent = 'Idle';
        }
        
        function cleanupRealtime() {
          try { if (dataChannel) dataChannel.close(); } catch {}
          dataChannel = null;
          try { if (rtcPeer) rtcPeer.close(); } catch {}
          rtcPeer = null;
        
          if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
            micStream = null;
          }
          botStream = null;
        
          if (audioCtx) {
            try { audioCtx.close(); } catch {}
            audioCtx = null; mixDest = null;
          }
          recorderStarted = false;
          rtRemoteAudio.srcObject = null;
        }
        
        rtStartBtn.addEventListener('click', startRealtime);
        rtStopBtn.addEventListener('click', stopRealtime);
        /* ==== /Realtime STS WebRTC (UNIFIED) ==== */

    </script>
</body>
</html>
    """)


@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "watch_folder": WATCH_FOLDER,
        "device": str(device),
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }


MOVE_SOURCE_FOLDER = r"Z:\1002"
os.makedirs(MOVE_SOURCE_FOLDER, exist_ok=True)


class MoveFileHandler(FileSystemEventHandler):
    def __init__(self, dest_folder: str, process_handler: NewFileHandler):
        self.dest_folder = dest_folder
        self.process_handler = process_handler

    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.wav', '.mp3', '.flac')):
            return
        source_path = event.src_path
        logger.info(f"[Move Watcher] Detected new file in source directory: {source_path}")
        time.sleep(2)
        try:
            filename = os.path.basename(source_path)
            dest_path = os.path.join(self.dest_folder, filename)
            logger.info(f"[Move Watcher] Moving '{source_path}' to '{dest_path}'")
            shutil.move(source_path, dest_path)
            logger.info(f"[Move Watcher] File move complete.")
            logger.info(f"[Move Watcher] Triggering AI processing for '{dest_path}'...")
            self.process_handler.process_file(dest_path)
        except Exception as e:
            logger.error(f"[Move Watcher] Failed to move or process file {source_path}. Reason: {e}")


def start_file_watcher():
    process_handler = NewFileHandler()
    move_handler = MoveFileHandler(
        dest_folder=WATCH_FOLDER,
        process_handler=process_handler
    )
    move_observer = Observer()
    move_observer.schedule(move_handler, path=MOVE_SOURCE_FOLDER, recursive=False)
    move_observer.start()
    logger.info(f"[File Watcher] Now monitoring source folder: {MOVE_SOURCE_FOLDER}")
    logger.info(f"[File Watcher] New files will be moved to '{WATCH_FOLDER}' and then processed.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received, stopping file watcher.")
        move_observer.stop()
    move_observer.join()
    logger.info("File watcher stopped.")

# ==== [ADD START] Realtime (OpenAI) STS branch ====
import requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # set in .env
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "verse")  # any supported voice

@app.post("/realtime/session")
def create_realtime_session():
    """
    Returns a short-lived ephemeral token for the browser to open
    a WebRTC session directly with OpenAI's Realtime API.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")

    try:
        r = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1",
            },
            json={
                "model": OPENAI_REALTIME_MODEL,
                "voice": OPENAI_VOICE,
                "modalities": ["audio", "text"],
                "instructions": "Sən Azərbaycan Qida Təhlükəsizliyi Agentliyi (AQTA) üçün səsli çağrı mərkəzi operatorusan. Nəzakətli, ardıcıl və aydın danış; cavabları qısa ver. Məlumat çatışmırsa, aydınlaşdırıcı sual ver. Hüquqi və tibbi məsləhət vermə; vətəndaşı rəsmi mənbələrə yönləndir: aqta.gov.az və 1003 Çağrı Mərkəzi.",
                "turn_detection": {  # <— ADD THIS
                    "type": "server_vad",
                    "threshold": 0.5,  # tweak if needed
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                }
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        logger.error(f"Failed to create realtime session: {getattr(e.response, 'text', e)}")
        raise HTTPException(status_code=500, detail="Failed to create realtime session.")
# ==== [ADD END] Realtime (OpenAI) STS branch ====


if __name__ == "__main__":
    watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
    watcher_thread.start()
    print(f"[INFO] Starting server at http://localhost:8001")
    print(f"[INFO] WebSocket endpoint: ws://localhost:8001/ws")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
