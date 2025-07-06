import torch
import torch.nn.functional as F
from typing import Tuple, List
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor, pipeline
from io import BytesIO
from pydub import AudioSegment
from pyannote.audio import Pipeline
from models_upd import CallProcessor
import os
import time
import json
import asyncio
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import logging
import queue
import concurrent.futures
from fastapi.staticfiles import StaticFiles

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

# This tells FastAPI that any request starting with /audio should be served from the WATCH_FOLDER.
app.mount("/audio", StaticFiles(directory=r"records/1003"), name="audio")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# This block replaces the `sw_model = load_sw_model(...)` line.
print("Loading the fine-tuned Whisper model...")
ft_model_path = r"Preprocessing-and-STT-FT/whisper-az-small-finetuned"
ft_model = WhisperForConditionalGeneration.from_pretrained(ft_model_path).to(device)
ft_processor = WhisperProcessor.from_pretrained(ft_model_path)

# Ensure the generation config is set correctly
#ft_model.config.forced_decoder_ids = ft_processor.get_decoder_prompt_ids(language="azerbaijani", task="transcribe")

# Create a pipeline for the fine-tuned model WITH LONG-FORM TRANSCRIPTION ENABLED
print("Creating fine-tuned pipeline with long-form audio support...")

# Define generation arguments to prevent repetition and specify the task
generate_kwargs = {
    "language": "azerbaijani",
    "task": "transcribe",
    "no_repeat_ngram_size": 3 # This is the key to stopping "Ədədədə..."
}

fine_tuned_pipe = pipeline(
    "automatic-speech-recognition",
    model=ft_model,
    tokenizer=ft_processor.tokenizer,
    feature_extractor=ft_processor.feature_extractor,
    device=0 if device.type == "cuda" else -1, # Pass device index for cuda
    chunk_length_s=30,
    stride_length_s=6,
    generate_kwargs=generate_kwargs
)
print("Fine-tuned model loaded and pipeline created successfully.")

WATCH_FOLDER = r"records/1003"
os.makedirs(WATCH_FOLDER, exist_ok=True)

processor = CallProcessor(
    hf_token=os.getenv("HF_TOKEN"),
    whisper_model=fine_tuned_pipe,
    id_model_path="./speakar-idenfication",
    sum_model_path="./mt5-summarize-callcenter-az-final",
    device="cuda" if torch.cuda.is_available() else "cpu"
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
        """Send message to all connected clients"""
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket: {e}")
                    disconnected.append(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.disconnect(conn)

    def queue_message(self, message: dict):
        """Thread-safe method to queue messages"""
        message_queue.put(message)


manager = ConnectionManager()


# Background task to process message queue
async def process_message_queue():
    """Process messages from the queue and send to WebSocket clients"""
    while True:
        try:
            # Check for messages in queue (non-blocking)
            try:
                message = message_queue.get_nowait()
                await manager.send_message(message)
            except queue.Empty:
                pass

            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing message queue: {e}")
            await asyncio.sleep(1)


class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        filepath = event.src_path
        if filepath.endswith((".wav", ".mp3", ".flac")):
            print(f"[INFO] New file detected: {filepath}")

            # Send file detection notification (thread-safe)
            manager.queue_message({
                "type": "file_detected",
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "timestamp": datetime.now().isoformat()
            })

            time.sleep(1)  # wait 1 second before processing

            try:
                # Send processing start notification
                manager.queue_message({
                    "type": "processing_started",
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "timestamp": datetime.now().isoformat()
                })

                for output in processor.process_call(filepath):
                    print("*" * 100)
                    print(f"Stage: {output['stage']}")

                    # <<< --- FIX: CORRECTED PRINT LOGIC --- >>>
                    result_data = output['result']
                    if isinstance(result_data, list):
                        # If it's a list (like dialogue), join with newlines
                        print("\n".join(result_data))
                    else:
                        # If it's a single string, print it directly
                        print(result_data)

                    # Send real-time processing updates (thread-safe)
                    manager.queue_message({
                        "type": "processing_update",
                        "filepath": filepath,
                        "filename": os.path.basename(filepath),
                        "stage": output['stage'],
                        "result": output['result'],
                        "timestamp": datetime.now().isoformat()
                    })

                # Send completion notification
                manager.queue_message({
                    "type": "processing_completed",
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                print(f"[ERROR] Failed to process {filepath}: {e}")

                # Send error notification
                manager.queue_message({
                    "type": "processing_error",
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Start the message queue processor
    asyncio.create_task(process_message_queue())
    logger.info("Message queue processor started")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": f"Connected to folder monitor: {WATCH_FOLDER}",
            "timestamp": datetime.now().isoformat()
        }))

        # Keep connection alive
        while True:
            # Receive ping messages from client to keep connection alive
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
            except asyncio.TimeoutError:
                # Send heartbeat if no message received
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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

        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
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
            margin-bottom: 2rem;
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
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }

        .play-button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }

        .play-button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .audio-player {
            display: none;
            margin-top: 1rem;
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
            
            max-height: 350px;     /* Set a maximum height for the content area */
            overflow-y: auto;      /* Add a scrollbar only if content overflows */
            padding-right: 0.5rem; /* Add some padding so scrollbar doesn't cover text */
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
            <h1>POC: CALL CENTER SUMMARIZATION</h1>
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

        const stepConfig = [
            { key: 'file_detected', title: 'NEW FILE', subtitle: 'DETECTED', icon: '' },
            { key: 'dialogue', title: 'TRANSCRIPTION', subtitle: 'DIARIZATION + STT', icon: '' },
            { key: 'identified_dialogue', title: 'SPEAKER ID', subtitle: 'SPEAKER LABELING', icon: '' },
            { key: 'emotion_analysis', title: 'EMOTION TREND', subtitle: 'CUSTOMER ANALYSIS', icon: '' }, // New Step
            { key: 'summary', title: 'SUMMARY', subtitle: 'GENERATION', icon: '' },
            { key: 'classification', title: 'CLASSIFICATION', subtitle: 'QUALITY ASSESSMENT', icon: '' }
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

        function handleMessage(data) {
            if (data.type === 'heartbeat' || data.type === 'connection_established') return;

            const filename = data.filename;

            if (data.type === 'file_detected') {
                createFileCard(filename, data.timestamp);
                updateStep(filename, 'file_detected', 'completed');
                updateStep(filename, 'dialogue', 'processing');
            } else if (data.type === 'processing_started') {
                // File processing started - dialogue step already set to processing
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
            // Remove no-files message
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
                    <div class="audio-controls">
                        <button class="play-button" onclick="toggleAudio('${filename}')" id="play-btn-${filename}">
                            
                        </button>
                    </div>
                </div>
                <div class="audio-player" id="audio-player-${filename}">
                    <audio controls id="audio-${filename}">
                        <source src="/audio/${filename}" type="audio/mpeg">
                        <source src="/audio/${filename}" type="audio/wav">
                        <source src="/audio/${filename}" type="audio/mp3">
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

            processingGrid.appendChild(card);
            currentFiles[filename] = { card, results: [] };
        }

        function toggleAudio(filename) {
            const audioPlayer = document.getElementById(`audio-player-${filename}`);
            const audio = document.getElementById(`audio-${filename}`);
            const playBtn = document.getElementById(`play-btn-${filename}`);

            if (audioPlayer.style.display === 'none' || audioPlayer.style.display === '') {
                audioPlayer.style.display = 'block';
                playBtn.textContent = '';
                audio.play();
            } else {
                audioPlayer.style.display = 'none';
                playBtn.textContent = '';
                audio.pause();
            }

            // Update button based on audio state
            audio.addEventListener('play', () => playBtn.textContent = '');
            audio.addEventListener('pause', () => playBtn.textContent = '');
            audio.addEventListener('ended', () => playBtn.textContent = '');
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

            // Map stages to steps and update status
            if (stage === 'dialogue') {
                updateStep(filename, 'dialogue', 'completed');
                updateStep(filename, 'identified_dialogue', 'processing');
            } else if (stage === 'identified_dialogue') {
                updateStep(filename, 'identified_dialogue', 'completed');
                updateStep(filename, 'emotion_analysis', 'processing'); // Next step is emotion analysis
            } else if (stage === 'emotion_analysis') {
                updateStep(filename, 'emotion_analysis', 'completed');
                updateStep(filename, 'summary', 'processing'); // Next step is summary
            } else if (stage === 'summary') {
                updateStep(filename, 'summary', 'completed');
                updateStep(filename, 'classification', 'processing');
            } else if (stage === 'classification') {
                updateStep(filename, 'classification', 'completed');
            }

            addResult(filename, stage, data.result);
        }

        function addResult(filename, stage, result) {
            const fileData = currentFiles[filename];
            if (!fileData) return;

            const resultsSection = document.getElementById(`results-${filename}`);
            resultsSection.style.display = 'block';

            const resultItem = document.createElement('div');
            const isCollapsible = stage.toLowerCase().includes('dialogue');
            const isClassification = stage.toLowerCase() === 'classification';
            const isEmotionAnalysis = stage.toLowerCase() === 'emotion_analysis';
            
            if (isCollapsible) {
                resultItem.className = 'result-item collapsible-item';
                // Display emotion emojis next to the text
                const resultContent = Array.isArray(result) ? result.map(line => {
                    let coloredLine = line;
                    coloredLine = coloredLine.replace(/\(hap\)/g, '<span>(😊 Happy)</span>');
                    coloredLine = coloredLine.replace(/\(neu\)/g, '<span>(😐 Neutral)</span>');
                    coloredLine = coloredLine.replace(/\(sad\)/g, '<span>(😢 Sad)</span>');
                    coloredLine = coloredLine.replace(/\(ang\)/g, '<span>(😠 Angry)</span>');
                    return coloredLine;
                }).join('<br>') : result;
                const collapsibleId = `collapsible-${filename}-${Date.now()}`;

                resultItem.innerHTML = `
                    <div class="result-title">
                        <div class="collapsible-header" onclick="toggleCollapsible('${collapsibleId}')">
                            <span>${stage.toUpperCase()}</span>
                            <span class="toggle-icon">▼</span>
                        </div>
                    </div>
                    <div class="collapsible-content" id="${collapsibleId}">
                        <div class="result-content">${resultContent}</div>
                    </div>
                `;
            } else if (isClassification) {
                resultItem.className = 'result-item classification';
                const isGoodService = result.includes('yaxşı') || result.includes('good');
                const classificationClass = isGoodService ? 'classification-good' : 'classification-bad';
                
                resultItem.innerHTML = `
                    <div class="result-title">QUALITY CLASSIFICATION</div>
                    <div class="result-content">
                        <div class="classification-result ${classificationClass}">
                            ${result}
                        </div>
                    </div>
                `;
            } else if (isEmotionAnalysis) {
                // Specific styling for the emotion analysis card
                resultItem.className = 'result-item';
                resultItem.style.borderLeft = '4px solid #f97316'; // Distinct orange color

                resultItem.innerHTML = `
                    <div class="result-title" style="color: #c2410c;">EMOTION TREND ANALYSIS</div>
                    <div class="result-content" style="font-weight: 500; font-size: 1.05rem;">${result}</div>
                `;
            }
            else { // This will now handle the 'summary' stage
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

        // Send ping every 25 seconds to keep connection alive
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 25000);
    </script>
</body>
</html>
    """)


@app.get("/status")
async def get_status():
    """Get current status"""
    return {
        "status": "running",
        "watch_folder": WATCH_FOLDER,
        "device": str(device),
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }


def start_file_watcher():
    """Start the file watcher in a separate thread"""
    print(f"[INFO] Watching folder: {WATCH_FOLDER}")
    observer = Observer()
    observer.schedule(NewFileHandler(), path=WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    # Start file watcher in a separate thread
    watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
    watcher_thread.start()

    # Start FastAPI server
    print(f"[INFO] Starting server at http://localhost:8001")
    print(f"[INFO] WebSocket endpoint: ws://localhost:8001/ws")
    uvicorn.run(
        app,
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8001,
        log_level="info"
    )