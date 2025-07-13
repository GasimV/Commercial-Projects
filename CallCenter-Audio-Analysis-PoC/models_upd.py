import torch
import torch.nn.functional as F
import torchaudio
from typing import Tuple, List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, \
    BertModel, pipeline
from io import BytesIO
from pydub import AudioSegment
from pyannote.audio import Pipeline
from denoiser import pretrained
from denoiser.dsp import convert_audio
import uuid
import pandas as pd

import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List

# ========================== MODELS ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Diarizer:
    def __init__(self, hf_token: str, emotion_pipeline, device=device,
                 debug_dir="./debug"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        ).to(device)
        self.device = device
        self.debug_dir = debug_dir
        self.denoise_model = pretrained.dns64().to(device)
        self.emotion_pipeline = emotion_pipeline

    def denoise_audio(self, audio_path):
        os.makedirs(self.debug_dir, exist_ok=True)
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.denoise_model.sample_rate, self.denoise_model.chin)
        with torch.no_grad():
            enhanced = self.denoise_model(wav.unsqueeze(0).to(self.device))
        enhanced = enhanced.squeeze(0).cpu()
        out_path = os.path.join(self.debug_dir, f"denoised_{uuid.uuid4().hex}.wav")
        torchaudio.save(out_path, enhanced, self.denoise_model.sample_rate)
        return out_path

    def transcribe(self, wav_path: str, stt_pipeline, num_speakers=2, language="azerbaijani") -> List[Dict]:
        diarization = self.pipeline(wav_path, num_speakers=num_speakers)
        audio = AudioSegment.from_file(wav_path)

        dialogue_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
            segment_audio = audio[start_ms:end_ms]

            if len(segment_audio) < 200:
                continue

            buffer = BytesIO()
            segment_audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()

            transcription = stt_pipeline(audio_bytes)
            text = transcription['text'].strip()

            if not text or "zlədiyiniz üçün təşəkkürlər" in text.lower():
                continue

            emotion_results = self.emotion_pipeline(audio_bytes, top_k=1)
            emotion_label = emotion_results[0]['label'] if emotion_results else 'neu'

            dialogue_segments.append({
                "speaker": speaker,
                "text": text,
                "emotion": emotion_label,
                "start": turn.start,
                "end": turn.end
            })

        if not dialogue_segments:
            return []

        merged_dialogue = [dialogue_segments[0]]
        for i in range(1, len(dialogue_segments)):
            current_segment = dialogue_segments[i]
            last_merged = merged_dialogue[-1]

            if current_segment['speaker'] == last_merged['speaker']:
                last_merged['text'] += " " + current_segment['text']
                last_merged['end'] = current_segment['end']
                last_merged['emotion'] = current_segment['emotion']
            else:
                merged_dialogue.append(current_segment)

        final_dialogue = [
            {"speaker": seg["speaker"], "text": seg["text"], "emotion": seg["emotion"]}
            for seg in merged_dialogue
        ]

        return final_dialogue


class SpeakerIdentifier:
    def __init__(self, model_path: str, device=device):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def predict_speaker_and_flip(self, text_a: str, text_b: str):
        inputs = self.tokenizer(text_a, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_class_id = torch.max(probs, dim=-1)

        label_map = {0: "customer", 1: "operator"}
        label = predicted_class_id.item()
        conf = confidence.item()
        print(f"Confidence A: {conf:.3f}")

        if conf > 0.98:
            return label_map[label], label_map[1 - label]

        inputs = self.tokenizer(text_b, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted_class_id = torch.max(probs, dim=-1)

        label = predicted_class_id.item()
        conf = confidence.item()
        print(f"Confidence B: {conf:.3f}")

        if conf > 0.95:
            return label_map[1 - label], label_map[label]
        else:
            return label_map[label], label_map[label]

    def identify_speakers(self, dialogue: List[Dict]) -> List[Dict]:
        if not dialogue:
            return []

        speaker_texts = {}
        for segment in dialogue:
            spk = segment['speaker']
            text = segment['text']
            if spk not in speaker_texts:
                speaker_texts[spk] = []
            speaker_texts[spk].append(text)

        speakers = list(speaker_texts.keys())
        if len(speakers) != 2:
            for segment in dialogue:
                segment['speaker'] = f"speaker_{segment['speaker'][-2:]}"
            return dialogue

        spk_a, spk_b = speakers
        text_a = " ".join(speaker_texts[spk_a])
        text_b = " ".join(speaker_texts[spk_b])

        label_a, label_b = self.predict_speaker_and_flip(text_a, text_b)
        role_map = {spk_a: label_a, spk_b: label_b}

        labeled_dialogue = []
        for segment in dialogue:
            original_speaker = segment['speaker']
            labeled_segment = segment.copy()
            labeled_segment['speaker'] = role_map.get(original_speaker, "unknown")
            labeled_dialogue.append(labeled_segment)

        return labeled_dialogue


class Summarizer:
    def __init__(self, model_path: str, device=device):
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    def summarize(self, dialogue: List[str]) -> str:
        input_text = "\n".join(dialogue)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            preds = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=8,  # Reduced beams slightly
                num_return_sequences=1,
                no_repeat_ngram_size=2,  # Increased ngram size to avoid repetition
                min_length=15,  # Set a minimum length
                max_length=150  # Increased max length
            )

        return self.tokenizer.decode(preds[0], skip_special_tokens=True)


class CallProcessor:
    def __init__(self, hf_token: str, whisper_model, id_model_path: str, sum_model_path: str, device=device,
                 debug_dir="./debug"):
        self.device = device
        self.whisper_model = whisper_model

        self.emotion_pipeline = pipeline(
            "audio-classification",
            model="superb/wav2vec2-large-superb-er",
            device=0 if device.type == "cuda" else -1
        )

        # Load Topic model components
        print("Loading Topic Classification pipeline...")
        self.topic_classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=-1  # Force to CPU to avoid TorchScript bug and use accurate processing
        )

        # Load and process topics for hierarchical classification
        try:
            topics_df = pd.read_csv("topics.csv").dropna(subset=['Class', 'Sub_class'])
            self.topic_classes = topics_df['Class'].unique().tolist()
            self.class_to_subclass_map = topics_df.groupby('Class')['Sub_class'].apply(list).to_dict()
            print(f"Loaded {len(self.topic_classes)} classes and their subclasses for topic classification.")
        except FileNotFoundError:
            print("[ERROR] topics.csv not found. Topic classification will be disabled.")
            self.topic_classes = []
            self.class_to_subclass_map = {}

        print("Loading Quality Assessment (Sentiment) model...")
        self.quality_classifier = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if device.type == "cuda" else -1
        )

        self.diarizer = Diarizer(hf_token, self.emotion_pipeline, device=device, debug_dir=debug_dir)
        self.identifier = SpeakerIdentifier(id_model_path, device=device)
        self.summarizer = Summarizer(sum_model_path, device=device)

    def classify_quality(self, summary_text: str) -> Dict[str, any]:
        """Classifies the quality of the interaction and returns a label and score."""
        if not summary_text:
            return {"label": "Not enough text for quality classification.", "score": 0}

        try:
            result = self.quality_classifier(summary_text)[0]
            score = int(result['label'].split()[0])  # '5 stars' -> 5
            label = "Operator tərəfindən yaxşı cavablandırıldı" if score >= 4 else "Operator tərəfindən pis cavablandırıldı"

            return {"label": label, "score": score}
        except Exception as e:
            print(f"[ERROR] Quality classification failed: {e}")
            return {"label": "Error during quality classification.", "score": 0}


    def classify_topic(self, text_to_classify: str) -> str:
        """Classifies text using a 2-step hierarchical approach: Class -> Sub-class."""
        if not self.topic_classes or not text_to_classify or len(text_to_classify.split()) < 3:
            return "Not enough data for topic classification."

        # Step 1: Predict the main class using the pipeline
        print("--- Topic Classification: Step 1 (Class) ---")
        class_output = self.topic_classifier(text_to_classify, self.topic_classes, multi_label=False)
        predicted_class = class_output['labels'][0]
        print(f"Pipeline Prediction: '{predicted_class}' with score {class_output['scores'][0]:.2f}")


        # Step 2: Predict the sub-class from the filtered list
        sub_class_options = self.class_to_subclass_map.get(predicted_class)
        if not sub_class_options:
            return f"{predicted_class}: (No sub-classes found)"

        print(f"--- Topic Classification: Step 2 (Sub-class for '{predicted_class}') ---")
        sub_class_output = self.topic_classifier(text_to_classify, sub_class_options, multi_label=False)
        predicted_sub_class = sub_class_output['labels'][0]
        print(f"Pipeline Prediction: '{predicted_sub_class}' with score {sub_class_output['scores'][0]:.2f}")

        return f"{predicted_class}: {predicted_sub_class}"


    def _analyze_emotion_trend(self, dialogue: List[Dict]) -> str:
        customer_emotions = [seg['emotion'] for seg in dialogue if seg['speaker'] == 'customer']

        if not customer_emotions:
            return "No customer speech detected to analyze emotion."

        emotion_map = {'hap': 'Happy', 'neu': 'Neutral', 'sad': 'Sad', 'ang': 'Angry'}
        trend = " -> ".join([emotion_map.get(e, e) for e in customer_emotions])
        start_emotion = emotion_map.get(customer_emotions[0], customer_emotions[0])
        end_emotion = emotion_map.get(customer_emotions[-1], customer_emotions[-1])

        summary = f"Customer emotion trend: {trend}. "
        if len(customer_emotions) > 1:
            if start_emotion == end_emotion:
                summary += f"The customer's emotion remained {end_emotion}."
            else:
                summary += f"The customer started {start_emotion} and finished {end_emotion}."
        else:
            summary += f"The customer's emotion was {start_emotion}."
        return summary

    def process_call(self, wav_path: str, num_speakers: int = 2, language: str = "azerbaijani"):
        dialogue = self.diarizer.transcribe(wav_path, self.whisper_model, num_speakers=num_speakers, language=language)
        yield {"stage": "dialogue", "result": [f"{d['speaker']} ({d['emotion']}): {d['text']}" for d in dialogue]}

        identified_dialogue = self.identifier.identify_speakers(dialogue)
        yield {"stage": "identified_dialogue", "result": [f"{d['speaker']} ({d['emotion']}): {d['text']}" for d in identified_dialogue]}

        emotion_summary = self._analyze_emotion_trend(identified_dialogue)
        yield {"stage": "emotion_analysis", "result": emotion_summary}

        summary = self.summarizer.summarize([f"{d['speaker']}: {d['text']}" for d in identified_dialogue])
        yield {"stage": "summary", "result": summary}

        topic = self.classify_topic(summary)
        yield {"stage": "topic_classification", "result": topic}

        # <<< MODIFIED: Pass the entire dictionary from classify_quality
        quality_result = self.classify_quality(summary)
        yield {"stage": "classification", "result": quality_result}