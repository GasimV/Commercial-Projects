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
import joblib
import uuid

import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import List

# ========================== MODELS ==========================

device = torch.device("cuda")


class Diarizer:
    def __init__(self, hf_token: str, emotion_pipeline, device="cuda",
                 debug_dir="./debug"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        self.device = device
        self.debug_dir = debug_dir
        self.denoise_model = pretrained.dns64().to(device)
        self.emotion_pipeline = emotion_pipeline

    def denoise_audio(self, audio_path):
        """Denoise audio file and return path to denoised version"""
        os.makedirs(self.debug_dir, exist_ok=True)
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.denoise_model.sample_rate, self.denoise_model.chin)
        with torch.no_grad():
            enhanced = self.denoise_model(wav.to(self.device))
        enhanced = enhanced.squeeze(0).cpu()
        out_path = os.path.join(self.debug_dir, f"denoised_{uuid.uuid4().hex}.wav")
        torchaudio.save(out_path, enhanced, self.denoise_model.sample_rate)
        return out_path

    # <<< MODIFIED: The entire transcribe function is updated to include emotion
    def transcribe(self, wav_path: str, stt_pipeline, num_speakers=2, language="azerbaijani") -> List[Dict]:
        """
        Transcribes audio, performs diarization, and detects emotion for each segment.
        Returns a list of dictionaries, each containing speaker, text, and emotion.
        """
        diarization = self.pipeline(wav_path, num_speakers=num_speakers)
        audio = AudioSegment.from_file(wav_path)

        dialogue_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
            segment_audio = audio[start_ms:end_ms]

            # <<< --- FIX: SKIP AUDIO SEGMENTS THAT ARE TOO SHORT --- >>>
            # A duration of 200ms is a safe minimum.
            if len(segment_audio) < 200:
                continue  # Skip to the next segment

            # Prepare buffer for transcription and emotion analysis
            buffer = BytesIO()
            segment_audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()

            # Transcribe
            transcription = stt_pipeline(audio_bytes)
            text = transcription['text'].strip()

            # Skip empty or boilerplate transcriptions
            if not text or "zlədiyiniz üçün təşəkkürlər" in text.lower():
                continue

            # Analyze emotion
            emotion_results = self.emotion_pipeline(audio_bytes, top_k=1)
            emotion_label = emotion_results[0]['label'] if emotion_results else 'neu'  # Default to neutral

            dialogue_segments.append({
                "speaker": speaker,
                "text": text,
                "emotion": emotion_label,
                "start": turn.start,
                "end": turn.end
            })

        # Merge consecutive segments from the same speaker
        if not dialogue_segments:
            return []

        merged_dialogue = [dialogue_segments[0]]
        for i in range(1, len(dialogue_segments)):
            current_segment = dialogue_segments[i]
            last_merged = merged_dialogue[-1]

            if current_segment['speaker'] == last_merged['speaker']:
                # Merge text and update end time
                last_merged['text'] += " " + current_segment['text']
                last_merged['end'] = current_segment['end']
                # The emotion of the last part of the merged segment is kept
                last_merged['emotion'] = current_segment['emotion']
            else:
                merged_dialogue.append(current_segment)

        # Format the final output to be simpler for the next stages
        final_dialogue = [
            {"speaker": seg["speaker"], "text": seg["text"], "emotion": seg["emotion"]}
            for seg in merged_dialogue
        ]

        return final_dialogue


class SpeakerIdentifier:
    def __init__(self, model_path: str, device="cuda"):
        self.model_path = model_path
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def predict_speaker_and_flip(self, text_a: str, text_b: str):
        # ... (This function remains unchanged)
        inputs = self.tokenizer(text_a, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
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

        inputs = self.tokenizer(text_b, return_tensors="pt", truncation=True, padding=True, max_length=512).to(
            self.device)
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

    # This function accepts and returns a list of dictionaries
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
            # If not 2 speakers, just label them generically and return
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
    def __init__(self, model_path: str, device="cuda"):
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
                num_beams=15,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                max_length=128
            )

        return self.tokenizer.decode(preds[0], skip_special_tokens=True)


class CallProcessor:
    def __init__(self, hf_token: str, whisper_model, id_model_path: str, sum_model_path: str, device="cuda",
                 debug_dir="./debug"):
        # Initialize the emotion pipeline
        self.emotion_pipeline = pipeline(
            "audio-classification",
            model="superb/wav2vec2-large-superb-er",
            device=0 if device == "cuda" else -1  # Use 0 for first GPU, -1 for CPU
        )

        # Pass the emotion pipeline to the Diarizer
        self.diarizer = Diarizer(hf_token, self.emotion_pipeline, device=device, debug_dir=debug_dir)

        self.identifier = SpeakerIdentifier(id_model_path)
        self.summarizer = Summarizer(sum_model_path, device=device)
        self.whisper_model = whisper_model
        self.device = device

        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.bert_model.eval()
        self.bert_model.to(device)
        self.knn_model = joblib.load("knn_model/knn_model.pkl")

    def classify_transcription(self, wav_path):
        # ... (This function remains unchanged)
        denoised_path = self.diarizer.denoise_audio(wav_path)
        result = self.whisper_model(denoised_path)
        text = result['text'].strip().lower()
        encoded = self.tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.bert_model(**encoded)
            embedding = output.pooler_output.detach().cpu().numpy().squeeze()
        prediction = self.knn_model.predict([embedding])[0]
        label = "Operator tərəfindən yaxşı cavablandırıldı" if prediction == 1 else "Operator tərəfindən pis cavablandırıldı"
        return label

    # A helper for emotion trend analysis
    def _analyze_emotion_trend(self, dialogue: List[Dict]) -> str:
        """Analyzes the emotional trend of the customer."""
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

    # The main processing generator
    def process_call(self, wav_path: str, num_speakers: int = 2, language: str = "azerbaijani"):
        # This returns a list of dicts: [{'speaker': '...', 'text': '...', 'emotion': '...'}]
        dialogue = self.diarizer.transcribe(wav_path, self.whisper_model, num_speakers=num_speakers, language=language)

        # Format for display on the front end
        dialogue_for_display = [f"{d['speaker']} ({d['emotion']}): {d['text']}" for d in dialogue]
        yield {"stage": "dialogue", "result": dialogue_for_display}

        # This also returns a list of dicts with 'customer'/'operator' labels
        identified_dialogue = self.identifier.identify_speakers(dialogue)

        identified_dialogue_for_display = [f"{d['speaker']} ({d['emotion']}): {d['text']}" for d in identified_dialogue]
        yield {"stage": "identified_dialogue", "result": identified_dialogue_for_display}

        # Emotion Analysis
        emotion_summary = self._analyze_emotion_trend(identified_dialogue)
        yield {"stage": "emotion_analysis", "result": emotion_summary}

        # Prepare dialogue for summarizer (it needs a list of strings)
        dialogue_for_summary = [f"{d['speaker']}: {d['text']}" for d in identified_dialogue]
        summary = self.summarizer.summarize(dialogue_for_summary)
        yield {"stage": "summary", "result": summary}

        # Classification
        classification_label = self.classify_transcription(wav_path)
        yield {"stage": "classification", "result": classification_label}