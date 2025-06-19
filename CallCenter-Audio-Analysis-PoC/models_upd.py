import torch
import torch.nn.functional as F
import torchaudio
from typing import Tuple, List
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, BertModel
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
    def __init__(self, hf_token: str, device="cuda", debug_dir="./debug"):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        self.device = device
        self.debug_dir = debug_dir
        # Initialize denoise model
        self.denoise_model = pretrained.dns64().to(device)

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

    def transcribe(self, wav_path: str, sw_model, num_speakers=2, language="azerbaijani") -> List[str]:
        diarization = self.pipeline(wav_path, num_speakers=num_speakers)
        audio = AudioSegment.from_file(wav_path)
        raw_dialogue = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms, end_ms = int(turn.start * 1000), int(turn.end * 1000)
            segment = audio[start_ms:end_ms]

            buffer = BytesIO()
            segment.export(buffer, format="wav")
            buffer.seek(0)

            transcription = sw_model.transcribe(buffer.read(), language=language, word_timestamps=False)
            text = transcription.text.strip()
            text = text.lower().replace("zlədiyiniz üçün təşəkkürlər", "").strip()

            if text and "zlədiyiniz üçün təşəkkürlər" not in text.lower():
                raw_dialogue.append((speaker, text))

        # Merge consecutive lines by the same speaker
        dialogue = []
        for speaker, text in raw_dialogue:
            if dialogue and dialogue[-1].startswith(f"{speaker}:"):
                dialogue[-1] = f"{speaker}: {dialogue[-1].split(': ', 1)[1]} {text}"
            else:
                dialogue.append(f"{speaker}: {text}")

        return dialogue


class SpeakerIdentifier:
    def __init__(self, model_path: str, device="cuda"):
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

        # Try flipping
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

    def identify_speakers(self, dialogue: List[str]) -> List[str]:
        if not dialogue or len(dialogue) < 2:
            return dialogue  # Not enough data

        # Step 1: Group all text by speaker ID
        speaker_texts = {}
        for line in dialogue:
            spk, text = line.split(":", 1)
            text = text.strip()
            if spk not in speaker_texts:
                speaker_texts[spk] = []
            speaker_texts[spk].append(text)

        speakers = list(speaker_texts.keys())
        if len(speakers) != 2:
            raise ValueError("Expected exactly 2 speakers for prediction.")

        spk_a, spk_b = speakers
        text_a = " ".join(speaker_texts[spk_a])
        text_b = " ".join(speaker_texts[spk_b])

        # Step 2: Predict roles
        label_a, label_b = self.predict_speaker_and_flip(text_a, text_b)
        label_map = {spk_a: label_a, spk_b: label_b}

        # Step 3: Apply mapping
        labeled_dialogue = []
        for line in dialogue:
            spk, text = line.split(":", 1)
            role = label_map.get(spk, "unknown")
            labeled_dialogue.append(f"{role}: {text.strip()}")

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
    def __init__(self, hf_token: str, whisper_model, id_model_path: str, sum_model_path: str, device="cuda", debug_dir="./debug"):
        # Pass device and debug_dir to Diarizer
        self.diarizer = Diarizer(hf_token, device=device, debug_dir=debug_dir)
        self.identifier = SpeakerIdentifier(id_model_path)
        self.summarizer = Summarizer(sum_model_path, device=device)
        self.whisper_model = whisper_model
        self.device = device

        # Initialize BERT + KNN models (denoise is now handled by Diarizer)
        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.bert_model.eval()
        self.bert_model.to(device)
        self.knn_model = joblib.load("knn_model/knn_model.pkl")

    def classify_transcription(self, wav_path):
        # Step 1: Denoise the audio
        denoised_path = self.diarizer.denoise_audio(wav_path)

        # Step 2: Transcribe using Whisper
        result = self.whisper_model.transcribe(denoised_path, language="azerbaijani", word_timestamps=False)
        text = result.text.strip().lower()

        # Step 3: Encode and classify using BERT + KNN
        encoded = self.tokenizer_bert(text, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.bert_model(**encoded)
            embedding = output.pooler_output.detach().cpu().numpy().squeeze()
        prediction = self.knn_model.predict([embedding])[0]

        label = "Operator tərəfindən yaxşı cavablandırıldı" if prediction == 1 else "Operator tərəfindən pis cavablandırıldı"
        return label

    def process_call(self, wav_path: str, num_speakers: int = 2, language: str = "azerbaijani"):
        # Step 1 & 2: Denoise and transcribe (handled by Diarizer)
        dialogue = self.diarizer.transcribe(wav_path, self.whisper_model, num_speakers=num_speakers, language=language)
        yield {"stage": "dialogue", "result": dialogue}

        # Step 3: Identify speakers
        identified_dialogue = self.identifier.identify_speakers(dialogue)
        yield {"stage": "identified_dialogue", "result": identified_dialogue}

        # Step 4: Generate summary
        summary = self.summarizer.summarize(identified_dialogue)
        yield {"stage": "summary", "result": summary}

        # Step 5: BERT + KNN classification at the end
        classification_label = self.classify_transcription(wav_path)
        yield {"stage": "classification", "result": classification_label}
