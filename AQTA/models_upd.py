# --- START OF FILE models_upd.py ---

import torch
#import torchaudio
import soundfile as sf
import numpy as np
from typing import List, Dict
from denoiser import pretrained
from denoiser.dsp import convert_audio
import uuid
import os
import time
import json

import google.generativeai as genai

# ========================== MODELS ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GeminiAnalyzer:
    """
    Handles all audio analysis tasks (diarization, transcription, emotion, etc.)
    through a single call to the Google Gemini API.
    """

    def __init__(self):
        try:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel('models/gemini-2.5-flash') # models/gemini-2.5-flash-lite or models/gemini-2.5-pro
            print("Gemini Analyzer initialized successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to initialize GeminiAnalyzer. Please check your GOOGLE_API_KEY. Error: {e}")
            self.model = None

    def analyze_audio(self, audio_path: str) -> Dict:
        if not self.model:
            return {"error": "Gemini model not initialized."}

        print(f"Uploading '{audio_path}' to Google API for full analysis...")
        try:
            audio_file = genai.upload_file(path=audio_path)
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            if audio_file.state.name == "FAILED":
                raise ValueError("Audio file processing failed on the server.")

            print("File uploaded. Generating full analysis...")

            # <<< MODIFICATION: Added Topic Classification to the prompt >>>
            prompt = """
                Sən çağrı mərkəzi audio analizində mütəxəssis olan süni intellekt köməkçisisən. Sənin vəzifən təqdim olunmuş Azərbaycan dilindəki audio faylı analiz etmək və strukturlaşdırılmış JSON obyektini qaytarmaqdır.
                
                Təlimatlara dəqiq əməl et:
                1. **Diarizasiya və Transkripsiya:** Danışanları “müştəri” və “operator” kimi müəyyən et. Bütün söhbətin dönüş-dönüş transkripsiyasını Azərbaycan dilində ver.
                2. **Emosiyaların Analizi:** Dialoqdakı hər bir növbədə danışanın emosiyasını bu siyahıdan seç: `Xoşbəxt`, `Neytral`, `Kədərli`, `Qəzəbli`.
                3. **Mövzu Təsnifatı:** Söhbətin əsas mövzusunu bu siyahıdan seç: `Hesab Məlumatları`, `Kredit Müraciəti`, `Kart Sifarişi`, `Texniki Dəstək`, `Şikayət`, `Ümumi Məlumat`.
                4. **Xülasə:** Zəngin məqsədini və nəticəsini Azərbaycan dilində bir cümlə ilə qısa şəkildə ifadə et.
                5. **Keyfiyyət Qiymətləndirilməsi:** Operatorun xidmətini 1-dən 5-ə qədər ulduzla qiymətləndir. Həmçinin bir sözlə etiketlə: 4-5 ulduz üçün "Yaxşı cavab", 1-3 ulduz üçün "Pis cavab".
                
                YALNIZ aşağıdakı formatda bir JSON obyekti qaytar. Başqa heç bir mətn və ya markdown formatlaşdırma əlavə etmə.
                
                {
                  "dialogue": [
                    {"speaker": "customer", "text": "Müştərinin transkripsiya olunmuş sözü.", "emotion": "Neytral"},
                    {"speaker": "operator", "text": "Operatorun transkripsiya olunmuş sözü.", "emotion": "Xoşbəxt"}
                  ],
                  "topic": "Yuxarıdakı siyahıdan seçilmiş mövzu.",
                  "summary": "Zəngin məqsədi və nəticəsi barədə Azərbaycan dilində bir cümləlik xülasə.",
                  "quality_assessment": {
                    "score": 5,
                    "label": "Yaxşı cavab"
                  }
                }
            """

            response = self.model.generate_content([prompt, audio_file])
            genai.delete_file(audio_file.name)
            print("Full analysis received from Gemini.")

            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)

        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON from Gemini response: {response.text}")
            return {"error": "Invalid JSON format received from the API."}
        except Exception as e:
            print(f"[ERROR] Gemini analysis failed: {e}")
            return {"error": str(e)}


class CallProcessor:
    def __init__(self, device=device, debug_dir="./debug"):
        self.device = device
        self.debug_dir = debug_dir
        self.denoise_model = pretrained.dns64().to(device)
        self.analyzer = GeminiAnalyzer()

    def denoise_audio(self, audio_path):
        # os.makedirs(self.debug_dir, exist_ok=True)
        # wav, sr = torchaudio.load(audio_path)
        # wav = convert_audio(wav, sr, self.denoise_model.sample_rate, self.denoise_model.chin)
        # with torch.no_grad():
        #     enhanced = self.denoise_model(wav.unsqueeze(0).to(self.device))
        # enhanced = enhanced.squeeze(0).cpu()
        # out_path = os.path.join(self.debug_dir, f"denoised_{uuid.uuid4().hex}.wav")
        # torchaudio.save(out_path, enhanced, self.denoise_model.sample_rate)
        # return out_path

        # Read with soundfile → (T, C) float32; convert to torch (C, T)
        audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # shape (T, C)
        wav = torch.from_numpy(audio_np.T)  # (C, T)

        # Match the denoiser model's expected SR & channels
        wav = convert_audio(
            wav, sr,
            self.denoise_model.sample_rate,
            self.denoise_model.chin
        )

        with torch.no_grad():
            enhanced = self.denoise_model(wav.unsqueeze(0).to(self.device))  # (1, C, T)
        enhanced = enhanced.squeeze(0).cpu()  # (C, T)

        out_path = os.path.join(self.debug_dir, f"denoised_{uuid.uuid4().hex}.wav")
        # soundfile expects (T, C)
        sf.write(out_path, enhanced.T.numpy(), self.denoise_model.sample_rate)
        return out_path

    def _analyze_emotion_trend(self, dialogue: List[Dict]) -> str:
        # customer_emotions = [seg['emotion'] for seg in dialogue if seg.get('speaker') == 'müştəri']
        customer_labels = {"müştəri", "mustəri", "müşteri", "customer", "Customer", "Müştəri"}

        customer_emotions = [
            seg["emotion"]
            for seg in dialogue
            if seg.get("speaker", "").strip().lower() in {lbl.lower() for lbl in customer_labels}
        ]

        if not customer_emotions:
            return "No customer speech detected to analyze emotion."

        trend = " -> ".join(customer_emotions)
        start_emotion = customer_emotions[0]
        end_emotion = customer_emotions[-1]

        summary = f"Customer emotion trend: {trend}. "
        if len(customer_emotions) > 1:
            if start_emotion == end_emotion:
                summary += f"The customer's emotion remained {end_emotion}."
            else:
                summary += f"The customer started {start_emotion} and finished {end_emotion}."
        else:
            summary += f"The customer's emotion was {start_emotion}."
        return summary

    def process_call(self, wav_path: str):
        denoised_path = self.denoise_audio(wav_path)
        yield {"stage": "denoised_audio_ready", "result": {"denoised_filename": os.path.basename(denoised_path)}}

        analysis_result = self.analyzer.analyze_audio(denoised_path)

        if "error" in analysis_result:
            error_message = analysis_result["error"]
            yield {"stage": "identified_dialogue", "result": [f"ERROR: {error_message}"]}
            yield {"stage": "emotion_analysis", "result": "Analysis failed."}
            yield {"stage": "summary", "result": "Analysis failed."}
            yield {"stage": "topic_classification", "result": "Analysis failed."}
            yield {"stage": "classification", "result": {"label": "Error", "score": 0}}
            return

        dialogue = analysis_result.get("dialogue", [])
        summary = analysis_result.get("summary", "No summary provided.")
        topic = analysis_result.get("topic", "No topic classified.") # <<< MODIFICATION: Get topic
        quality = analysis_result.get("quality_assessment", {"label": "N/A", "score": 0})

        # <<< MODIFICATION: The first "dialogue" yield has been REMOVED >>>
        # The first and only dialogue display will be the final, identified one.
        formatted_dialogue = [f"{d.get('speaker', 'unknown')} ({d.get('emotion', 'neu')}): {d.get('text', '')}" for d in
                              dialogue]
        yield {"stage": "identified_dialogue", "result": formatted_dialogue}

        emotion_summary = self._analyze_emotion_trend(dialogue)
        yield {"stage": "emotion_analysis", "result": emotion_summary}

        yield {"stage": "summary", "result": summary}

        # <<< MODIFICATION: Yield the topic classification stage >>>
        yield {"stage": "topic_classification", "result": topic}

        quality_label = "Operator tərəfindən yaxşı cavablandırıldı" if quality.get('score',
                                                                                   0) >= 4 else "Operator tərəfindən pis cavablandırıldı"
        yield {"stage": "classification", "result": {"label": quality_label, "score": quality.get('score', 0)}}