"""
Local model adapters for the emotional recovery assistant.

The adapters are intentionally lazy: heavy speech, vision, and text models are
imported only when their feature is used.
This keeps the desktop app usable even when one optional model is unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "data" / "model_config.json"
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
MODELS_ROOT = PROJECT_ROOT / "models"


def project_path(value: str | Path) -> Path:
    """Resolve relative config paths from the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_split_model_file(target: Path) -> None:
    """Rebuild a large model file from Git LFS parts when needed."""
    if target.exists():
        return

    parts = sorted(target.parent.glob(f"{target.name}.part*"))
    if not parts:
        return

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while True:
                    chunk = source.read(1024 * 1024 * 16)
                    if not chunk:
                        break
                    output.write(chunk)
    tmp_path.replace(target)


@dataclass
class LocalModelConfig:
    """Paths and endpoints used by local AI models."""

    speech_interaction_project: str = "third_party/speech_interaction"
    speech_generation_checkpoint: str = "models/speech_generation"
    speech_recognition_model_path: str = "models/speech_recognition_small/small.pt"
    speech_device: str = "cuda:0"
    facial_expression_project: str = "third_party/facial_expression_recognition"
    text_sentiment_project: str = "third_party/text_sentiment_recognition"
    text_sentiment_model_path: str = "models/text_sentiment_classifier"
    text_sentiment_server_url: str = ""
    local_dialogue_model_path: str = ""
    output_dir: str = "data/voice_outputs"

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "LocalModelConfig":
        if not path.exists():
            config = cls()
            config.save(path)
            return config

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            legacy_keys = {
                "mini_omni_project": "speech_interaction_project",
                "mini_omni_checkpoint": "speech_generation_checkpoint",
                "whisper_model_path": "speech_recognition_model_path",
                "mini_omni_device": "speech_device",
                "paz_project": "facial_expression_project",
                "sentiment_project": "text_sentiment_project",
                "sentiment_model_path": "text_sentiment_model_path",
                "sentiment_server_url": "text_sentiment_server_url",
            }
            for old_key, new_key in legacy_keys.items():
                if old_key in data and new_key not in data:
                    data[new_key] = data[old_key]
            defaults = asdict(cls())
            defaults.update({k: v for k, v in data.items() if k in defaults})
            return cls(**defaults)
        except Exception:
            return cls()

    def save(self, path: Path = DEFAULT_CONFIG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


class SentimentModelAdapter:
    """Text sentiment classifier backed by a local transformer classifier."""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._analyzer = None

    def classify(self, text: str) -> Dict:
        if not text.strip():
            return {"sentiment": "Neutral", "percentage": 0, "available": False}

        if self.config.text_sentiment_server_url:
            try:
                response = requests.get(
                    self.config.text_sentiment_server_url,
                    params={"text": text},
                    timeout=20,
                )
                response.raise_for_status()
                data = response.json()
                data["available"] = True
                return data
            except Exception as exc:
                return {"sentiment": "Unknown", "percentage": 0, "available": False, "error": str(exc)}

        try:
            analyzer = self._get_analyzer()
            sentiment, percentage = analyzer.classify_sentiment(text)
            return {"sentiment": sentiment, "percentage": percentage, "available": True}
        except Exception as exc:
            return {
                "sentiment": "Unknown",
                "percentage": 0,
                "available": False,
                "error": str(exc),
            }

    def _get_analyzer(self):
        if self._analyzer is not None:
            return self._analyzer

        project = project_path(self.config.text_sentiment_project)
        if not project.exists():
            raise FileNotFoundError(f"Sentiment project not found: {project}")

        old_argv = sys.argv[:]
        sys.path.insert(0, str(project))
        try:
            model_name_or_path = self._resolve_model_name_or_path()
            sys.argv = ["sentiment_adapter", "--model_name_or_path", model_name_or_path]
            from arguments import args  # type: ignore
            from analyzer import Analyzer  # type: ignore

            self._analyzer = Analyzer(will_train=False, args=args)
            return self._analyzer
        finally:
            sys.argv = old_argv

    def _resolve_model_name_or_path(self) -> str:
        configured = self.config.text_sentiment_model_path
        path = project_path(configured)
        return str(path) if path.exists() else configured


class FacialEmotionAdapter:
    """Facial expression recognition backed by a local CNN pipeline."""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._pipeline = None
        self._classifier = None
        self._haar_detectors = None
        self._lock = threading.Lock()
        self._last_face_result = None

    def analyze_image(self, image_path: str) -> Dict:
        try:
            import cv2

            bgr_image = cv2.imread(image_path)
            if bgr_image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            with self._lock:
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                pipeline_error = ""
                try:
                    pipeline = self._get_pipeline()
                    output = pipeline(rgb_image.copy())
                    detected_faces = self._boxes_to_faces(output.get("boxes2D", []))
                    if detected_faces:
                        return self._remember_face_result({"available": True, "faces": detected_faces, "method": "paz"})
                except Exception as exc:
                    pipeline_error = str(exc)

                detected_faces = self._detect_with_opencv(bgr_image)
                result = {"available": bool(detected_faces), "faces": detected_faces}
            if pipeline_error and not detected_faces:
                result["error"] = pipeline_error
            if result.get("available") and result.get("faces"):
                return self._remember_face_result(result)
            return result
        except Exception as exc:
            return {"available": False, "faces": [], "error": str(exc)}

    def _remember_face_result(self, result: Dict) -> Dict:
        if result.get("available") and result.get("faces"):
            face = result["faces"][0]
            if not face.get("classifier_error"):
                self._last_face_result = result
        return result

    def _boxes_to_faces(self, boxes2d) -> List[Dict]:
        faces = []
        for box in boxes2d:
            score = float(getattr(box, "score", 0.0))
            if score < 0.55:
                continue
            faces.append({
                "emotion": getattr(box, "class_name", "unknown"),
                "score": score,
            })
        return faces

    def _detect_with_opencv(self, bgr_image) -> List[Dict]:
        import cv2

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        min_size = max(32, min(w, h) // 12)
        for detector in self._get_haar_detectors():
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(min_size, min_size),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) == 0:
                continue
            x, y, fw, fh = max(faces, key=lambda item: item[2] * item[3])
            pad = int(0.18 * max(fw, fh))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(bgr_image.shape[1], x + fw + pad), min(bgr_image.shape[0], y + fh + pad)
            crop = cv2.cvtColor(bgr_image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            if not self._is_face_crop_quality_ok(crop):
                continue
            try:
                prediction = self._classify_crop(crop)
                prediction["method"] = "opencv-haar"
            except Exception as exc:
                prediction = {
                    "emotion": "face detected",
                    "score": 1.0,
                    "method": "opencv-haar",
                    "classifier_error": str(exc),
                }
            return [prediction]
        return []

    def _is_face_crop_quality_ok(self, rgb_crop) -> bool:
        """Reject blurred/covered crops before emotion classification."""
        import cv2
        import numpy as np

        if rgb_crop is None or rgb_crop.size == 0:
            return False
        h, w = rgb_crop.shape[:2]
        if h < 48 or w < 48:
            return False
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if mean < 35 or mean > 235:
            return False
        if std < 18:
            return False
        if blur_var < 35:
            return False
        return True

    def _get_haar_detectors(self):
        if self._haar_detectors is not None:
            return self._haar_detectors

        import cv2

        names = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml",
        ]
        detectors = []
        for name in names:
            path = cv2.data.haarcascades + name
            detector = cv2.CascadeClassifier(path)
            if not detector.empty():
                detectors.append(detector)
        self._haar_detectors = detectors
        return self._haar_detectors

    def _classify_crop(self, rgb_crop) -> Dict:
        import numpy as np

        classifier = self._get_classifier()
        prediction = classifier(rgb_crop)
        return {
            "emotion": prediction.get("class_name", "unknown"),
            "score": float(np.max(prediction.get("scores", [0.0]))),
        }

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        project = project_path(self.config.facial_expression_project)
        if not project.exists():
            raise FileNotFoundError(f"PAZ project not found: {project}")

        sys.path.insert(0, str(project))
        from paz.pipelines import DetectMiniXceptionFER  # type: ignore

        self._pipeline = DetectMiniXceptionFER([0.1, 0.1])
        return self._pipeline

    def _get_classifier(self):
        if self._classifier is not None:
            return self._classifier

        project = project_path(self.config.facial_expression_project)
        if not project.exists():
            raise FileNotFoundError(f"PAZ project not found: {project}")

        sys.path.insert(0, str(project))
        from paz.pipelines import MiniXceptionFER  # type: ignore

        self._classifier = MiniXceptionFER()
        return self._classifier


class MiniOmniVoiceAdapter:
    """Speech input/output adapter backed by local speech models."""

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._client = None
        self._whisper_asr_model = None
        self._step = 0

    def transcribe(self, audio_path: str, target_language: str = "zh") -> Dict:
        whisper_result = self._transcribe_with_whisper(audio_path, target_language=target_language)
        if whisper_result.get("available") and whisper_result.get("text"):
            return whisper_result
        return whisper_result

        try:
            client = self._get_client()
            from inference import A1_T1, _asr, _pad_a, get_input_ids_whisper, load_audio  # type: ignore

            mel, leng = load_audio(audio_path)
            audio_feature, input_ids = get_input_ids_whisper(
                mel, leng, client.whispermodel, client.device,
                special_token_a=_pad_a,
                special_token_t=_asr,
            )
            text = A1_T1(
                client.fabric,
                audio_feature,
                input_ids,
                leng,
                client.model,
                client.text_tokenizer,
                self._step,
            )
            self._step += 1
            text = text.strip()
            if text:
                return {"available": True, "text": text}
            return whisper_result
        except Exception as exc:
            error = str(exc)
            if whisper_result.get("error"):
                error = f"{whisper_result.get('error')} | advanced speech module: {error}"
            return {"available": False, "text": "", "error": error}

    def synthesize(self, text: str, language: str = "zh") -> Dict:
        if self._client is None:
            return self._synthesize_with_windows_sapi(text, language=language)

        mini_error = ""
        try:
            client = self._get_client()
            from inference import T1_A2, get_input_ids_TA  # type: ignore

            output_dir = project_path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            input_ids = get_input_ids_TA(text, client.text_tokenizer)
            T1_A2(
                client.fabric,
                input_ids,
                client.model,
                client.text_tokenizer,
                self._step,
                client.snacmodel,
                out_dir=str(output_dir),
            )
            audio_path = output_dir / "T1-A2" / f"{self._step:02d}.wav"
            self._step += 1
            if audio_path.exists():
                return {"available": True, "audio_path": str(audio_path), "engine": "advanced-speech"}
            mini_error = "advanced speech module did not produce an audio file"
        except Exception as exc:
            mini_error = str(exc)

        fallback = self._synthesize_with_windows_sapi(text, language=language)
        if fallback.get("available"):
            fallback["mini_omni_error"] = mini_error
            return fallback
        return {"available": False, "audio_path": "", "error": mini_error or fallback.get("error", "")}

    def _transcribe_with_whisper(self, audio_path: str, target_language: str = "zh") -> Dict:
        try:
            import torch
            import whisper

            model = self._get_whisper_asr_model()
            task = "translate" if target_language == "en" else "transcribe"
            language = None if target_language == "en" else "zh"
            result = model.transcribe(
                audio_path,
                fp16=torch.cuda.is_available(),
                language=language,
                task=task,
            )
            text = (result.get("text") or "").strip()
            if target_language == "zh":
                text = self._to_simplified_chinese(text)
            return {"available": bool(text), "text": text, "engine": "whisper"}
        except Exception as exc:
            return {"available": False, "text": "", "error": str(exc)}

    def _to_simplified_chinese(self, text: str) -> str:
        if not text:
            return text
        try:
            from opencc import OpenCC  # type: ignore
            return OpenCC("t2s").convert(text)
        except Exception:
            translation = str.maketrans({
                "現": "现", "開": "开", "關": "关", "語": "语", "聲": "声", "聽": "听",
                "說": "说", "這": "这", "個": "个", "還": "还", "讓": "让", "對": "对",
                "應": "应", "復": "复", "麼": "么", "為": "为", "會": "会", "來": "来",
                "過": "过", "難": "难", "壓": "压", "慮": "虑", "幫": "帮", "嗎": "吗",
                "沒": "没", "學": "学", "麼": "么", "點": "点", "離": "离", "憂": "忧",
                "樂": "乐", "愛": "爱", "謝": "谢", "歡": "欢", "實": "实", "體": "体",
            })
            return text.translate(translation)

    def _get_whisper_asr_model(self):
        if self._whisper_asr_model is not None:
            return self._whisper_asr_model

        import torch
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = project_path(getattr(self.config, "speech_recognition_model_path", ""))
        if model_path.exists():
            self._whisper_asr_model = whisper.load_model(str(model_path), device=device)
        else:
            self._whisper_asr_model = whisper.load_model("tiny", device=device)
        return self._whisper_asr_model

    def _synthesize_with_windows_sapi(self, text: str, language: str = "zh") -> Dict:
        output_dir = project_path(self.config.output_dir) / "sapi"
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / f"{int(time.time() * 1000)}.wav"
        escaped_path = str(audio_path).replace("'", "''")
        escaped_text = text[:500].replace("'", "''")
        culture = "zh-CN" if language == "zh" else "en-US"
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$culture = '{culture}'; "
            "$voice = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like \"$culture*\" } | Select-Object -First 1; "
            "if ($voice -ne $null) { $s.SelectVoice($voice.VoiceInfo.Name); } "
            f"$s.SetOutputToWaveFile('{escaped_path}'); "
            f"$s.Speak('{escaped_text}'); "
            "$s.Dispose();"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            return {"available": audio_path.exists(), "audio_path": str(audio_path), "engine": "windows-sapi"}
        except Exception as exc:
            return {"available": False, "audio_path": "", "error": str(exc)}

    def _get_client(self):
        if self._client is not None:
            return self._client

        project = project_path(self.config.speech_interaction_project)
        ckpt = project_path(self.config.speech_generation_checkpoint)
        ensure_split_model_file(ckpt / "lit_model.pth")
        if not project.exists():
            raise FileNotFoundError(f"speech interaction module not found: {project}")
        if not ckpt.exists():
            raise FileNotFoundError(f"speech generation checkpoint not found: {ckpt}")

        sys.path.insert(0, str(project))
        from inference import OmniInference  # type: ignore

        self._client = OmniInference(str(ckpt), self.config.speech_device)
        return self._client


class EmotionalRecoveryResponder:
    """Local emotional-support response composer.

    This is a lightweight baseline that can be replaced by a trained local
    dialogue model. It combines text sentiment, facial emotion and RAG context
    into a structured emotional recovery answer.
    """

    def respond(
        self,
        user_text: str,
        language: str,
        issue_type: str,
        rag_context: str = "",
        sentiment: Optional[Dict] = None,
        facial: Optional[Dict] = None,
    ) -> Tuple[str, str, str, str]:
        clean_text = (user_text or "").strip()
        if not clean_text:
            clean_text = "语音输入" if language == "zh" else "voice input"
        signal_text = self._describe_signals(language, sentiment, facial)
        reference = self._short_reference(language, rag_context)
        focus = self._focus_sentence(clean_text, language)
        issue_hint = self._issue_hint(issue_type, language)
        positive = self._is_positive_mood(clean_text, issue_type)

        if language == "zh":
            if positive:
                empathy = (
                    f"【情绪确认】我听到你说的是：{focus}。这是一种值得被看见和保留下来的积极感受。"
                    f"开心并不是小事，它说明此刻有一些东西正在支持你、滋养你，或者让你重新感到有力量。{signal_text}{reference}"
                )
                cognitive = (
                    "【认知巩固】我们可以把这份开心拆成三个线索：发生了什么、你从中感受到了什么、它说明你真正重视什么。"
                    "这样做不是分析掉快乐，而是帮你看清快乐从哪里来。下次状态低落时，这些线索也能成为提醒：你并不是只能被困难定义。"
                )
                behavioral = (
                    "【行动计划】现在可以做三个小动作：第一，把让你开心的原因写成一句话；第二，做一件能延续这份状态的小事，"
                    "比如听一首喜欢的歌、给自己倒杯水、整理一下今天的好事；第三，如果合适，把这份开心分享给一个安全的人。"
                )
                motivation = (
                    "【继续前进】请允许自己认真享受这份开心。恢复不只是在难过时撑住，也包括在变好时把好的感觉接住。"
                    "今天这份轻松和愉快，都是你生活里真实存在的资源。"
                )
            else:
                empathy = (
                    f"【情绪验证】我听到你说的是：{focus}。这不像是一个可以被一句“想开点”带过的感受，"
                    f"它更像是你已经承受了一段时间、现在终于需要被认真看见的压力。"
                    f"从分类看，它接近「{issue_type}」；{issue_hint}{signal_text}"
                    f"你愿意把它说出来，说明你还在尝试和自己的情绪合作，而不是完全放弃自己。{reference}"
                )
                cognitive = (
                    "【认知重构】我们先把这件事拆成三层：第一层是事实，第二层是你对事实的解释，第三层是解释带来的情绪。"
                    "比如“我现在很难过/没考好/关系受挫”是事实的一部分，但“我永远不行”“以后都完了”往往是大脑在痛苦中给出的结论，"
                    "不一定是事实本身。你可以试着写下两个问题：一是支持这个最糟糕想法的证据有哪些；二是有没有哪怕一点点反例。"
                    "目的不是强迫自己乐观，而是给大脑多留一个出口。"
                )
                behavioral = (
                    "【行动计划】接下来可以按“先稳定身体，再处理问题”的顺序来。第一步，做一次30秒呼吸：吸气4秒、停2秒、呼气6秒，重复三轮。"
                    "第二步，把最困扰你的那句话写下来，然后改成一个今天能执行的小句子，例如“我先把这件事整理成三条”。"
                    "第三步，找一个低成本支持源：给可信任的人发一句“我现在状态不太好，能不能陪我聊十分钟”。"
                    "如果暂时不想联系别人，就先完成喝水、洗脸、离开屏幕5分钟这类能让身体降温的小动作。"
                )
                motivation = (
                    "【继续前进】你现在需要的不是立刻变强，而是先让自己不要继续被情绪拖着走。"
                    "一个很小的动作也算恢复：说出来、记录下来、深呼吸一次、暂停自责一分钟，都算。"
                    "今天先不要用最终结果评价自己，只看你有没有比刚才多一点点稳定。只要多一点点，就已经是在往回走了。"
                )
        else:
            if positive:
                focus_clause = "" if focus == "your voice message." else f" {focus}"
                empathy = (
                    f"[Emotion validation] I hear that something feels good right now.{focus_clause} "
                    f"That positive feeling matters; it can show what supports you, restores you, or gives you energy. {signal_text}{reference}"
                )
                cognitive = (
                    "[Cognitive strengthening] Let us preserve this moment instead of rushing past it. Notice three clues: what happened, "
                    "what you felt in your body, and what need or value was being met. This helps your mind remember that your life contains resources, not only problems."
                )
                behavioral = (
                    "[Action plan] Write one sentence about what made you happy. Then do one small action that keeps the feeling alive, "
                    "such as playing a song you like, taking a short walk, or sharing the good news with a safe person."
                )
                motivation = (
                    "[Motivation] You are allowed to enjoy this. Recovery is not only surviving difficult moments; it is also receiving good moments when they arrive. "
                    "Let this happiness count as real evidence that steadiness and warmth can come back."
                )
            else:
                empathy = (
                    f"[Emotion validation] I hear the feeling in your message. {focus} This sounds connected to {issue_type}, "
                    f"and it deserves more than a generic reassurance. {issue_hint}{signal_text}"
                    f"The fact that you put it into words means part of you is still trying to protect and understand yourself. {reference}"
                )
                cognitive = (
                    "[Cognitive restructuring] Let us separate three layers: what happened, what your mind says it means, "
                    "and what emotion follows from that meaning. A painful fact is real, but conclusions like “I always fail” or "
                    "“nothing will improve” are often threat-based interpretations. Ask two questions: what evidence supports the harshest thought, "
                    "and what evidence slightly softens it? The goal is not forced positivity; it is giving your mind another route."
                )
                behavioral = (
                    "[Action plan] Start with body stabilization before problem solving. Try three rounds of breathing: inhale for 4, pause for 2, exhale for 6. "
                    "Then write the hardest sentence in your mind and turn it into one small action for today. If possible, message one safe person: "
                    "“I am not doing great. Could you stay with me for ten minutes?” If not, drink water, wash your face, and step away from the screen for five minutes."
                )
                motivation = (
                    "[Motivation] You do not have to become strong all at once. Speaking, pausing, breathing, and reducing self-blame for one minute all count as recovery. "
                    "For today, do not judge yourself by the final outcome. Look for one percent more steadiness than before; that is already movement."
                )
        return empathy, cognitive, behavioral, motivation

    def _focus_sentence(self, text: str, language: str) -> str:
        text = " ".join(text.split())
        if language == "en" and self._contains_cjk(text):
            return "your voice message."
        if len(text) <= 80:
            return f"“{text}”"
        return f"“{text[:77]}...”"

    def _contains_cjk(self, text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _is_positive_mood(self, text: str, issue_type: str) -> bool:
        lowered = text.lower()
        return issue_type == "positive mood" or any(
            word in lowered
            for word in ["开心", "高兴", "快乐", "幸福", "兴奋", "happy", "glad", "joy", "excited", "great"]
        )

    def _issue_hint(self, issue_type: str, language: str) -> str:
        zh = {
            "positive mood": "积极情绪说明你正在接触到支持感、满足感或恢复中的能量。",
            "academic anxiety": "学业压力常常会把一次结果放大成对自我价值的判断。",
            "romantic breakup": "亲密关系的失落会同时触发悲伤、愤怒、不甘和自我怀疑。",
            "workplace stress": "工作压力容易让人把疲惫误解成能力不足。",
            "family issues": "家庭困扰会格外消耗，因为它常常牵动安全感和责任感。",
            "interpersonal conflict": "人际冲突最伤人的地方，往往是既想被理解又害怕继续受伤。",
        }
        en = {
            "positive mood": "Positive emotion can point to support, satisfaction, or returning energy.",
            "academic anxiety": "Academic stress can make one result feel like a judgment on your worth.",
            "romantic breakup": "Relationship loss can bring grief, anger, longing, and self-doubt at the same time.",
            "workplace stress": "Work stress can make exhaustion feel like personal inadequacy.",
            "family issues": "Family pain is draining because it touches safety, loyalty, and responsibility.",
            "interpersonal conflict": "Conflict hurts because you may want understanding while also fearing more pain.",
        }
        return (zh if language == "zh" else en).get(issue_type, "")

    def _describe_signals(self, language: str, sentiment: Optional[Dict], facial: Optional[Dict]) -> str:
        parts = []
        if sentiment and sentiment.get("available"):
            parts.append(f"文本情绪模型判断为 {sentiment.get('sentiment')}（{sentiment.get('percentage')}%）"
                         if language == "zh"
                         else f"The text sentiment model reads this as {sentiment.get('sentiment')} ({sentiment.get('percentage')}%).")
        if facial and facial.get("available") and facial.get("faces"):
            top = facial["faces"][0]
            parts.append(f"面部表情模型检测到 {top.get('emotion')} 表情。"
                         if language == "zh"
                         else f"The facial model detected a {top.get('emotion')} expression.")
        if not parts:
            return "" if language == "zh" else ""
        return "；".join(parts) + "。" if language == "zh" else " ".join(parts)

    def _short_reference(self, language: str, rag_context: str) -> str:
        if not rag_context:
            return ""
        return "我也会结合知识库里相关的心理支持方法来回应你。" if language == "zh" else "I will also ground the response in relevant support strategies from the knowledge base."


class LocalDialogueModelAdapter:
    """Optional trainable local dialogue model.

    Configure ``local_dialogue_model_path`` with a Hugging Face compatible
    causal language model directory. If it is not configured, the app falls
    back to the lightweight responder above.
    """

    def __init__(self, config: LocalModelConfig):
        self.config = config
        self._tokenizer = None
        self._model = None

    def generate(self, user_text: str, language: str, issue_type: str, rag_context: str,
                 sentiment: Optional[Dict], facial: Optional[Dict]) -> Optional[Tuple[str, str, str, str]]:
        if not self.config.local_dialogue_model_path:
            return None

        try:
            tokenizer, model = self._load()
            import torch

            prompt = self._build_prompt(user_text, language, issue_type, rag_context, sentiment, facial)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=700,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            if not text:
                return None
            sections = self._split_sections(text)
            return tuple(sections)  # type: ignore[return-value]
        except Exception:
            return None

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_path = project_path(self.config.local_dialogue_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if not torch.cuda.is_available():
            self._model.to("cpu")
        self._model.eval()
        return self._tokenizer, self._model

    def _build_prompt(self, user_text, language, issue_type, rag_context, sentiment, facial):
        if language == "zh":
            return (
                "你是本地运行的情感恢复AI助手。请严格输出四段：情感支持、认知重构、行动计划、激励。\n"
                f"问题类型: {issue_type}\n文本情绪: {sentiment}\n表情情绪: {facial}\n参考资料: {rag_context[:1200]}\n"
                f"用户: {user_text}\n回复:"
            )
        return (
            "You are a local emotional recovery assistant. Output exactly four sections: Emotional Support, "
            "Cognitive Restructuring, Action Plan, Motivation.\n"
            f"Issue type: {issue_type}\nText sentiment: {sentiment}\nFacial emotion: {facial}\nReference: {rag_context[:1200]}\n"
            f"User: {user_text}\nResponse:"
        )

    def _split_sections(self, text: str) -> List[str]:
        chunks = [chunk.strip() for chunk in text.replace("\r", "").split("\n\n") if chunk.strip()]
        if len(chunks) >= 4:
            return chunks[:4]
        while len(chunks) < 4:
            chunks.append("")
        return chunks


class LocalModelManager:
    """Facade used by the GUI."""

    def __init__(self, config: Optional[LocalModelConfig] = None):
        self.config = config or LocalModelConfig.load()
        self.sentiment = SentimentModelAdapter(self.config)
        self.face = FacialEmotionAdapter(self.config)
        self.voice = MiniOmniVoiceAdapter(self.config)
        self.dialogue = LocalDialogueModelAdapter(self.config)
        self.responder = EmotionalRecoveryResponder()

    def analyze_text(self, text: str) -> Dict:
        return self.sentiment.classify(text)

    def analyze_faces(self, image_paths: List[str]) -> Dict:
        for image_path in image_paths:
            result = self.face.analyze_image(image_path)
            if result.get("available") and result.get("faces"):
                return result
        return {"available": False, "faces": []}

    def respond(self, **kwargs) -> Tuple[str, str, str, str]:
        if kwargs.get("language") not in ("zh", "en"):
            kwargs["language"] = "en" if str(kwargs.get("language")).lower().startswith("en") else "zh"
        generated = self.dialogue.generate(**kwargs)
        if generated:
            return generated
        return self.responder.respond(**kwargs)
