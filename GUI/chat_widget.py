"""
聊天组件 - 负责消息的显示和输入（支持打字机效果）
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QScrollArea, QLabel, QFileDialog,
    QMessageBox, QProgressDialog, QFrame, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from pathlib import Path
import time
import tempfile

from GUI.typewriter_label import TypewriterLabel


class LocalResponseWorker(QThread):
    """Generate a response with local multimodal adapters."""

    finished = pyqtSignal(tuple, dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, local_models, user_input: str, issue_type: str,
                 rag_context: str, language: str, image_paths=None, audio_path="", mode="text",
                 known_transcript=""):
        super().__init__()
        self.user_input = user_input
        self.issue_type = issue_type
        self.rag_context = rag_context
        self.language = language
        self.local_models = local_models
        self.image_paths = image_paths or []
        self.audio_path = audio_path
        self.mode = mode
        self.known_transcript = known_transcript

    def run(self):
        try:
            transcript = self.known_transcript
            voice_output = {}
            if self.audio_path and not transcript:
                self.progress.emit("正在加载并识别语音模型..." if self.language == "zh" else "Loading and running speech recognition model...")
                voice_result = self.local_models.voice.transcribe(self.audio_path, target_language=self.language)
                transcript = voice_result.get("text", "")
            if transcript:
                self.user_input = transcript if not self.user_input.strip() else self.user_input.strip()

            self.progress.emit("正在加载并识别文本情绪模型..." if self.language == "zh" else "Loading and running text sentiment model...")
            sentiment = self.local_models.analyze_text(self.user_input)

            facial = {"available": False, "faces": []}
            if self.image_paths:
                self.progress.emit("正在加载并识别表情模型..." if self.language == "zh" else "Loading and running facial emotion model...")
                facial = self.local_models.analyze_faces(self.image_paths)

            responses = self.local_models.respond(
                user_text=self.user_input,
                language=self.language,
                issue_type=self.issue_type,
                rag_context=self.rag_context,
                sentiment=sentiment,
                facial=facial,
            )

            if self.mode in ("voice", "video"):
                spoken_text = "\n\n".join(responses)
                self.progress.emit("正在加载并生成语音回复..." if self.language == "zh" else "Loading and generating voice reply...")
                voice_output = self.local_models.voice.synthesize(spoken_text[:800], language=self.language)

            meta = {
                "sentiment": sentiment,
                "facial": facial,
                "transcript": transcript,
                "voice_output": voice_output,
                "mode": self.mode,
                "language": self.language,
                "input_audio_path": self.audio_path,
                "rag_context": self.rag_context,
            }
            self.finished.emit(tuple(responses), meta)
        except Exception as e:
            self.error.emit(str(e))


class TranscribeWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, local_models, audio_path: str, target_language: str = "zh"):
        super().__init__()
        self.local_models = local_models
        self.audio_path = audio_path
        self.target_language = target_language

    def run(self):
        try:
            result = self.local_models.voice.transcribe(self.audio_path, target_language=self.target_language)
            if result.get("available"):
                self.finished.emit(result.get("text", ""))
            else:
                self.error.emit(result.get("error", "Transcription failed"))
        except Exception as e:
            self.error.emit(str(e))


class FaceEmotionWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self, local_models, image_path: str):
        super().__init__()
        self.local_models = local_models
        self.image_path = image_path

    def run(self):
        try:
            self.finished.emit(self.local_models.face.analyze_image(self.image_path))
        except Exception as e:
            self.finished.emit({"available": False, "faces": [], "error": str(e)})


class ChatWidget(QWidget):
    """聊天组件"""

    message_sent = pyqtSignal(str)
    message_received = pyqtSignal(str, str)
    runtime_status = pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 状态变量
        self.uploaded_images = []
        self.uploaded_audio = ""
        self.send_worker = None
        self.local_models = None
        self.enable_rag = True
        self.rag = None
        self.language = "zh"
        self.mode = "text"
        self.current_user_input = ""
        self.current_issue_type = "general"
        self.current_exchange = None
        self.conversation_records = []
        self.camera = None
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.last_camera_frame = ""
        self.face_timer = QTimer()
        self.face_timer.timeout.connect(self.detect_current_face_emotion)
        self.face_worker = None
        self.response_audio_path = ""
        self.is_recording = False
        self.audio_stream = None
        self.audio_chunks = []
        self.record_sample_rate = 16000
        self.record_start_time = 0.0
        self.recorded_audio_duration = 0
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_recording_ui)
        self.transcribe_worker = None
        self.welcome_label = None
        self.current_voice_bubble = None
        self.current_voice_transcript_label = None
        self.voice_bubbles = []
        self.latest_voice_transcript = ""
        self.video_turns = []
        self.audio_player = QMediaPlayer(self)
        self.audio_player.stateChanged.connect(self.on_audio_player_state_changed)
        self.current_audio_path = ""
        self.current_audio_button = None
        self.video_call_active = False
        self.face_model_loading = False
        self.face_model_loaded = False

        # 打字机效果开关
        self.typewriter_enabled = True
        self.typewriter_delay = 20  # 毫秒/字符

        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 消息显示区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #fafafa; border: none;")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setAlignment(Qt.AlignTop)
        self.messages_layout.setSpacing(10)
        self.messages_layout.setContentsMargins(10, 10, 10, 10)

        self.scroll_area.setWidget(self.messages_widget)
        layout.addWidget(self.scroll_area)

        # 语音助手区域
        self.voice_widget = QWidget()
        self.voice_widget.setVisible(False)
        self.voice_widget.setStyleSheet("background-color: #ffffff;")
        voice_layout = QVBoxLayout(self.voice_widget)
        voice_layout.setAlignment(Qt.AlignTop)
        voice_layout.setContentsMargins(36, 18, 36, 18)
        voice_layout.setSpacing(8)
        self.voice_logo = QLabel("🎙")
        self.voice_logo.setAlignment(Qt.AlignCenter)
        self.voice_logo.setStyleSheet("color: #3f4652; font-size: 92px;")
        self.voice_title = QLabel()
        self.voice_title.setAlignment(Qt.AlignCenter)
        self.voice_title.setStyleSheet("color: #1f2933; font-size: 24px; font-weight: bold;")
        self.voice_hint = QLabel()
        self.voice_hint.setAlignment(Qt.AlignCenter)
        self.voice_hint.setStyleSheet("color: #52616b; font-size: 14px;")
        self.voice_record_btn = QPushButton()
        self.voice_record_btn.setFixedSize(150, 44)
        self.voice_record_btn.setStyleSheet("""
            QPushButton {
                background-color: #2f80ed;
                color: white;
                border: none;
                border-radius: 22px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1f6fd1; }
        """)
        self.voice_record_btn.clicked.connect(self.toggle_recording)
        self.voice_record_btn.setVisible(False)
        self.voice_transcript = QLabel()
        self.voice_transcript.setWordWrap(True)
        self.voice_transcript.setAlignment(Qt.AlignCenter)
        self.voice_transcript.setStyleSheet("color: #25313b; font-size: 14px;")
        self.voice_transcript.setVisible(False)

        self.voice_bubble_area = QScrollArea()
        self.voice_bubble_area.setWidgetResizable(True)
        self.voice_bubble_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.voice_bubble_area.setMinimumHeight(260)
        self.voice_bubble_area.setStyleSheet("""
            QScrollArea {
                background-color: #f8fafc;
                border: 2px solid #d6dee8;
                border-radius: 14px;
            }
            QScrollBar:vertical {
                background: #eef2f6;
                width: 12px;
                border-radius: 6px;
            }
        """)
        self.voice_bubble_widget = QWidget()
        self.voice_bubble_widget.setStyleSheet("background-color: #f8fafc;")
        self.voice_bubble_layout = QVBoxLayout(self.voice_bubble_widget)
        self.voice_bubble_layout.setAlignment(Qt.AlignTop)
        self.voice_bubble_layout.setContentsMargins(18, 18, 18, 18)
        self.voice_bubble_layout.setSpacing(10)
        self.voice_bubble_area.setWidget(self.voice_bubble_widget)
        voice_layout.addWidget(self.voice_logo)
        voice_layout.addWidget(self.voice_title)
        voice_layout.addWidget(self.voice_hint)
        voice_layout.addWidget(self.voice_record_btn, alignment=Qt.AlignCenter)
        voice_layout.addWidget(self.voice_transcript)
        voice_layout.addWidget(self.voice_bubble_area, 1)
        layout.addWidget(self.voice_widget)

        # 图片预览区域
        self.preview_widget = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_widget)
        self.preview_layout.setContentsMargins(5, 5, 5, 5)
        self.preview_widget.setMaximumHeight(100)
        self.preview_widget.setVisible(False)
        layout.addWidget(self.preview_widget)

        # 视频助手区域
        self.video_widget = QWidget()
        self.video_widget.setVisible(False)
        self.video_widget.setStyleSheet("background-color: #f5f7fb;")
        video_layout = QVBoxLayout(self.video_widget)
        video_layout.setContentsMargins(18, 18, 18, 18)
        video_layout.setSpacing(12)

        video_top_layout = QHBoxLayout()
        video_top_layout.setSpacing(16)

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(330, 230)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera")
        self.camera_label.setStyleSheet("background-color: #111827; color: white; border-radius: 10px;")

        ai_panel = QFrame()
        ai_panel.setMinimumSize(330, 230)
        ai_panel.setStyleSheet("background-color: #eaf1ff; border: 1px solid #d7e2f3; border-radius: 10px;")
        ai_layout = QVBoxLayout(ai_panel)
        ai_layout.setContentsMargins(8, 8, 8, 8)
        self.ai_video_label = QLabel()
        self.ai_video_label.setAlignment(Qt.AlignCenter)
        self.ai_video_label.setWordWrap(True)
        self.ai_video_label.setStyleSheet("color: #344054; font-size: 16px; font-weight: bold;")
        ai_layout.addWidget(self.ai_video_label)

        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(8)
        camera_layout.addWidget(self.camera_label)
        self.video_emotion_label = QLabel()
        self.video_emotion_label.setAlignment(Qt.AlignCenter)
        self.video_emotion_label.setStyleSheet("color: #344054; font-size: 14px; padding: 4px;")
        camera_layout.addWidget(self.video_emotion_label)

        video_top_layout.addWidget(camera_panel, 1)
        video_top_layout.addWidget(ai_panel, 1)
        video_layout.addLayout(video_top_layout, 2)

        self.video_conversation = QTextEdit()
        self.video_conversation.setReadOnly(True)
        self.video_conversation.setPlaceholderText("视频通话内容会显示在这里")
        self.video_conversation.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: #1f2937;
                border: 1px solid #d7e2f3;
                border-radius: 10px;
                padding: 12px;
                font-size: 15px;
                line-height: 1.5;
            }
        """)
        video_layout.addWidget(self.video_conversation, 3)
        layout.addWidget(self.video_widget)

        # 输入区域
        self.input_widget = QWidget()
        input_layout = QHBoxLayout(self.input_widget)
        input_layout.setContentsMargins(5, 5, 5, 5)

        # 上传按钮
        self.upload_btn = QPushButton("📎")
        self.upload_btn.setToolTip("上传图片")
        self.upload_btn.setFixedSize(40, 40)
        self.upload_btn.clicked.connect(self.upload_images)
        input_layout.addWidget(self.upload_btn)

        self.audio_btn = QPushButton("🎙")
        self.audio_btn.setToolTip("开始/停止录音")
        self.audio_btn.setFixedSize(120, 44)
        self.audio_btn.setStyleSheet("""
            QPushButton {
                background-color: #e53935;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #c62828; }
        """)
        self.audio_btn.clicked.connect(self.toggle_recording)
        input_layout.addWidget(self.audio_btn)

        self.camera_btn = QPushButton("关闭摄像头")
        self.camera_btn.setToolTip("打开摄像头")
        self.camera_btn.setFixedSize(130, 44)
        self.camera_btn.setStyleSheet("""
            QPushButton {
                background-color: #455a64;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #263238; }
        """)
        self.camera_btn.clicked.connect(self.toggle_camera)
        input_layout.addWidget(self.camera_btn)

        # 文本输入
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("分享你的感受...")
        self.input_text.setMaximumHeight(100)
        self.input_text.setMinimumHeight(60)
        self.input_text.setStyleSheet("border: 1px solid #ddd; border-radius: 8px; padding: 8px;")
        input_layout.addWidget(self.input_text)

        # 发送按钮
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedWidth(80)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        layout.addWidget(self.input_widget)

        # 设置拉伸因子（在添加所有控件之后）
        layout.setStretchFactor(self.scroll_area, 1)  # 消息区域可拉伸
        layout.setStretchFactor(self.voice_widget, 1)
        layout.setStretchFactor(self.video_widget, 1)
        layout.setStretchFactor(self.preview_widget, 0)
        layout.setStretchFactor(self.input_widget, 0)

        self.update_language(self.language)

        # 添加欢迎消息（只添加一次）
        self.add_welcome_message()
        self.set_mode(self.mode)

    def add_welcome_message(self):
        """添加欢迎消息（根据语言）- 只添加一次，不清空现有消息"""
        if self.welcome_label is not None:
            self.update_welcome_message()
            return
        if self.language == "zh":
            self.welcome_label = self.add_message("AI 助手", "你好！我是你的情感恢复 AI 助手。\n\n请分享你的感受，我会为你提供专业、温暖的支持。")
        else:
            self.welcome_label = self.add_message(
                "AI Assistant",
                "Hello! I'm your emotional recovery AI assistant.\n\nPlease share your feelings, and I will provide professional, warm support."
            )

    def update_language(self, language: str):
        """更新语言设置"""
        language_changed = language != self.language
        self.language = language
        if self.language == "zh":
            self.upload_btn.setToolTip("上传图片")
            self.audio_btn.setToolTip("开始/停止录音")
            self.camera_btn.setToolTip("打开/关闭摄像头")
            if self.camera_timer.isActive():
                self.camera_btn.setText("关闭摄像头")
            else:
                self.camera_btn.setText("打开摄像头")
            self.input_text.setPlaceholderText("分享你的感受...")
            self.send_btn.setText("发送")
            self.voice_title.setText("语音助手")
            self.voice_hint.setText("点击开始录音，说完后再次点击停止")
            self.voice_record_btn.setText("开始录音")
            if self.mode == "video":
                self.audio_btn.setText("🎙 说话")
                self.video_conversation.setPlaceholderText("视频通话内容会显示在这里")
            if not self.camera_timer.isActive():
                self.update_camera_off_view()
            if not self.face_worker or not self.face_worker.isRunning():
                self.video_emotion_label.setText("表情识别：等待中")
        else:
            self.upload_btn.setToolTip("Upload image")
            self.audio_btn.setToolTip("Start/stop recording")
            self.camera_btn.setToolTip("Toggle camera")
            if self.camera_timer.isActive():
                self.camera_btn.setText("Camera Off")
            else:
                self.camera_btn.setText("Camera On")
            self.input_text.setPlaceholderText("Share your feelings...")
            self.send_btn.setText("Send")
            self.voice_title.setText("Voice Assistant")
            self.voice_hint.setText("Click to record. Click again to stop.")
            self.voice_record_btn.setText("Start recording")
            if self.mode == "video":
                self.audio_btn.setText("🎙 Speak")
                self.video_conversation.setPlaceholderText("Video call transcript will appear here")
            if not self.camera_timer.isActive():
                self.update_camera_off_view()
            if not self.face_worker or not self.face_worker.isRunning():
                self.video_emotion_label.setText("Facial emotion: waiting")

        self.update_voice_bubble_language()
        self.update_welcome_message()
        self.render_video_conversation()
        self.update_audio_button_text()
        if language_changed and self.mode == "text" and getattr(self, "conversation_records", None):
            self.rebuild_conversation_for_language()

    def update_audio_button_text(self):
        if not hasattr(self, "audio_btn") or self.is_recording:
            return
        if self.mode == "video":
            self.audio_btn.setText("🎙 说话" if self.language == "zh" else "🎙 Speak")
        elif self.mode == "voice":
            self.audio_btn.setText("🎙 录音" if self.language == "zh" else "🎙 Record")

    def localized_transcribe_text(self, state: str = "ready") -> str:
        if state == "working":
            return "转换中" if self.language == "zh" else "Working"
        return "转文字" if self.language == "zh" else "Transcribe"

    def localized_transcribe_error(self) -> str:
        return (
            "转文字失败，请检查语音模型依赖。"
            if self.language == "zh"
            else "Transcription failed. Check voice model dependencies."
        )

    def update_voice_bubble_language(self):
        for bubble in getattr(self, "voice_bubbles", []):
            button = bubble.get("button")
            label = bubble.get("label")
            status = bubble.get("status", "ready")
            if button:
                button.setText(self.localized_transcribe_text("working" if status == "working" else "ready"))
            if label and status == "failed":
                label.setText(self.localized_transcribe_error())
            elif label and status == "done":
                if not bubble.get("is_user") and bubble.get("responses_by_lang"):
                    responses_by_lang = bubble.setdefault("responses_by_lang", {})
                    responses = responses_by_lang.get(self.language)
                    if responses is None and self.local_models is not None:
                        responses = self.local_models.respond(
                            user_text=self.display_text_for_language(bubble.get("source_text", "")),
                            language=self.language,
                            issue_type=bubble.get("issue_type", "general"),
                            rag_context=bubble.get("rag_context", ""),
                            sentiment=bubble.get("meta", {}).get("sentiment"),
                            facial=bubble.get("meta", {}).get("facial"),
                        )
                        responses_by_lang[self.language] = tuple(responses)
                    if responses:
                        label.setText("\n\n".join(responses))
                        continue
                text_by_lang = bubble.setdefault("text_by_lang", {})
                text = text_by_lang.get(self.language)
                if not text:
                    source = text_by_lang.get("zh") or text_by_lang.get("en") or bubble.get("text", "")
                    text = self.display_text_for_language(source)
                    if text:
                        text_by_lang[self.language] = text
                if text:
                    label.setText(text)

    def current_voice_response_text(self, record: dict) -> str:
        responses_by_lang = record.setdefault("responses_by_lang", {})
        responses = responses_by_lang.get(self.language)
        if responses is None and self.local_models is not None:
            responses = self.local_models.respond(
                user_text=self.display_text_for_language(record.get("source_text", "")),
                language=self.language,
                issue_type=record.get("issue_type", "general"),
                rag_context=record.get("rag_context", ""),
                sentiment=record.get("meta", {}).get("sentiment"),
                facial=record.get("meta", {}).get("facial"),
            )
            responses_by_lang[self.language] = tuple(responses)
        return "\n\n".join(responses or ())

    def has_cjk(self, text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")

    def display_text_for_language(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return text
        if self.language == "en" and self.has_cjk(text):
            return self.zh_to_en_display(text)
        if self.language == "zh" and not self.has_cjk(text):
            return self.en_to_zh_display(text)
        return text

    def zh_to_en_display(self, text: str) -> str:
        pairs = [
            (("开心", "高兴", "快乐", "很好"), "I feel very happy."),
            (("难过", "伤心", "痛苦"), "I feel sad."),
            (("压力", "累", "崩溃"), "I feel under a lot of pressure."),
            (("焦虑", "紧张", "担心"), "I feel anxious."),
            (("生气", "愤怒"), "I feel angry."),
            (("分手", "失恋"), "I am dealing with a breakup."),
            (("考试", "没考好"), "I am worried about my studies or exam results."),
        ]
        for keys, value in pairs:
            if any(key in text for key in keys):
                return value
        return "Voice message."

    def en_to_zh_display(self, text: str) -> str:
        lower = (text or "").lower()
        pairs = [
            (("happy", "glad", "good", "joy"), "我很开心。"),
            (("sad", "upset", "depressed"), "我很难过。"),
            (("stress", "pressure", "overwhelmed", "tired"), "我压力很大。"),
            (("anxious", "anxiety", "worried", "nervous"), "我很焦虑。"),
            (("angry", "mad"), "我很生气。"),
            (("breakup", "relationship"), "我正在经历关系困扰。"),
            (("exam", "study", "school", "academic"), "我担心学习或考试。"),
        ]
        for keys, value in pairs:
            if any(key in lower for key in keys):
                return value
        return "语音内容。"

    def localized_emotion_name(self, emotion: str) -> str:
        if self.language != "zh":
            return emotion or ""
        mapping = {
            "neutral": "平静",
            "happy": "开心",
            "sad": "难过",
            "angry": "生气",
            "surprise": "惊讶",
            "surprised": "惊讶",
            "fear": "害怕",
            "fearful": "害怕",
            "disgust": "厌恶",
            "disgusted": "厌恶",
        }
        return mapping.get((emotion or "").lower(), emotion or "")

    def update_welcome_message(self):
        """更新欢迎消息（切换语言时替换现有欢迎消息）"""
        if self.welcome_label is None:
            return
        if self.language == "zh":
            sender = "AI 助手"
            message = "你好！我是你的情感恢复 AI 助手。\n\n请分享你的感受，我会为你提供专业、温暖的支持。"
        else:
            sender = "AI Assistant"
            message = "Hello! I'm your emotional recovery AI assistant.\n\nPlease share your feelings, and I will provide professional, warm support."
        self.welcome_label.setText(self.format_message_html(sender, message, is_user=False))

    def clear_message_widgets(self):
        """Clear visible text chat widgets without deleting conversation records."""
        while self.messages_layout.count():
            child = self.messages_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.welcome_label = None

    def rebuild_conversation_for_language(self):
        """Re-render previous exchanges without re-running response generation."""
        if not getattr(self, "conversation_records", None):
            return

        self.clear_message_widgets()
        self.add_welcome_message()

        for record in self.conversation_records:
            display_input = self.display_text_for_language(record.get("display_input", "") or record.get("source_text", ""))
            self.add_message("You" if self.language == "en" else "你", display_input, is_user=True)
            responses = self.cached_responses_for_language(record)
            self.add_response_sections(tuple(responses), animate=False)

    def cached_responses_for_language(self, record: dict) -> tuple:
        """Return current-language responses from cache, projecting existing text if needed."""
        responses_by_lang = record.setdefault("responses_by_lang", {})
        responses = responses_by_lang.get(self.language)
        if responses:
            return tuple(responses)

        source_responses = responses_by_lang.get("zh") or responses_by_lang.get("en") or ()
        responses = self.project_response_for_language(record, tuple(source_responses))
        responses_by_lang[self.language] = tuple(responses)
        return tuple(responses)

    def project_response_for_language(self, record: dict, source_responses: tuple = ()) -> tuple:
        """Fast display-only language projection used during language switching."""
        source_text = record.get("source_text", "") or record.get("display_input", "")
        focus = self.display_text_for_language(source_text)
        issue_type = record.get("issue_type", "general")
        meta = record.get("meta", {}) or {}
        sentiment = (meta.get("sentiment") or {}).get("sentiment")
        facial = meta.get("facial") or {}
        face = ""
        if facial.get("available") and facial.get("faces"):
            face = self.localized_emotion_name(facial["faces"][0].get("emotion"))

        if self.language == "zh":
            signal = ""
            if sentiment:
                signal += f"文本情绪识别为 {sentiment}。"
            if face:
                signal += f"面部表情识别为 {face}。"
            return (
                f"【情绪确认】我听到你说的是：{focus or '这段感受'}。这份感受值得被认真看见和接住。{signal}",
                "【认知巩固】我们先把它分成三层：发生了什么、你如何理解它、它带来了什么情绪。这样做不是否定感受，而是让大脑多一个出口。",
                "【行动计划】先做一个很小的步骤：深呼吸一次，把最关键的一句话写下来，再选择一个今天能完成的小动作。",
                "【继续前进】你不需要马上解决所有问题。能把感受说出来、停下来照顾自己一点点，就已经是在恢复。"
            )

        signal = ""
        if sentiment:
            signal += f"The text sentiment model reads this as {sentiment}. "
        if face:
            signal += f"The facial model detected {face}. "
        return (
            f"[Emotion validation] I hear you saying: {focus or 'this feeling'}. This deserves to be seen and handled with care. {signal}",
            "[Cognitive restructuring] Let us separate three layers: what happened, how your mind interpreted it, and what emotion followed. This does not dismiss the feeling; it gives your mind another route.",
            "[Action plan] Start with one small step: take one slow breath, write the hardest sentence down, and choose one action you can complete today.",
            "[Motivation] You do not have to solve everything immediately. Naming the feeling and caring for yourself a little already counts as recovery."
        )

    def rerender_conversation_records(self):
        """Backward-compatible alias for older language-switch code paths."""
        self.rebuild_conversation_for_language()

    def add_response_sections(self, responses: tuple, animate: bool = True):
        if self.language == "zh":
            titles = ["💝 情感支持", "🧠 认知重构", "📋 行动计划", "💪 激励"]
        else:
            titles = ["💝 Emotional Support", "🧠 Cognitive Restructuring", "📋 Action Plan", "💪 Motivation"]

        labels = []
        for title, content in zip(titles, responses):
            labels.append(self.add_response_section(title, content, animate=animate))
        return labels

    def set_config(self, enable_rag: bool, rag, local_models, language: str = "zh", mode: str = "text"):
        """设置配置"""
        self.enable_rag = enable_rag
        self.rag = rag
        self.local_models = local_models

        self.update_language(language)
        self.set_mode(mode)

    def set_mode(self, mode: str):
        """Switch assistant mode."""
        if mode != self.mode:
            self.stop_audio_playback()
            if self.is_recording:
                self.stop_recording(auto_send=False)
        self.mode = mode
        is_voice = mode in ("voice", "video")
        is_video = mode == "video"
        is_text = mode == "text"
        self.scroll_area.setVisible(is_text)
        self.voice_widget.setVisible(mode == "voice")
        self.video_widget.setVisible(is_video)
        self.audio_btn.setVisible(is_voice)
        self.camera_btn.setVisible(is_video)
        self.upload_btn.setVisible(is_text)
        self.input_text.setVisible(is_text)
        self.send_btn.setVisible(not is_video)
        if is_video:
            self.load_ai_avatar()
            self.update_audio_button_text()
            self.video_conversation.setPlaceholderText(
                "视频通话内容会显示在这里" if self.language == "zh" else "Video call transcript will appear here"
            )
            self.update_video_intro_language()
            self.start_camera()
            if not self.face_timer.isActive():
                self.face_timer.start(1800)
        else:
            if self.face_timer.isActive():
                self.face_timer.stop()
            self.stop_camera()
            self.update_audio_button_text()

    def load_ai_avatar(self):
        avatar_path = Path(__file__).resolve().parent.parent / "images" / "ai_video_avatar.png"
        if avatar_path.exists():
            pixmap = QPixmap(str(avatar_path))
            if not pixmap.isNull():
                self.ai_video_label.setPixmap(pixmap.scaled(
                    self.ai_video_label.width() or 360,
                    self.ai_video_label.height() or 260,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                ))
                return
        self.ai_video_label.setText("AI")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "mode", "") == "video":
            self.load_ai_avatar()

    def append_video_conversation(self, speaker: str, text: str):
        if not hasattr(self, "video_conversation") or not text:
            return
        current = self.video_conversation.toPlainText().strip()
        line = f"{speaker}: {text.strip()}"
        self.video_conversation.setPlainText(f"{current}\n\n{line}" if current else line)
        self.video_conversation.verticalScrollBar().setValue(
            self.video_conversation.verticalScrollBar().maximum()
        )

    def update_video_intro_language(self):
        if not hasattr(self, "video_conversation"):
            return
        self.render_video_conversation()

    def video_intro_text(self) -> str:
        if self.language == "zh":
            return "AI：你好呀～我是你的 AI 情感恢复助手，很高兴见到你！"
        return "AI: Hi there. I am your emotional recovery AI assistant. I am glad to see you."

    def add_video_turn(self, role: str, text: str = "", audio_path: str = "",
                       responses_by_lang=None, source_text: str = "", issue_type: str = "general",
                       rag_context: str = "", meta=None, text_language: str = ""):
        lang = text_language or self.language
        turn = {
            "role": role,
            "text_by_lang": {lang: text.strip()} if text else {},
            "audio_path": audio_path,
            "responses_by_lang": responses_by_lang or {},
            "source_text": source_text or text,
            "issue_type": issue_type,
            "rag_context": rag_context,
            "meta": meta or {},
        }
        self.video_turns.append(turn)
        self.render_video_conversation()
        return turn

    def get_video_turn_text(self, turn: dict) -> str:
        if turn.get("role") == "ai":
            responses_by_lang = turn.setdefault("responses_by_lang", {})
            responses = responses_by_lang.get(self.language)
            if responses is None and self.local_models is not None:
                responses = self.local_models.respond(
                    user_text=self.display_text_for_language(turn.get("source_text", "")),
                    language=self.language,
                    issue_type=turn.get("issue_type", "general"),
                    rag_context=turn.get("rag_context", ""),
                    sentiment=turn.get("meta", {}).get("sentiment"),
                    facial=turn.get("meta", {}).get("facial"),
                )
                responses_by_lang[self.language] = tuple(responses)
            if responses:
                return "\n\n".join(responses)

        text_by_lang = turn.setdefault("text_by_lang", {})
        text = text_by_lang.get(self.language)
        if text:
            return text
        source = text_by_lang.get("zh") or text_by_lang.get("en") or turn.get("source_text", "")
        text = self.display_text_for_language(source)
        text_by_lang[self.language] = text
        return text

    def render_video_conversation(self):
        if not hasattr(self, "video_conversation"):
            return
        lines = [self.video_intro_text()]
        for turn in getattr(self, "video_turns", []):
            text = self.get_video_turn_text(turn).strip()
            if not text:
                continue
            if turn.get("role") == "user":
                speaker = "我" if self.language == "zh" else "Me"
            else:
                speaker = "AI"
            sep = "：" if self.language == "zh" else ": "
            lines.append(f"{speaker}{sep}{text}")
        self.video_conversation.setPlainText("\n\n".join(lines))
        self.video_conversation.verticalScrollBar().setValue(
            self.video_conversation.verticalScrollBar().maximum()
        )

    def add_message(self, sender: str, message: str, is_user: bool = False):
        """添加消息到聊天区域（普通消息，不需要打字机效果）"""
        html = self.format_message_html(sender, message, is_user)
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        self.messages_layout.addWidget(label)

        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))
        return label

    def format_message_html(self, sender: str, message: str, is_user: bool = False):
        """Format a chat message as rich text."""
        message = message.replace('\n', '<br>')

        if is_user:
            color = "#4CAF50"
            bg_color = "#4CAF50"
            text_color = "white"
            align = "right"
        else:
            color = "#2196F3"
            bg_color = "#f0f0f0"
            text_color = "#333333"
            align = "left"

        html = f"""
        <div style="text-align: {align}; margin: 10px 0;">
            <div style="display: inline-block; background-color: {bg_color}; color: {text_color}; 
                        padding: 8px 12px; border-radius: 12px; max-width: 80%; text-align: left;">
                <b style="color: {color};">{sender}</b><br>{message}
            </div>
        </div>
        """
        return html

    def add_response_section(self, title: str, content: str, animate: bool = True):
        """添加回复分区；新回复使用打字机，重绘历史时静态显示。"""
        if self.language == "zh":
            colors = {
                "💝 情感支持": "#4CAF50",
                "🧠 认知重构": "#2196F3",
                "📋 行动计划": "#FF9800",
                "💪 激励": "#9C27B0"
            }
        else:
            colors = {
                "💝 Emotional Support": "#4CAF50",
                "🧠 Cognitive Restructuring": "#2196F3",
                "📋 Action Plan": "#FF9800",
                "💪 Motivation": "#9C27B0"
            }

        bg_color = colors.get(title, "#2196F3")

        # 创建容器
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            background-color: {bg_color}; 
            color: white; 
            padding: 8px 12px; 
            font-weight: bold;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        """)
        title_label.setFixedHeight(35)
        container_layout.addWidget(title_label)

        # 内容
        content_label = TypewriterLabel()
        content_label.setStyleSheet(
            "background-color: white; padding: 12px; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;")
        container_layout.addWidget(content_label)

        # 添加到布局
        self.messages_layout.addWidget(container)

        # 滚动到底部
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

        if animate:
            content_label.start_typewriter(content, self.typewriter_delay)
            content_label.mousePressEvent = lambda e: content_label.skip_to_end()
        else:
            content_label.setText(content.replace('\n', '<br>'))

        return content_label

    def upload_images(self):
        """上传图片"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片" if self.language == "zh" else "Select Images",
            "", "Image files (*.png *.jpg *.jpeg *.bmp)"
        )

        for file_path in files:
            self.uploaded_images.append(file_path)
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                preview = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label = QLabel()
                label.setPixmap(preview)
                label.setToolTip(file_path)
                self.preview_layout.addWidget(label)

        if self.uploaded_images:
            self.preview_widget.setVisible(True)
            msg = f"已上传 {len(files)} 张图片" if self.language == "zh" else f"Uploaded {len(files)} images"
            self.add_message("System", msg)

    def toggle_recording(self):
        """Start or stop microphone recording."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        try:
            import sounddevice as sd

            self.audio_chunks = []
            self.record_start_time = time.time()
            self.recorded_audio_duration = 0
            self.uploaded_audio = ""
            self.latest_voice_transcript = ""
            self.voice_transcript.setText("")
            self.stop_audio_playback()

            def callback(indata, frames, time_info, status):
                self.audio_chunks.append(indata.copy())

            self.audio_stream = sd.InputStream(
                samplerate=self.record_sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self.audio_stream.start()
            self.is_recording = True
            self.record_timer.start(250)
            self.update_recording_ui()
        except Exception as e:
            if "sounddevice" in str(e):
                msg = (
                    "无法开始录音：当前环境没有安装 sounddevice。\n请在 DL 环境运行：pip install sounddevice soundfile"
                    if self.language == "zh"
                    else "Cannot start recording: sounddevice is not installed.\nRun in the DL environment: pip install sounddevice soundfile"
                )
            else:
                msg = (
                    f"无法开始录音：{e}\n请确认麦克风权限已开启。"
                    if self.language == "zh"
                    else f"Cannot start recording: {e}\nPlease allow microphone access."
                )
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)

    def stop_recording(self, auto_send: bool = True):
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            self.record_timer.stop()
            self.is_recording = False

            if not self.audio_chunks:
                return
            import numpy as np
            import soundfile as sf

            audio = np.concatenate(self.audio_chunks, axis=0)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            sf.write(tmp.name, audio, self.record_sample_rate)
            self.uploaded_audio = tmp.name
            self.recorded_audio_duration = max(1, int(time.time() - self.record_start_time))
            if self.mode == "voice" and auto_send:
                self.add_voice_bubble(self.recorded_audio_duration, audio_path=self.uploaded_audio, is_user=True)
            elif self.mode == "video" and auto_send:
                QTimer.singleShot(80, self.send_message)
            self.update_recording_ui()
        except Exception as e:
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", str(e))

    def update_recording_ui(self):
        if self.is_recording:
            elapsed = int(time.time() - self.record_start_time)
            if self.mode == "video":
                text = f"结束说话 {elapsed}s" if self.language == "zh" else f"End Speaking {elapsed}s"
            else:
                text = f"停止录音 {elapsed}s" if self.language == "zh" else f"Stop {elapsed}s"
            self.voice_record_btn.setText(text)
            self.audio_btn.setText(text)
            if self.mode == "voice":
                self.voice_hint.setText("正在录音..." if self.language == "zh" else "Recording...")
        else:
            self.voice_record_btn.setText("开始录音" if self.language == "zh" else "Start recording")
            if self.mode == "video":
                self.audio_btn.setText("🎙 说话" if self.language == "zh" else "🎙 Speak")
            else:
                self.audio_btn.setText("🎙 录音" if self.language == "zh" else "🎙 Record")
            if self.uploaded_audio and self.mode == "voice":
                text = f"录音完成：{self.recorded_audio_duration}s" if self.language == "zh" else f"Recorded: {self.recorded_audio_duration}s"
                self.voice_hint.setText(text)

    def add_voice_bubble(self, duration: int, audio_path: str, is_user: bool, transcript: str = "",
                         responses_by_lang=None, source_text: str = "", issue_type: str = "general",
                         rag_context: str = "", meta=None):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(10)
        row_layout.setAlignment(Qt.AlignRight if is_user else Qt.AlignLeft)

        play_btn = QPushButton(f"▶ {duration}\"")
        play_btn.setFixedSize(190, 44)
        play_btn.setStyleSheet("""
            QPushButton {
                background-color: #2f343a;
                color: white;
                border: none;
                border-radius: 8px;
                text-align: left;
                padding-left: 18px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3b424a; }
        """)
        play_btn.clicked.connect(lambda: self.toggle_audio_playback(audio_path, play_btn))

        transcribe_btn = QPushButton(self.localized_transcribe_text())
        transcribe_btn.setFixedSize(112, 34)
        transcribe_btn.setStyleSheet("""
            QPushButton {
                background-color: #eceff1;
                color: #263238;
                border: 1px solid #cfd8dc;
                border-radius: 16px;
            }
            QPushButton:hover { background-color: #dfe6ea; }
        """)

        transcript_label = QLabel(transcript)
        transcript_label.setWordWrap(True)
        transcript_label.setMaximumWidth(480)
        transcript_label.setStyleSheet("color: #37474f; padding: 6px; line-height: 1.35;")
        transcript_label.setVisible(bool(transcript))

        column = QWidget()
        column.setMaximumWidth(520)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.addWidget(play_btn)
        column_layout.addWidget(transcript_label)

        if is_user:
            row_layout.addStretch()
            row_layout.addWidget(column)
            row_layout.addWidget(transcribe_btn)
        else:
            row_layout.addWidget(transcribe_btn)
            row_layout.addWidget(column)
            row_layout.addStretch()

        bubble_record = {
            "button": transcribe_btn,
            "label": transcript_label,
            "is_user": is_user,
            "audio_path": audio_path,
            "status": "done" if transcript else "ready",
            "text": transcript,
            "text_by_lang": {self.language: transcript} if transcript else {},
            "play_button": play_btn,
            "responses_by_lang": responses_by_lang or {},
            "source_text": source_text,
            "issue_type": issue_type,
            "rag_context": rag_context,
            "meta": meta or {},
        }
        self.voice_bubbles.append(bubble_record)
        transcribe_btn.clicked.connect(
            lambda: self.transcribe_audio_to_label(audio_path, transcribe_btn, transcript_label, is_user, bubble_record)
        )
        self.voice_bubble_layout.addWidget(row)
        QTimer.singleShot(100, lambda: self.voice_bubble_area.verticalScrollBar().setValue(
            self.voice_bubble_area.verticalScrollBar().maximum()
        ))
        if is_user:
            self.current_voice_bubble = row
            self.current_voice_transcript_label = transcript_label
        return row

    def transcribe_recorded_audio(self):
        if not self.uploaded_audio:
            return
        if self.local_models is None:
            return
        self.transcribe_audio_to_label(self.uploaded_audio, None, self.current_voice_transcript_label, True)

    def transcribe_audio_to_label(self, audio_path: str, button, label, is_user: bool, record=None):
        if not audio_path or self.local_models is None:
            return
        if button:
            button.setEnabled(False)
            button.setText(self.localized_transcribe_text("working"))
        if record:
            record["status"] = "working"
        self.pending_transcribe_button = button
        self.pending_transcribe_label = label
        self.pending_transcribe_is_user = is_user
        self.pending_transcribe_record = record
        self.transcribe_worker = TranscribeWorker(self.local_models, audio_path, self.language)
        self.transcribe_worker.finished.connect(self.on_transcript_ready)
        self.transcribe_worker.error.connect(self.on_transcript_error)
        self.transcribe_worker.start()

    def on_transcript_ready(self, text: str):
        button = getattr(self, "pending_transcribe_button", None)
        label = getattr(self, "pending_transcribe_label", None)
        record = getattr(self, "pending_transcribe_record", None)
        if button:
            button.setEnabled(True)
            button.setText(self.localized_transcribe_text())
        display_text = text
        if record and not record.get("is_user") and record.get("responses_by_lang"):
            display_text = self.current_voice_response_text(record) or text
        if label:
            label.setText(display_text)
            label.setVisible(bool(display_text))
        if record:
            record["status"] = "done"
            record["text"] = text
            record.setdefault("text_by_lang", {})[self.language] = text
        if getattr(self, "pending_transcribe_is_user", False):
            self.latest_voice_transcript = text

    def on_transcript_error(self, error: str):
        button = getattr(self, "pending_transcribe_button", None)
        label = getattr(self, "pending_transcribe_label", None)
        record = getattr(self, "pending_transcribe_record", None)
        if button:
            button.setEnabled(True)
            button.setText(self.localized_transcribe_text())
        message = self.localized_transcribe_error()
        if label:
            label.setText(message)
            label.setVisible(True)
        if record:
            record["status"] = "failed"
            record["text"] = ""
        self.voice_transcript.setText("")
        self.runtime_status.emit(message, False)

    def toggle_camera(self):
        if self.camera_timer.isActive():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera and self.camera_timer.isActive():
            return
        try:
            import cv2

            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.camera.isOpened():
                self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Cannot open camera")
            self.camera_timer.start(60)
            self.camera_btn.setText("关闭摄像头" if self.language == "zh" else "Camera Off")
            self.camera_label.setText("")
            self.last_camera_frame = ""
            if hasattr(self, "video_emotion_label"):
                self.video_emotion_label.setText("表情识别：等待中" if self.language == "zh" else "Facial emotion: waiting")
            if self.mode == "video" and not self.face_timer.isActive():
                self.face_timer.start(1800)
        except Exception as e:
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", str(e))

    def stop_camera(self):
        if self.face_timer.isActive():
            self.face_timer.stop()
        if self.camera_timer.isActive():
            self.camera_timer.stop()
        if self.camera:
            self.camera.release()
            self.camera = None
        if hasattr(self, "camera_btn"):
            self.camera_btn.setText("打开摄像头" if self.language == "zh" else "Camera On")
        self.update_camera_off_view()
        self.last_camera_frame = ""

    def update_camera_off_view(self):
        if not hasattr(self, "camera_label"):
            return
        self.camera_label.clear()
        self.camera_label.setText("摄像头已关闭" if self.language == "zh" else "Camera is off")
        self.camera_label.setStyleSheet("background-color: #000; color: white; border-radius: 6px;")

    def update_camera_frame(self):
        if not self.camera:
            return
        ok, frame = self.camera.read()
        if not ok:
            return
        try:
            import cv2
            import tempfile

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.close()
            cv2.imwrite(tmp.name, frame)
            self.last_camera_frame = tmp.name
        except Exception:
            return

    def detect_current_face_emotion(self):
        if self.local_models is None or not self.last_camera_frame:
            return
        if self.face_worker is not None and self.face_worker.isRunning():
            return
        if not self.face_model_loaded and not self.face_model_loading:
            self.face_model_loading = True
            self.runtime_status.emit(
                "表情模型加载中..." if self.language == "zh" else "Loading facial emotion model...",
                True,
            )
        self.face_worker = FaceEmotionWorker(self.local_models, self.last_camera_frame)
        self.face_worker.finished.connect(self.on_face_emotion_detected)
        self.face_worker.start()

    def on_face_emotion_detected(self, result: dict):
        self.update_face_model_runtime_status(result)
        if result.get("available") and result.get("faces"):
            face = result["faces"][0]
            if face.get("classifier_error"):
                text = (
                    "表情识别：已检测到人脸（表情模型依赖未加载）"
                    if self.language == "zh"
                    else "Facial emotion: face detected (emotion model dependency missing)"
                )
                self.video_emotion_label.setText(text)
                return
            suffix = "（估计）" if face.get("estimated") and self.language == "zh" else (" (estimated)" if face.get("estimated") else "")
            emotion = self.localized_emotion_name(face.get("emotion"))
            text = (
                f"表情识别：{emotion} ({face.get('score'):.2f}){suffix}"
                if self.language == "zh"
                else f"Facial emotion: {emotion} ({face.get('score'):.2f}){suffix}"
            )
            self.video_emotion_label.setText(text)
        else:
            error = result.get("error")
            if error:
                short_error = str(error).splitlines()[-1][:80]
                text = (
                    f"表情识别：模型错误 - {short_error}"
                    if self.language == "zh"
                    else f"Facial emotion: model error - {short_error}"
                )
            else:
                text = "表情识别：未检测到人脸" if self.language == "zh" else "Facial emotion: no face detected"
            self.video_emotion_label.setText(text)

    def update_face_model_runtime_status(self, result: dict):
        """Keep detailed emotions in the camera area, not in the global status."""
        if self.face_model_loaded:
            return
        self.face_model_loading = False
        if result.get("error"):
            self.runtime_status.emit(
                "表情模型加载失败" if self.language == "zh" else "Facial emotion model failed to load",
                False,
            )
            return
        self.face_model_loaded = True
        self.runtime_status.emit(
            "表情模型加载完成" if self.language == "zh" else "Facial emotion model loaded",
            False,
        )

    def clear_preview(self):
        """清空图片预览"""
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.uploaded_images = []
        self.preview_widget.setVisible(False)

    def clear_chat(self):
        """清空聊天记录"""
        self.conversation_records = []
        self.video_turns = []
        self.current_exchange = None
        self.clear_message_widgets()
        self.render_video_conversation()

        if self.language == "zh":
            self.welcome_label = self.add_message("AI 助手", "聊天记录已清空。有什么我可以帮助你的吗？")
        else:
            self.welcome_label = self.add_message("AI Assistant", "Chat history cleared. How can I help you?")

    def add_thinking_message(self, text: str) -> QLabel:
        """添加"分析中..."消息，返回标签以便后续替换"""
        html = f"""
        <div style="margin: 10px 0;">
            <div style="display: inline-block; background-color: #f0f0f0; 
                        padding: 8px 12px; border-radius: 12px;">
                <b style="color: #FF9800;">🤔</b> {text}
            </div>
        </div>
        """
        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        self.messages_layout.addWidget(label)

        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

        return label

    def remove_thinking_message(self, thinking_label: QLabel):
        """移除"分析中..."消息"""
        if thinking_label:
            thinking_label.deleteLater()

    def send_message(self):
        """发送消息"""
        try:
            from core.language_manager import get_current_language
            current_language = get_current_language()
            if current_language != self.language:
                self.update_language(current_language)
        except Exception:
            pass

        user_input = self.input_text.toPlainText().strip()

        has_video_frame = self.mode == "video" and bool(self.last_camera_frame)
        if not user_input and not self.uploaded_images and not self.uploaded_audio and not has_video_frame:
            msg = "请分享你的感受、上传图片或语音" if self.language == "zh" else "Please share your feelings, upload images or audio"
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)
            return

        if self.mode in ("voice", "video") and not self.uploaded_audio:
            msg = "请先点击录音，说完后再次点击停止" if self.language == "zh" else "Please click record, speak, then click again to stop."
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)
            return

        if self.local_models is None:
            msg = "多模态功能仍在初始化，请稍后再试" if self.language == "zh" else "Multimodal features are still initializing. Please try again shortly."
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)
            return

        known_transcript = self.latest_voice_transcript.strip() if self.uploaded_audio else ""
        if known_transcript:
            known_transcript = self.display_text_for_language(known_transcript)

        # 保存当前用户输入用于历史记录
        if user_input:
            self.current_user_input = user_input
        elif known_transcript:
            self.current_user_input = known_transcript
        elif self.uploaded_audio:
            self.current_user_input = "[语音]" if self.language == "zh" else "[Voice]"
        elif has_video_frame:
            self.current_user_input = "[Video emotion check]"
        else:
            self.current_user_input = "[Uploaded images]"

        # 显示用户消息
        if self.mode == "text":
            self.add_message("You" if self.language == "en" else "你",
                             self.current_user_input, is_user=True)

        # 在聊天界面显示"分析中..."消息
        if self.language == "zh":
            thinking_msg = "🤔 分析中..."
        else:
            thinking_msg = "🤔 Thinking..."
        thinking_label = self.add_thinking_message(thinking_msg)

        # 处理输入
        final_input = user_input or known_transcript
        image_paths = list(self.uploaded_images)
        if self.mode == "video" and self.last_camera_frame:
            image_paths.append(self.last_camera_frame)
        audio_path = self.uploaded_audio

        # OCR text from uploaded images in text mode.
        if self.uploaded_images:
            try:
                from core.utils import ocr_image
                ocr_texts = []
                for file_path in self.uploaded_images:
                    with open(file_path, 'rb') as f:
                        text = ocr_image(f)
                        if text:
                            ocr_texts.append(f"【Image {Path(file_path).name}】\n{text}")
                if ocr_texts:
                    final_input = "\n\n".join(ocr_texts) + "\n\n" + user_input
            except Exception as e:
                print(f"OCR失败: {e}")

        # 清空输入和预览
        self.input_text.clear()
        self.clear_preview()
        self.uploaded_audio = ""
        self.latest_voice_transcript = ""

        # 分类问题类型
        try:
            from core.utils import classify_issue_type
            issue_type = classify_issue_type(final_input)
        except:
            issue_type = "general"

        self.current_issue_type = issue_type
        self.current_exchange = {
            "display_input": self.current_user_input,
            "source_text": final_input,
            "issue_type": issue_type,
            "rag_context": "",
            "meta": {},
            "responses_by_lang": {},
            "audio_path": audio_path,
        }

        # 从 FAISS 数据库检索知识
        rag_context = ""
        retrieved_items = []

        if self.enable_rag and self.rag and hasattr(self.rag, 'is_ready') and self.rag.is_ready:
            try:
                rag_msg = (
                    "正在加载向量模型并检索知识库..."
                    if self.language == "zh"
                    else "Loading vector model and searching knowledge base..."
                )
                self.on_worker_progress(rag_msg, thinking_label)
                self.runtime_status.emit(rag_msg, True)
                QApplication.processEvents()
                retrieved_items = self.rag.search(final_input, issue_type=issue_type, k=3)

                if retrieved_items:
                    rag_parts = []
                    for i, item in enumerate(retrieved_items):
                        rag_parts.append(
                            f"【参考{i + 1}】{item['title']}\n来源: {item['source']}\n内容: {item['content'][:500]}")
                        print(f"📚 RAG检索到: {item['title']} (相似度: {item['score']:.2f})")

                    rag_context = "\n\n".join(rag_parts)
                    if self.current_exchange is not None:
                        self.current_exchange["rag_context"] = rag_context
                    self.runtime_status.emit(
                        "RAG 已检索到相关心理学知识" if self.language == "zh" else "RAG retrieved relevant psychology knowledge",
                        False,
                    )
                    if self.language == "zh":
                        self.add_message("📚 知识库", "检索到相关心理学知识")
                    else:
                        self.add_message("📚 Knowledge Base", "Retrieved relevant psychology knowledge")
                else:
                    print("📚 RAG未检索到相关知识")
                    self.runtime_status.emit(
                        "RAG 未检索到足够相关的知识" if self.language == "zh" else "RAG found no sufficiently relevant knowledge",
                        False,
                    )
            except Exception as e:
                print(f"RAG搜索失败: {e}")
                self.runtime_status.emit(
                    "RAG 检索失败" if self.language == "zh" else "RAG search failed",
                    False,
                )

        # 启动工作线程
        self.send_worker = LocalResponseWorker(
            self.local_models, final_input, issue_type, rag_context,
            self.language, image_paths=image_paths, audio_path=audio_path, mode=self.mode,
            known_transcript=known_transcript
        )
        self.send_worker.progress.connect(lambda msg: self.on_worker_progress(msg, thinking_label))
        self.send_worker.finished.connect(lambda r, m: self.on_response_received(r, retrieved_items, thinking_label, m))
        self.send_worker.error.connect(lambda e: self.on_response_error(e, thinking_label))
        self.send_worker.start()

    def on_response_received(self, responses: tuple, retrieved_items: list, thinking_label: QLabel, meta: dict):
        """收到回复"""
        response_mode = meta.get("mode", self.mode)
        response_language = meta.get("language", self.language)
        # 移除"分析中..."消息
        self.remove_thinking_message(thinking_label)

        transcript = meta.get("transcript")
        if transcript and self.current_exchange is not None and not self.current_exchange.get("source_text"):
            self.current_exchange["source_text"] = transcript
            self.current_exchange["display_input"] = "[语音]" if self.language == "zh" else "[Voice]"
        if transcript and response_mode == "video":
            self.add_video_turn(
                "user",
                transcript,
                audio_path=meta.get("input_audio_path", ""),
                source_text=transcript,
                text_language=response_language,
            )
        elif transcript and response_mode == "text":
            self.add_message("ASR", transcript)

        sentiment = meta.get("sentiment") or {}
        if sentiment.get("available") and response_mode == "text":
            if self.language == "zh":
                self.add_message("文本情绪识别", f"{sentiment.get('sentiment')} ({sentiment.get('percentage')}%)")
            else:
                self.add_message("Text Sentiment", f"{sentiment.get('sentiment')} ({sentiment.get('percentage')}%)")

        facial = meta.get("facial") or {}
        if facial.get("available") and facial.get("faces") and response_mode == "text":
            face = facial["faces"][0]
            emotion = self.localized_emotion_name(face.get("emotion"))
            if self.language == "zh":
                self.add_message("面部表情识别", f"{emotion} ({face.get('score'):.2f})")
            else:
                self.add_message("Facial Emotion", f"{emotion} ({face.get('score'):.2f})")

        completed_exchange = self.current_exchange
        if completed_exchange is not None:
            completed_exchange["meta"] = meta
            completed_exchange.setdefault("responses_by_lang", {})[response_language] = tuple(responses)
            self.conversation_records.append(completed_exchange)
            self.current_exchange = None

        typewriter_labels = self.add_response_sections(responses) if response_mode == "text" else []

        if response_mode in ("voice", "video"):
            combined_preview = "\n\n".join(responses)
            if response_mode == "voice":
                self.voice_hint.setText("语音回复已生成，点击 AI 语音条播放" if self.language == "zh" else "Voice reply ready. Click the AI voice bubble to play.")
            else:
                self.add_video_turn(
                    "ai",
                    combined_preview,
                    responses_by_lang={response_language: tuple(responses)},
                    source_text=(completed_exchange or {}).get("source_text", transcript or self.current_user_input),
                    issue_type=(completed_exchange or {}).get("issue_type", self.current_issue_type),
                    rag_context=(completed_exchange or {}).get("rag_context", meta.get("rag_context", "")),
                    meta=meta,
                    text_language=response_language,
                )

        voice_output = meta.get("voice_output") or {}
        if voice_output.get("available"):
            path = voice_output.get("audio_path", "")
            if response_mode == "voice":
                self.add_voice_bubble(
                    self.get_audio_duration(path),
                    audio_path=path,
                    is_user=False,
                    transcript="",
                    responses_by_lang={response_language: tuple(responses)},
                    source_text=(completed_exchange or {}).get("source_text", transcript or self.current_user_input),
                    issue_type=(completed_exchange or {}).get("issue_type", self.current_issue_type),
                    rag_context=(completed_exchange or {}).get("rag_context", meta.get("rag_context", "")),
                    meta=meta,
                )
            elif response_mode == "video" and self.mode == "video":
                self.toggle_audio_playback(path)
            elif response_mode == "text":
                message = f"语音回复已生成: {path}" if self.language == "zh" else f"Voice reply generated: {path}"
                self.add_message("Audio", message)

        def check_all_finished():
            if not typewriter_labels or all(not label.is_typing() for label in typewriter_labels):
                combined = "\n\n".join(responses)
                self.message_received.emit("assistant", combined)
                self.save_history(combined)
            else:
                QTimer.singleShot(100, check_all_finished)

        QTimer.singleShot(100, check_all_finished)

    def on_worker_progress(self, message: str, thinking_label: QLabel):
        if thinking_label:
            thinking_label.setText(message)
        if self.mode in ("voice", "video"):
            self.voice_hint.setText(message)
        self.runtime_status.emit(message, True)

    def toggle_audio_playback(self, audio_path: str, button=None):
        """Play, pause, or resume a voice bubble."""
        if not audio_path:
            return
        try:
            if self.current_audio_path == audio_path and self.audio_player.state() == QMediaPlayer.PlayingState:
                self.audio_player.pause()
                if button:
                    button.setText(button.text().replace("⏸", "▶", 1))
                self.voice_hint.setText("已暂停语音回复" if self.language == "zh" else "Voice paused")
                return
            if self.current_audio_path == audio_path and self.audio_player.state() == QMediaPlayer.PausedState:
                self.audio_player.play()
                if button:
                    button.setText(button.text().replace("▶", "⏸", 1))
                self.voice_hint.setText("正在播放语音回复" if self.language == "zh" else "Playing voice reply")
                return

            self.stop_audio_playback()
            self.current_audio_path = audio_path
            self.current_audio_button = button
            self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))
            self.audio_player.play()
            if button:
                button.setText(button.text().replace("▶", "⏸", 1))
            self.voice_hint.setText("正在播放语音回复" if self.language == "zh" else "Playing voice reply")
        except Exception as e:
            print(f"音频播放失败: {e}")

    def stop_audio_playback(self):
        if getattr(self, "audio_player", None):
            self.audio_player.stop()
        if self.current_audio_button:
            text = self.current_audio_button.text()
            if text.startswith("⏸"):
                self.current_audio_button.setText(text.replace("⏸", "▶", 1))
        self.current_audio_path = ""
        self.current_audio_button = None

    def on_audio_player_state_changed(self, state):
        if state != QMediaPlayer.PlayingState and self.current_audio_button:
            text = self.current_audio_button.text()
            if text.startswith("⏸"):
                self.current_audio_button.setText(text.replace("⏸", "▶", 1))

    def get_audio_duration(self, audio_path: str) -> int:
        try:
            import soundfile as sf

            info = sf.info(audio_path)
            return max(1, int(round(info.frames / info.samplerate)))
        except Exception:
            return 1

    def on_response_error(self, error_msg: str, thinking_label: QLabel):
        """处理错误"""
        self.remove_thinking_message(thinking_label)

        msg = f"生成失败:\n{error_msg}" if self.language == "zh" else f"Generation failed:\n{error_msg}"
        self.runtime_status.emit("模型运行失败" if self.language == "zh" else "Model run failed", False)
        QMessageBox.critical(self, "错误" if self.language == "zh" else "Error", msg)

    def save_history(self, response: str):
        """保存对话历史"""
        try:
            import json
            from datetime import datetime

            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "input": self.current_user_input,
                "response": response,
                "issue_type": self.current_issue_type,
                "mode": self.mode,
                "language": self.language
            }

            # 保存到文件
            history_path = Path("./data/history/history.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)

            if history_path.exists():
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []

            history.append(history_entry)

            if len(history) > 100:
                history = history[-100:]

            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

            print(f"✅ 历史记录已保存: {len(history)} 条")

        except Exception as e:
            print(f"❌ 保存历史记录失败: {e}")
