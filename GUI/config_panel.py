"""
Configuration panel for local multimodal assistants.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.local_models import LocalModelConfig, LocalModelManager


class ConfigPanel(QWidget):
    """Right-side panel for local model configuration and assistant mode."""

    config_changed = pyqtSignal()
    local_models_updated = pyqtSignal(object)
    rag_toggled = pyqtSignal(bool)
    language_changed = pyqtSignal(str)
    mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_language = "zh"
        self.enable_rag = True
        self.mode = "text"
        self.rag = None
        self.model_config = LocalModelConfig.load()
        self.local_models = None

        self.init_ui()
        self.initialize_local_models(show_message=False)

    def init_ui(self):
        self.setStyleSheet("""
            QLabel { color: #333333; }
            QComboBox {
                background-color: #ffffff;
                color: #222222;
                border: 1px solid #cfcfcf;
                border-radius: 3px;
                padding: 4px 8px;
                min-height: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #222222;
                selection-background-color: #2f80ed;
                selection-color: #ffffff;
                outline: 0;
            }
            QGroupBox {
                color: #333333;
                font-weight: bold;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)

        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #ffffff;")
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(14)
        layout.setContentsMargins(15, 15, 15, 15)

        self.language_group = QGroupBox()
        lang_layout = QVBoxLayout(self.language_group)
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["中文", "English"])
        self.lang_combo.currentTextChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.lang_combo)
        layout.addWidget(self.language_group)

        self.mode_group = QGroupBox()
        mode_layout = QVBoxLayout(self.mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("文本助手", "text")
        self.mode_combo.addItem("语音助手", "voice")
        self.mode_combo.addItem("视频助手", "video")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        layout.addWidget(self.mode_group)

        self.rag_group = QGroupBox()
        rag_layout = QVBoxLayout(self.rag_group)
        self.rag_checkbox = QCheckBox()
        self.rag_checkbox.setChecked(True)
        self.rag_checkbox.stateChanged.connect(self.on_rag_changed)
        self.rag_info = QLabel()
        self.rag_info.setWordWrap(True)
        rag_layout.addWidget(self.rag_checkbox)
        rag_layout.addWidget(self.rag_info)
        layout.addWidget(self.rag_group)

        self.status_group = QGroupBox()
        status_layout = QVBoxLayout(self.status_group)
        self.status_label = QLabel()
        self.rag_status_label = QLabel()
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.rag_status_label)
        layout.addWidget(self.status_group)

        layout.addStretch()
        scroll_area.setWidget(main_widget)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)
        self.update_ui_texts()

    def update_ui_texts(self):
        self.mode_combo.blockSignals(True)
        if self.current_language == "zh":
            self.mode_combo.setItemText(0, "文本助手")
            self.mode_combo.setItemText(1, "语音助手")
            self.mode_combo.setItemText(2, "视频助手")
            self.language_group.setTitle("语言")
            self.mode_group.setTitle("功能")
            self.rag_group.setTitle("知识库 (RAG)")
            self.status_group.setTitle("状态")
            self.rag_checkbox.setText("启用知识库检索")
            self.rag_info.setText("用于把心理学知识补充到回复中。")
        else:
            self.mode_combo.setItemText(0, "Text Assistant")
            self.mode_combo.setItemText(1, "Voice Assistant")
            self.mode_combo.setItemText(2, "Video Assistant")
            self.language_group.setTitle("Language")
            self.mode_group.setTitle("Assistant Mode")
            self.rag_group.setTitle("Knowledge Base (RAG)")
            self.status_group.setTitle("Status")
            self.rag_checkbox.setText("Enable knowledge retrieval")
            self.rag_info.setText("Adds psychology knowledge context to responses.")
        self.mode_combo.blockSignals(False)
        self.update_status()
        self.update_rag_status()

    def set_runtime_status(self, message: str, busy: bool = True):
        """Show first-use model loading and runtime status in the side panel."""
        if not hasattr(self, "status_label"):
            return
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #f59e0b;" if busy else "color: #4CAF50;")

    def update_status(self):
        if self.local_models:
            self.status_label.setText("内置多模态功能已就绪" if self.current_language == "zh" else "Built-in multimodal features ready")
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setText("功能初始化中" if self.current_language == "zh" else "Initializing features")
            self.status_label.setStyleSheet("color: #f44336;")

    def update_rag_status(self):
        if not hasattr(self, "rag_status_label"):
            return
        if self.rag and hasattr(self.rag, "is_ready") and self.rag.is_ready:
            stats = self.rag.get_stats()
            text = f"RAG: 已启用 ({stats['total_entries']} 条)" if self.current_language == "zh" else f"RAG: Enabled ({stats['total_entries']} entries)"
        else:
            text = "RAG: 等待加载" if self.current_language == "zh" else "RAG: Waiting"
        self.rag_status_label.setText(text)

    def on_language_changed(self, text):
        self.current_language = "zh" if text == "中文" else "en"
        from core.language_manager import set_language

        set_language(self.current_language)
        self.update_ui_texts()
        self.language_changed.emit(self.current_language)

    def on_mode_changed(self):
        self.mode = self.mode_combo.currentData()
        self.mode_changed.emit(self.mode)

    def on_rag_changed(self, state):
        self.enable_rag = state == Qt.Checked
        self.rag_toggled.emit(self.enable_rag)

    def initialize_local_models(self, show_message=True):
        self.model_config.save()
        self.local_models = LocalModelManager(self.model_config)
        self.local_models_updated.emit(self.local_models)
        self.update_status()

    def get_config(self):
        return {
            "enable_rag": self.enable_rag,
            "rag": self.rag,
            "language": self.current_language,
            "mode": self.mode,
            "local_models": self.local_models,
        }
