"""
配置面板 - 管理 API Key、模型选择等设置
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QCheckBox, QGroupBox,
    QPushButton, QMessageBox, QScrollArea  # 添加 QScrollArea
)
from PyQt5.QtCore import pyqtSignal, Qt
from core import build_agents
from core.knowledge_base import RAGKnowledgeBase


class ConfigPanel(QWidget):
    """配置面板组件"""

    # 信号
    config_changed = pyqtSignal()
    agents_updated = pyqtSignal(object)
    rag_toggled = pyqtSignal(bool)
    language_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 状态
        self.api_key = ""
        self.model_choice = "gemini"
        self.enable_rag = True
        self.agents = None
        self.rag = None
        self.current_language = "zh"

        self.init_ui()
        self.init_rag()

    def init_ui(self):
        """初始化界面"""
        # 设置整体样式
        self.setStyleSheet("""
            QLabel {
                color: #333333;
            }
            QGroupBox {
                color: #333333;
                font-weight: bold;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QCheckBox {
                color: #333333;
            }
        """)

        # 使用 QScrollArea 包装
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        scroll_area.setStyleSheet("background-color: transparent;")

        # 主容器 - 设置背景色
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #ffffff;")
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 模型配置组
        self.model_group = QGroupBox()
        self.model_group.setObjectName("model_group")
        self.model_group.setStyleSheet("""
            QGroupBox {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
        """)
        model_layout = QVBoxLayout(self.model_group)
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(15, 15, 15, 15)

        # 语言选择
        lang_layout = QHBoxLayout()
        self.lang_label = QLabel()
        self.lang_label.setStyleSheet("color: #333333; font-weight: bold; min-width: 60px;")
        self.lang_combo = QComboBox()
        self.lang_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                color: #333333;
            }
            QComboBox:hover {
                border-color: #86b7fe;
            }
        """)
        self.lang_combo.addItems(["中文", "English"])
        self.lang_combo.currentTextChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.lang_label)
        lang_layout.addWidget(self.lang_combo)
        model_layout.addLayout(lang_layout)

        # 模型选择
        model_select_layout = QHBoxLayout()
        self.model_label = QLabel()
        self.model_label.setStyleSheet("color: #333333; font-weight: bold; min-width: 60px;")
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                color: #333333;
            }
            QComboBox:hover {
                border-color: #86b7fe;
            }
        """)
        self.model_combo.addItems(["gemini", "openai", "claude", "deepseek"])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_select_layout.addWidget(self.model_label)
        model_select_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_select_layout)

        # API Key 输入
        api_layout = QVBoxLayout()
        self.api_label = QLabel()
        self.api_label.setStyleSheet("color: #333333; font-weight: bold;")
        self.api_input = QLineEdit()
        self.api_input.setEchoMode(QLineEdit.Password)
        self.api_input.setPlaceholderText("请输入 API Key")
        self.api_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 8px;
                background-color: white;
                color: #333333;
            }
            QLineEdit:focus {
                border-color: #86b7fe;
                outline: none;
            }
        """)
        self.api_input.textChanged.connect(self.on_api_key_changed)
        api_layout.addWidget(self.api_label)
        api_layout.addWidget(self.api_input)
        model_layout.addLayout(api_layout)

        # API Key 链接
        self.link_label = QLabel()
        self.link_label.setStyleSheet("color: #0d6efd; font-size: 11px;")
        model_layout.addWidget(self.link_label)

        layout.addWidget(self.model_group)

        # RAG 配置组
        self.rag_group = QGroupBox()
        self.rag_group.setObjectName("rag_group")
        self.rag_group.setStyleSheet("""
            QGroupBox {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
        """)
        rag_layout = QVBoxLayout(self.rag_group)
        rag_layout.setSpacing(10)
        rag_layout.setContentsMargins(15, 15, 15, 15)

        self.rag_checkbox = QCheckBox()
        self.rag_checkbox.setChecked(True)
        self.rag_checkbox.setStyleSheet("color: #333333;")
        self.rag_checkbox.stateChanged.connect(self.on_rag_changed)
        rag_layout.addWidget(self.rag_checkbox)

        self.rag_info = QLabel()
        self.rag_info.setWordWrap(True)
        self.rag_info.setStyleSheet("color: #6c757d; font-size: 11px;")
        rag_layout.addWidget(self.rag_info)

        layout.addWidget(self.rag_group)

        # 状态显示组
        self.status_group = QGroupBox()
        self.status_group.setObjectName("status_group")
        self.status_group.setStyleSheet("""
            QGroupBox {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #333333;
            }
        """)
        status_layout = QVBoxLayout(self.status_group)
        status_layout.setSpacing(8)
        status_layout.setContentsMargins(15, 15, 15, 15)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #ff9800;")
        status_layout.addWidget(self.status_label)

        self.model_status_label = QLabel()
        self.model_status_label.setStyleSheet("color: #333333;")
        status_layout.addWidget(self.model_status_label)

        self.rag_status_label = QLabel()
        self.rag_status_label.setStyleSheet("color: #333333;")
        status_layout.addWidget(self.rag_status_label)

        layout.addWidget(self.status_group)

        # 初始化按钮
        self.init_btn = QPushButton()
        self.init_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #0a58ca;
            }
        """)
        self.init_btn.clicked.connect(self.initialize_agents)
        layout.addWidget(self.init_btn)

        layout.addStretch()

        scroll_area.setWidget(main_widget)

        # 将滚动区域添加到主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

        # 更新所有文本
        self.update_ui_texts()

    def update_ui_texts(self):
        """更新界面文本（根据当前语言）"""
        if self.current_language == "zh":
            # 中文模式
            self.lang_label.setText("语言:")
            self.model_label.setText("AI 模型:")
            self.api_label.setText("API Key:")
            self.rag_checkbox.setText("启用知识库检索")
            self.rag_info.setText("💡 知识库包含：共情技巧、CBT方法、压力管理策略等")
            self.status_label.setText("⚪ 未连接")
            self.init_btn.setText("🚀 初始化 Agent")

            # 分组框标题
            self.model_group.setTitle("🤖 模型配置")
            self.rag_group.setTitle("📚 知识库 (RAG)")
            self.status_group.setTitle("📊 状态")
        else:
            # 英文模式
            self.lang_label.setText("Language:")
            self.model_label.setText("AI Model:")
            self.api_label.setText("API Key:")
            self.rag_checkbox.setText("Enable RAG")
            self.rag_info.setText("💡 Knowledge base includes: empathy techniques, CBT methods, stress management")
            self.status_label.setText("⚪ Not connected")
            self.init_btn.setText("🚀 Initialize Agent")

            # 分组框标题
            self.model_group.setTitle("🤖 Model Configuration")
            self.rag_group.setTitle("📚 Knowledge Base (RAG)")
            self.status_group.setTitle("📊 Status")

        self.update_status()
        self.update_rag_status()

    def init_rag(self):
        """初始化 RAG 知识库"""
        try:
            self.rag = RAGKnowledgeBase()
            self.rag.load()
            self.update_rag_status()
        except Exception as e:
            self.rag = None
            self.update_rag_status()

    def update_rag_status(self):
        """更新RAG状态显示"""
        if self.current_language == "zh":
            if self.rag and hasattr(self.rag, 'is_ready') and self.rag.is_ready:
                stats = self.rag.get_stats()
                self.rag_status_label.setText(f"RAG: 已启用 ✅ ({stats['total_entries']} 条)")
            else:
                self.rag_status_label.setText("RAG: 不可用 ❌")
        else:
            if self.rag and hasattr(self.rag, 'is_ready') and self.rag.is_ready:
                stats = self.rag.get_stats()
                self.rag_status_label.setText(f"RAG: Enabled ✅ ({stats['total_entries']} entries)")
            else:
                self.rag_status_label.setText("RAG: Unavailable ❌")

    def update_link_label(self):
        """更新 API Key 获取链接"""
        links = {
            "gemini": "https://makersuite.google.com/app/apikey",
            "openai": "https://platform.openai.com/api-keys",
            "claude": "https://console.anthropic.com/settings/keys",
            "deepseek": "https://platform.deepseek.com/api-keys"
        }
        if self.current_language == "zh":
            self.link_label.setText(
                f'<a href="{links[self.model_choice]}" style="color: #2196F3;">'
                f'获取 {self.model_choice.upper()} API Key →</a>'
            )
        else:
            self.link_label.setText(
                f'<a href="{links[self.model_choice]}" style="color: #2196F3;">'
                f'Get {self.model_choice.upper()} API Key →</a>'
            )
        self.link_label.setOpenExternalLinks(True)

    def on_language_changed(self, text: str):
        """语言变化"""
        self.current_language = "zh" if text == "中文" else "en"
        from core.language_manager import set_language
        set_language(self.current_language)
        self.agents = None
        self.update_ui_texts()
        self.update_link_label()
        self.update_status()
        self.update_rag_status()

        # 发射语言变化信号
        self.language_changed.emit(self.current_language)

    def on_model_changed(self, text: str):
        """模型变化"""
        self.model_choice = text
        self.update_link_label()
        self.agents = None
        self.update_status()

    def on_api_key_changed(self, text: str):
        """API Key 变化"""
        self.api_key = text
        self.agents = None
        self.update_status()

    def on_rag_changed(self, state):
        """RAG 开关变化"""
        self.enable_rag = (state == Qt.Checked)
        self.rag_toggled.emit(self.enable_rag)

    def update_status(self):
        """更新状态显示"""
        if self.current_language == "zh":
            if self.api_key and self.agents is not None:
                self.status_label.setText("✅ 已连接")
                self.status_label.setStyleSheet("color: #4CAF50;")
            elif self.api_key:
                self.status_label.setText("🟡 已配置 API Key，点击初始化")
                self.status_label.setStyleSheet("color: #ff9800;")
            else:
                self.status_label.setText("🔴 未配置 API Key")
                self.status_label.setStyleSheet("color: #f44336;")

            self.model_status_label.setText(f"模型: {self.model_choice.upper()}")
        else:
            if self.api_key and self.agents is not None:
                self.status_label.setText("✅ Connected")
                self.status_label.setStyleSheet("color: #4CAF50;")
            elif self.api_key:
                self.status_label.setText("🟡 API Key configured, click initialize")
                self.status_label.setStyleSheet("color: #ff9800;")
            else:
                self.status_label.setText("🔴 API Key not configured")
                self.status_label.setStyleSheet("color: #f44336;")

            self.model_status_label.setText(f"Model: {self.model_choice.upper()}")

    def initialize_agents(self):
        """初始化 Agent"""
        if not self.api_key:
            if self.current_language == "zh":
                QMessageBox.warning(self, "提示", "请先输入 API Key")
            else:
                QMessageBox.warning(self, "Warning", "Please enter your API Key first")
            return

        try:
            if self.current_language == "zh":
                self.status_label.setText("⏳ 正在初始化...")
            else:
                self.status_label.setText("⏳ Initializing...")
            self.status_label.setStyleSheet("color: #2196F3;")

            self.agents = build_agents(self.api_key, self.model_choice, self.current_language)

            self.update_status()

            # 传递配置给聊天组件
            config = self.get_config()
            if self.parent() and hasattr(self.parent(), 'chat_widget'):
                self.parent().chat_widget.set_config(
                    api_key=config['api_key'],
                    model_choice=config['model_choice'],
                    enable_rag=config['enable_rag'],
                    rag=config['rag'],
                    agents=self.agents,
                    language=self.current_language
                )

            self.agents_updated.emit(self.agents)

            if self.current_language == "zh":
                QMessageBox.information(self, "成功", f"Agent 初始化成功！\n模型: {self.model_choice.upper()}")
            else:
                QMessageBox.information(self, "Success", f"Agent initialized successfully!\nModel: {self.model_choice.upper()}")

        except Exception as e:
            if self.current_language == "zh":
                QMessageBox.critical(self, "错误", f"初始化失败:\n{str(e)}")
                self.status_label.setText("❌ 初始化失败")
            else:
                QMessageBox.critical(self, "Error", f"Initialization failed:\n{str(e)}")
                self.status_label.setText("❌ Initialization failed")
            self.status_label.setStyleSheet("color: #f44336;")

    def get_config(self):
        """获取当前配置"""
        return {
            'api_key': self.api_key,
            'model_choice': self.model_choice,
            'enable_rag': self.enable_rag,
            'rag': self.rag,
            'agents': self.agents,
            'language': self.current_language
        }