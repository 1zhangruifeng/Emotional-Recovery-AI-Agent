"""
聊天组件 - 负责消息的显示和输入（支持打字机效果）
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QScrollArea, QLabel, QFileDialog,
    QMessageBox, QProgressDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QPixmap
from pathlib import Path

from GUI.typewriter_label import TypewriterLabel


class SendWorker(QThread):
    """发送消息的后台工作线程"""

    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, agents, user_input: str, issue_type: str,
                 rag_context: str, model_choice: str, api_key: str, language: str):
        super().__init__()
        self.agents = agents
        self.user_input = user_input
        self.issue_type = issue_type
        self.rag_context = rag_context
        self.model_choice = model_choice
        self.api_key = api_key
        self.language = language

    def run(self):
        try:
            empathy, cognitive, behavioral, motivational = self.agents
            responses = []

            # 根据语言设置提示词
            if self.language == "zh":
                prompt_e = f"问题类型: {self.issue_type}\n参考资料: {self.rag_context}\n用户: {self.user_input}\n请用中文提供共情回应，验证用户的情绪。"
                prompt_c = f"用户: {self.user_input}\n请用中文进行认知重构，帮助用户识别和挑战负面思维。"
                prompt_b = f"用户: {self.user_input}\n请用中文提供具体的行为建议和应对策略。"
                prompt_m = f"用户: {self.user_input}\n请用中文提供激励性对话，增强用户的自我效能感。"
            else:
                prompt_e = f"Issue type: {self.issue_type}\nReference: {self.rag_context}\nUser: {self.user_input}\nPlease respond in English with empathy, validating the user's emotions."
                prompt_c = f"User: {self.user_input}\nPlease respond in English with cognitive restructuring to help identify and challenge negative thoughts."
                prompt_b = f"User: {self.user_input}\nPlease respond in English with practical coping strategies and behavioral suggestions."
                prompt_m = f"User: {self.user_input}\nPlease respond in English with motivational dialogue to enhance self-efficacy."

            resp_e = empathy.run(input=prompt_e).content
            responses.append(resp_e)

            resp_c = cognitive.run(input=prompt_c).content
            responses.append(resp_c)

            resp_b = behavioral.run(input=prompt_b).content
            responses.append(resp_b)

            resp_m = motivational.run(input=prompt_m).content
            responses.append(resp_m)

            self.finished.emit(tuple(responses))
        except Exception as e:
            self.error.emit(str(e))


class ChatWidget(QWidget):
    """聊天组件"""

    message_sent = pyqtSignal(str)
    message_received = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 状态变量
        self.uploaded_images = []
        self.send_worker = None
        self.agents = None
        self.enable_rag = True
        self.rag = None
        self.model_choice = "gemini"
        self.api_key = ""
        self.language = "zh"
        self.current_user_input = ""
        self.current_issue_type = "general"

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

        # 图片预览区域
        self.preview_widget = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_widget)
        self.preview_layout.setContentsMargins(5, 5, 5, 5)
        self.preview_widget.setMaximumHeight(100)
        self.preview_widget.setVisible(False)
        layout.addWidget(self.preview_widget)

        # 输入区域
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(5, 5, 5, 5)

        # 上传按钮
        self.upload_btn = QPushButton("📎")
        self.upload_btn.setToolTip("上传图片")
        self.upload_btn.setFixedSize(40, 40)
        self.upload_btn.clicked.connect(self.upload_images)
        input_layout.addWidget(self.upload_btn)

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

        layout.addWidget(input_widget)

        # 设置拉伸因子（在添加所有控件之后）
        layout.setStretchFactor(self.scroll_area, 1)  # 消息区域可拉伸
        layout.setStretchFactor(self.preview_widget, 0)
        layout.setStretchFactor(input_widget, 0)

        # 添加欢迎消息（只添加一次）
        self.add_welcome_message()

    def add_welcome_message(self):
        """添加欢迎消息（根据语言）- 只添加一次，不清空现有消息"""
        if self.language == "zh":
            self.add_message("AI 助手", "你好！我是你的情感恢复 AI 助手。\n\n请分享你的感受，我会为你提供专业、温暖的支持。")
        else:
            self.add_message("AI Assistant",
                             "Hello! I'm your emotional recovery AI assistant.\n\nPlease share your feelings, and I will provide professional, warm support.")

    def update_language(self, language: str):
        """更新语言设置"""
        self.language = language
        if self.language == "zh":
            self.upload_btn.setToolTip("上传图片")
            self.input_text.setPlaceholderText("分享你的感受...")
            self.send_btn.setText("发送")
        else:
            self.upload_btn.setToolTip("Upload image")
            self.input_text.setPlaceholderText("Share your feelings...")
            self.send_btn.setText("Send")

        # 更新欢迎消息
        self.update_welcome_message()

    def update_welcome_message(self):
        """更新欢迎消息（切换语言时替换现有欢迎消息）"""
        # 找到并移除当前的欢迎消息（假设它是第一条消息）
        if self.messages_layout.count() > 0:
            first_item = self.messages_layout.takeAt(0)
            if first_item and first_item.widget():
                first_item.widget().deleteLater()

        # 添加新的欢迎消息
        if self.language == "zh":
            self.add_message("AI 助手", "你好！我是你的情感恢复 AI 助手。\n\n请分享你的感受，我会为你提供专业、温暖的支持。")
        else:
            self.add_message("AI Assistant",
                             "Hello! I'm your emotional recovery AI assistant.\n\nPlease share your feelings, and I will provide professional, warm support.")

    def set_config(self, api_key: str, model_choice: str, enable_rag: bool, rag, agents, language: str = "zh"):
        """设置配置"""
        self.api_key = api_key
        self.model_choice = model_choice
        self.enable_rag = enable_rag
        self.rag = rag
        self.agents = agents

        # 只有当语言真正改变时才更新
        if self.language != language:
            self.language = language
            self.update_language(language)

    def add_message(self, sender: str, message: str, is_user: bool = False):
        """添加消息到聊天区域（普通消息，不需要打字机效果）"""
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

        label = QLabel(html)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText)
        self.messages_layout.addWidget(label)

        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def add_response_section(self, title: str, content: str):
        """添加回复分区（使用打字机效果）"""
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

        # 内容（打字机效果）
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

        # 开始打字机效果
        content_label.start_typewriter(content, self.typewriter_delay)

        # 可选：鼠标点击时跳过动画
        content_label.mousePressEvent = lambda e: content_label.skip_to_end()

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
        while self.messages_layout.count():
            child = self.messages_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if self.language == "zh":
            self.add_message("AI 助手", "聊天记录已清空。有什么我可以帮助你的吗？")
        else:
            self.add_message("AI Assistant", "Chat history cleared. How can I help you?")

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
        user_input = self.input_text.toPlainText().strip()

        if not user_input and not self.uploaded_images:
            msg = "请分享你的感受或上传图片" if self.language == "zh" else "Please share your feelings or upload images"
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)
            return

        if not self.api_key:
            msg = "请先在右侧配置中输入 API Key" if self.language == "zh" else "Please enter your API Key in the configuration panel"
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)
            return

        if self.agents is None:
            msg = "请先点击「初始化 Agent」" if self.language == "zh" else "Please click 'Initialize Agent' first"
            QMessageBox.warning(self, "提示" if self.language == "zh" else "Warning", msg)
            return

        # 保存当前用户输入用于历史记录
        self.current_user_input = user_input if user_input else "[Uploaded images]"

        # 显示用户消息
        self.add_message("You" if self.language == "en" else "你",
                         user_input if user_input else "[Uploaded images]", is_user=True)

        # 在聊天界面显示"分析中..."消息
        if self.language == "zh":
            thinking_msg = "🤔 分析中..."
        else:
            thinking_msg = "🤔 Thinking..."
        thinking_label = self.add_thinking_message(thinking_msg)

        # 处理输入
        final_input = user_input

        # DeepSeek 需要用 OCR 提取图片文字
        if self.model_choice == "deepseek" and self.uploaded_images:
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

        # 分类问题类型
        try:
            from core.utils import classify_issue_type
            issue_type = classify_issue_type(final_input)
        except:
            issue_type = "general"

        self.current_issue_type = issue_type

        # 从 FAISS 数据库检索知识
        rag_context = ""
        retrieved_items = []

        if self.enable_rag and self.rag and hasattr(self.rag, 'is_ready') and self.rag.is_ready:
            try:
                retrieved_items = self.rag.search(final_input, issue_type=issue_type, k=3)

                if retrieved_items:
                    rag_parts = []
                    for i, item in enumerate(retrieved_items):
                        rag_parts.append(
                            f"【参考{i + 1}】{item['title']}\n来源: {item['source']}\n内容: {item['content'][:500]}")
                        print(f"📚 RAG检索到: {item['title']} (相似度: {item['score']:.2f})")

                    rag_context = "\n\n".join(rag_parts)
                    if self.language == "zh":
                        self.add_message("📚 知识库", "检索到相关心理学知识")
                    else:
                        self.add_message("📚 Knowledge Base", "Retrieved relevant psychology knowledge")
                else:
                    print("📚 RAG未检索到相关知识")
            except Exception as e:
                print(f"RAG搜索失败: {e}")

        # 启动工作线程
        self.send_worker = SendWorker(
            self.agents, final_input, issue_type, rag_context,
            self.model_choice, self.api_key, self.language
        )
        self.send_worker.finished.connect(lambda r: self.on_response_received(r, retrieved_items, thinking_label))
        self.send_worker.error.connect(lambda e: self.on_response_error(e, thinking_label))
        self.send_worker.start()

    def on_response_received(self, responses: tuple, retrieved_items: list, thinking_label: QLabel):
        """收到回复"""
        # 移除"分析中..."消息
        self.remove_thinking_message(thinking_label)

        if self.language == "zh":
            titles = ["💝 情感支持", "🧠 认知重构", "📋 行动计划", "💪 激励"]
        else:
            titles = ["💝 Emotional Support", "🧠 Cognitive Restructuring", "📋 Action Plan", "💪 Motivation"]

        typewriter_labels = []

        for title, content in zip(titles, responses):
            label = self.add_response_section(title, content)
            typewriter_labels.append(label)

        def check_all_finished():
            if all(not label.is_typing() for label in typewriter_labels):
                combined = "\n\n".join(responses)
                self.message_received.emit("assistant", combined)
                self.save_history(combined)
            else:
                QTimer.singleShot(100, check_all_finished)

        QTimer.singleShot(100, check_all_finished)

    def on_response_error(self, error_msg: str, thinking_label: QLabel):
        """处理错误"""
        self.remove_thinking_message(thinking_label)

        if "Insufficient Balance" in error_msg or "quota" in error_msg.lower():
            if self.language == "zh":
                QMessageBox.critical(self, "错误",
                                     f"💰 {self.model_choice.upper()} 账户余额不足！\n\n请充值或切换其他模型。")
            else:
                QMessageBox.critical(self, "Error",
                                     f"💰 {self.model_choice.upper()} account balance insufficient!\n\nPlease recharge or switch to another model.")
        else:
            msg = f"生成回复失败:\n{error_msg}" if self.language == "zh" else f"Failed to generate response:\n{error_msg}"
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
                "mode": "general",
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
