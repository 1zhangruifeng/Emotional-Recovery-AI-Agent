"""
主窗口 - 整合所有 GUI 组件
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QPushButton, QMessageBox, QStatusBar,
    QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from GUI.chat_widget import ChatWidget
from GUI.config_panel import ConfigPanel
from GUI.history_dialog import HistoryDialog


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1300, 800)
        icon_path = Path(__file__).resolve().parent.parent / "images" / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # 初始化属性
        self.rag = None
        self.chat_widget = None
        self.config_panel = None
        self.status_bar = None
        self.history_btn = None
        self.clear_btn = None
        self.current_language = "zh"

        # 初始化UI
        self.init_ui()
        self.init_status_bar()
        if self.config_panel and getattr(self.config_panel, "local_models", None):
            self.on_local_models_updated(self.config_panel.local_models)

        # 加载知识库
        self.load_knowledge_base()

    def init_ui(self):
        """初始化界面"""
        # 创建中央窗口
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局 - 使用 QHBoxLayout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建分割器
        self.splitter = QSplitter(Qt.Horizontal)

        # 左侧：聊天组件
        self.chat_widget = ChatWidget()
        self.splitter.addWidget(self.chat_widget)

        # 右侧：配置面板
        self.config_panel = ConfigPanel()
        self.config_panel.setMinimumWidth(280)
        self.config_panel.setMaximumWidth(350)
        self.splitter.addWidget(self.config_panel)

        # 设置分割器的拉伸因子
        self.splitter.setStretchFactor(0, 3)  # 聊天区域占3份
        self.splitter.setStretchFactor(1, 1)  # 配置面板占1份

        # 设置初始分割位置
        self.splitter.setSizes([900, 350])

        # 将分割器添加到主布局
        main_layout.addWidget(self.splitter)

        # 连接信号
        self.connect_signals()

        # 设置窗口标题
        self.update_window_title()

    def init_status_bar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 初始状态栏消息
        self.status_bar.showMessage("Ready")

        # 创建工具栏容器
        toolbar_widget = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        self.history_btn = QPushButton()
        self.history_btn.clicked.connect(self.show_history)
        toolbar_layout.addWidget(self.history_btn)

        self.clear_btn = QPushButton()
        self.clear_btn.clicked.connect(self.clear_chat)
        toolbar_layout.addWidget(self.clear_btn)

        self.status_bar.addPermanentWidget(toolbar_widget)

        # 更新状态栏文本
        self.update_status_bar_text()

    def update_status_bar_text(self):
        """更新状态栏文本（根据语言）"""
        if self.current_language == "zh":
            self.history_btn.setText("📜 历史记录")
            self.clear_btn.setText("🗑️ 清空对话")
        else:
            self.history_btn.setText("📜 History")
            self.clear_btn.setText("🗑️ Clear")

    def update_window_title(self):
        """更新窗口标题（根据语言）"""
        if self.current_language == "zh":
            self.setWindowTitle("情感恢复 AI 助手")
        else:
            self.setWindowTitle("Emotional Recovery AI Assistant")

    def update_language(self, language: str):
        """更新语言（从配置面板调用）"""
        self.current_language = language
        self.update_window_title()
        self.update_status_bar_text()

        if self.chat_widget:
            self.chat_widget.update_language(language)

        # 更新状态栏的知识库状态消息
        if self.rag and hasattr(self.rag, 'is_ready') and self.rag.is_ready:
            if self.current_language == "zh":
                self.status_bar.showMessage("✅ 知识库已加载")
            else:
                self.status_bar.showMessage("✅ Knowledge base loaded")

    def connect_signals(self):
        """连接信号"""
        if self.config_panel:
            self.config_panel.config_changed.connect(self.on_config_changed)
            self.config_panel.local_models_updated.connect(self.on_local_models_updated)
            self.config_panel.rag_toggled.connect(self.on_rag_toggled)
            self.config_panel.language_changed.connect(self.update_language)
            self.config_panel.mode_changed.connect(self.on_mode_changed)
            if getattr(self.config_panel, "local_models", None):
                self.on_local_models_updated(self.config_panel.local_models)

        if self.chat_widget:
            self.chat_widget.message_received.connect(self.on_message_received)
            self.chat_widget.runtime_status.connect(self.on_runtime_status)

    def load_knowledge_base(self):
        """启动时加载 FAISS 知识库"""
        try:
            from core.knowledge_base import RAGKnowledgeBase

            self.rag = RAGKnowledgeBase()
            success = self.rag.load()

            if success:
                if self.config_panel:
                    self.config_panel.rag = self.rag
                    self.config_panel.update_rag_status()
                    config = self.config_panel.get_config()
                    if self.chat_widget:
                        self.chat_widget.set_config(
                            enable_rag=config.get('enable_rag', True),
                            rag=self.rag,
                            local_models=config.get('local_models'),
                            language=self.current_language,
                            mode=config.get('mode', 'text')
                        )
                if self.current_language == "zh":
                    self.status_bar.showMessage("✅ 知识库已加载")
                else:
                    self.status_bar.showMessage("✅ Knowledge base loaded")
            else:
                if self.current_language == "zh":
                    self.status_bar.showMessage("⚠️ 未找到知识库，请先运行 python scripts/build_knowledge_base.py")
                else:
                    self.status_bar.showMessage("⚠️ Knowledge base not found, please run: python scripts/build_knowledge_base.py")
                self.rag = None
                if self.config_panel:
                    self.config_panel.rag = None
                    self.config_panel.update_rag_status()
        except Exception as e:
            print(f"Failed to load knowledge base: {e}")
            if self.current_language == "zh":
                self.status_bar.showMessage("⚠️ 知识库加载失败")
            else:
                self.status_bar.showMessage("⚠️ Failed to load knowledge base")
            self.rag = None

    def on_config_changed(self):
        """配置变化"""
        if self.current_language == "zh":
            self.status_bar.showMessage("配置已更新")
        else:
            self.status_bar.showMessage("Configuration updated")

    def on_local_models_updated(self, local_models):
        """Local model manager updated."""
        config = self.config_panel.get_config() if self.config_panel else {}

        if self.chat_widget:
            self.chat_widget.set_config(
                enable_rag=config.get('enable_rag', True),
                rag=self.rag,
                local_models=local_models,
                language=self.current_language,
                mode=config.get('mode', 'text')
            )

        if self.current_language == "zh":
            if self.status_bar:
                self.status_bar.showMessage("✅ 内置多模态功能已就绪")
        else:
            if self.status_bar:
                self.status_bar.showMessage("✅ Built-in multimodal features ready")

    def on_mode_changed(self, mode):
        """Assistant mode changed."""
        if self.chat_widget:
            self.chat_widget.set_mode(mode)
        labels = {
            "text": ("文本助手", "Text assistant"),
            "voice": ("语音助手", "Voice assistant"),
            "video": ("视频助手", "Video assistant"),
        }
        zh, en = labels.get(mode, labels["text"])
        self.status_bar.showMessage(f"当前功能: {zh}" if self.current_language == "zh" else f"Current mode: {en}")

    def on_rag_toggled(self, enabled):
        """RAG 开关变化"""
        if self.current_language == "zh":
            self.status_bar.showMessage(f"RAG 知识库: {'已启用' if enabled else '已禁用'}")
        else:
            self.status_bar.showMessage(f"RAG: {'Enabled' if enabled else 'Disabled'}")

    def on_message_received(self, sender, message):
        """收到消息时的处理"""
        ready = "内置多模态功能已就绪" if self.current_language == "zh" else "Built-in multimodal features ready"
        self.on_runtime_status(ready, False)

    def on_runtime_status(self, message: str, busy: bool = True):
        """Show model/RAG runtime status in both the status bar and side panel."""
        if self.status_bar:
            prefix = "⏳ " if busy else "✅ "
            self.status_bar.showMessage(prefix + message)
        if self.config_panel:
            self.config_panel.set_runtime_status(message, busy)

    def show_history(self):
        """显示历史记录对话框"""
        if not self.chat_widget:
            return
        dialog = HistoryDialog(self)
        dialog.history_loaded.connect(self.load_history_record)
        dialog.exec_()

    def load_history_record(self, record):
        """加载历史记录到聊天界面"""
        if not self.chat_widget:
            return

        self.chat_widget.clear_chat()

        user_input = record.get('input', '')
        response = record.get('response', '')
        timestamp = record.get('timestamp', '')

        if self.current_language == "zh":
            self.chat_widget.add_message("你", user_input, is_user=True)
            self.chat_widget.add_message("AI 助手", f"[历史记录 - {timestamp[:19]}]\n\n{response}")
            self.status_bar.showMessage(f"已加载历史记录: {timestamp[:19]}")
        else:
            self.chat_widget.add_message("You", user_input, is_user=True)
            self.chat_widget.add_message("AI Assistant", f"[History - {timestamp[:19]}]\n\n{response}")
            self.status_bar.showMessage(f"Loaded history: {timestamp[:19]}")

    def clear_chat(self):
        """清空对话"""
        if not self.chat_widget:
            return

        if self.current_language == "zh":
            title = "确认清空"
            message = "确定要清空当前对话吗？"
        else:
            title = "Confirm Clear"
            message = "Are you sure you want to clear the current conversation?"

        reply = QMessageBox.question(self, title, message,
                                      QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.chat_widget.clear_chat()
            if self.current_language == "zh":
                self.status_bar.showMessage("对话已清空")
            else:
                self.status_bar.showMessage("Chat cleared")

    def resizeEvent(self, event):
        """窗口大小改变时的处理"""
        super().resizeEvent(event)
        # 更新分割器大小比例
        total_width = self.width()
        if total_width > 0:
            # 聊天区域占 70%，配置面板占 30%
            chat_width = int(total_width * 0.7)
            config_width = total_width - chat_width
            self.splitter.setSizes([chat_width, config_width])

    def closeEvent(self, event):
        """关闭窗口时的处理"""
        if self.chat_widget and hasattr(self.chat_widget, "stop_camera"):
            self.chat_widget.stop_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Emotional Recovery AI Assistant")
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
