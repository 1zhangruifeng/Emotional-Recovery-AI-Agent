"""
打字机效果标签组件
"""

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QTimer, Qt, pyqtSignal


class TypewriterLabel(QLabel):
    """打字机效果标签"""

    finished = pyqtSignal()  # 打字完成信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_text = ""
        self.current_index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.type_next_char)
        self.delay = 20  # 毫秒/字符
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setStyleSheet("background-color: white; padding: 12px;")

    def start_typewriter(self, text: str, delay: int = 20):
        """开始打字机效果"""
        self.full_text = text
        self.current_index = 0
        self.delay = delay
        self.setText("")
        self.timer.start(delay)

    def type_next_char(self):
        """输出下一个字符"""
        if self.current_index < len(self.full_text):
            self.current_index += 1
            # 显示已输入的内容
            displayed = self.full_text[:self.current_index]
            # 处理HTML换行和特殊字符
            displayed = displayed.replace('\n', '<br>')
            self.setText(displayed)
        else:
            self.timer.stop()
            self.finished.emit()

    def skip_to_end(self):
        """跳过动画，直接显示全部内容"""
        if self.timer.isActive():
            self.timer.stop()
        self.setText(self.full_text.replace('\n', '<br>'))
        self.finished.emit()

    def is_typing(self) -> bool:
        """是否正在打字中"""
        return self.timer.isActive()