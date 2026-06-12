#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
情感恢复AI助手 - 入口文件
"""

import sys
import os
from pathlib import Path

# 设置HuggingFace镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from GUI.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Emotional Recovery AI Assistant")
    icon_path = Path(__file__).parent / "images" / "app_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # 设置应用样式
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
