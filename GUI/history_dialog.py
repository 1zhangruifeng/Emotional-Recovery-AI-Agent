"""
历史记录对话框 - 显示和管理对话历史
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
    QTextEdit, QPushButton, QSplitter, QMessageBox,
    QFileDialog, QListWidgetItem, QWidget
)
from PyQt5.QtCore import Qt, pyqtSignal


class HistoryDialog(QDialog):
    """历史记录对话框"""

    history_loaded = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # 获取父窗口的语言设置
        self.current_language = "zh"
        if parent and hasattr(parent, 'current_language'):
            self.current_language = parent.current_language

        if self.current_language == "zh":
            self.setWindowTitle("对话历史记录")
        else:
            self.setWindowTitle("Conversation History")

        self.setGeometry(200, 200, 800, 500)

        self.history_data = []
        self.history_path = Path("./data/history/history.json")

        self.init_ui()
        self.load_history()

    def init_ui(self):
        """初始化界面"""
        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Horizontal)

        # 左侧：历史列表
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_item_clicked)
        splitter.addWidget(self.history_list)

        # 右侧：详情显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        right_layout.addWidget(self.detail_text)

        # 按钮区域
        btn_layout = QHBoxLayout()

        if self.current_language == "zh":
            self.load_btn = QPushButton("📂 加载")
            self.delete_btn = QPushButton("🗑️ 删除")
            self.export_btn = QPushButton("📎 导出")
            self.close_btn = QPushButton("关闭")
        else:
            self.load_btn = QPushButton("📂 Load")
            self.delete_btn = QPushButton("🗑️ Delete")
            self.export_btn = QPushButton("📎 Export")
            self.close_btn = QPushButton("Close")

        self.load_btn.clicked.connect(self.load_selected)
        btn_layout.addWidget(self.load_btn)

        self.delete_btn.clicked.connect(self.delete_selected)
        btn_layout.addWidget(self.delete_btn)

        self.export_btn.clicked.connect(self.export_history)
        btn_layout.addWidget(self.export_btn)

        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)

        right_layout.addLayout(btn_layout)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 500])

        layout.addWidget(splitter)

    def load_history(self):
        """加载历史记录"""
        self.history_list.clear()
        self.history_data = []

        if not self.history_path.exists():
            if self.current_language == "zh":
                self.detail_text.setText("暂无历史记录")
            else:
                self.detail_text.setText("No history records found.")
            return

        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                self.history_data = json.load(f)

            for i, item in enumerate(reversed(self.history_data)):
                timestamp = item.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%m-%d %H:%M")
                    except:
                        time_str = timestamp[:16]
                else:
                    time_str = "Unknown"

                user_input = item.get('input', '')[:50]
                if len(user_input) >= 50:
                    user_input += "..."

                display_text = f"[{time_str}] {user_input}"
                list_item = QListWidgetItem(display_text)
                list_item.setData(Qt.UserRole, i)
                self.history_list.addItem(list_item)

            if self.history_data:
                self.history_list.setCurrentRow(0)
                self.on_item_clicked(self.history_list.item(0))
            else:
                if self.current_language == "zh":
                    self.detail_text.setText("暂无历史记录")
                else:
                    self.detail_text.setText("No history records found.")

        except Exception as e:
            self.detail_text.setText(f"Failed to load: {str(e)}")

    def on_item_clicked(self, item):
        """点击历史项时显示详情"""
        if item is None:
            return

        original_idx = item.data(Qt.UserRole)
        if original_idx is None or original_idx >= len(self.history_data):
            return

        record = self.history_data[original_idx]

        if self.current_language == "zh":
            detail = f"""
## 📝 对话详情

**时间**: {record.get('timestamp', '未知')}

**问题类型**: {record.get('issue_type', '未分类')}

**语言**: {record.get('language', 'zh')}

**用户输入**:
> {record.get('input', '')}

**AI 回复**:
{record.get('response', '')}
            """
        else:
            detail = f"""
## 📝 Conversation Details

**Time**: {record.get('timestamp', 'Unknown')}

**Issue Type**: {record.get('issue_type', 'Unclassified')}

**Language**: {record.get('language', 'zh')}

**User Input**:
> {record.get('input', '')}

**AI Response**:
{record.get('response', '')}
            """

        self.detail_text.setMarkdown(detail)
        self.current_record = record

    def load_selected(self):
        """加载选中的记录到主界面"""
        current_item = self.history_list.currentItem()
        if current_item is None:
            if self.current_language == "zh":
                QMessageBox.warning(self, "提示", "请先选择一条记录")
            else:
                QMessageBox.warning(self, "Warning", "Please select a record first")
            return

        original_idx = current_item.data(Qt.UserRole)
        if original_idx is None or original_idx >= len(self.history_data):
            return

        record = self.history_data[original_idx]
        self.history_loaded.emit(record)
        self.close()

    def delete_selected(self):
        """删除选中的记录"""
        current_item = self.history_list.currentItem()
        if current_item is None:
            if self.current_language == "zh":
                QMessageBox.warning(self, "提示", "请先选择一条记录")
            else:
                QMessageBox.warning(self, "Warning", "Please select a record first")
            return

        original_idx = current_item.data(Qt.UserRole)
        if original_idx is None or original_idx >= len(self.history_data):
            return

        if self.current_language == "zh":
            title = "确认删除"
            msg = "确定要删除这条记录吗？"
        else:
            title = "Confirm Delete"
            msg = "Are you sure you want to delete this record?"

        reply = QMessageBox.question(self, title, msg, QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            del self.history_data[original_idx]

            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.history_path, 'w', encoding='utf-8') as f:
                    json.dump(self.history_data, f, ensure_ascii=False, indent=2)

                self.load_history()
                if self.current_language == "zh":
                    QMessageBox.information(self, "成功", "记录已删除")
                else:
                    QMessageBox.information(self, "Success", "Record deleted successfully")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {str(e)}")

    def export_history(self):
        """导出历史记录"""
        if not self.history_data:
            if self.current_language == "zh":
                QMessageBox.warning(self, "提示", "没有可导出的历史记录")
            else:
                QMessageBox.warning(self, "Warning", "No history data to export")
            return

        # 根据语言创建不同的对话框
        if self.current_language == "zh":
            # 中文模式：创建自定义对话框
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("导出格式")
            msg_box.setText("请选择导出格式:")

            # 创建自定义按钮
            json_btn = msg_box.addButton("JSON", QMessageBox.YesRole)
            csv_btn = msg_box.addButton("CSV", QMessageBox.NoRole)
            cancel_btn = msg_box.addButton("取消", QMessageBox.RejectRole)

            msg_box.exec_()

            clicked_btn = msg_box.clickedButton()

            if clicked_btn == cancel_btn:
                return
            elif clicked_btn == json_btn:
                file_format = "json"
            elif clicked_btn == csv_btn:
                file_format = "csv"
            else:
                return
        else:
            # 英文模式：使用标准问答对话框
            reply = QMessageBox.question(
                self, "Export Format",
                "Select export format:\n\nJSON\nCSV",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )

            # 重命名按钮文字
            yes_button = msg_box.button(QMessageBox.Yes)
            no_button = msg_box.button(QMessageBox.No)
            if yes_button:
                yes_button.setText("JSON")
            if no_button:
                no_button.setText("CSV")

            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                file_format = "json"
            else:
                file_format = "csv"

        # 选择保存路径
        if file_format == "json":
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存 JSON" if self.current_language == "zh" else "Save JSON",
                f"history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存 CSV" if self.current_language == "zh" else "Save CSV",
                f"history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "CSV Files (*.csv)"
            )

        if not file_path:
            return

        try:
            if file_format == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.history_data, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['timestamp', 'input', 'response', 'issue_type', 'language'])
                    writer.writeheader()
                    for record in self.history_data:
                        writer.writerow({
                            'timestamp': record.get('timestamp', ''),
                            'input': record.get('input', ''),
                            'response': record.get('response', ''),
                            'issue_type': record.get('issue_type', ''),
                            'language': record.get('language', 'zh')
                        })

            if self.current_language == "zh":
                QMessageBox.information(self, "成功", f"已导出 {len(self.history_data)} 条记录到:\n{file_path}")
            else:
                QMessageBox.information(self, "Success", f"Exported {len(self.history_data)} records to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")