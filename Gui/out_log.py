import sys
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtGui import QTextCursor

class OutLog:
    def __init__(self, text_edit: QTextEdit):
        """
        Initializes the OutLog with a QTextEdit widget.
        """
        self.text_edit = text_edit
        self.original_stdout = sys.stdout # Store original stdout

    def write(self, message):
        """
        Writes the message to the QTextEdit and ensures the cursor is at the end.
        """
        self.text_edit.moveCursor(QTextCursor.End)
        self.text_edit.insertPlainText(message)
        self.text_edit.ensureCursorVisible()
        self.text_edit.repaint()  # Ensure the text edit is updated immediately
        
        # Optionally, also write to the original stdout for console visibility
        self.original_stdout.write(message) 

    def flush(self):
        """
        Required for file-like objects, but often no action needed for QTextEdit.
        """
        pass