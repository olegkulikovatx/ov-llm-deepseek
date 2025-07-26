import sys
import logging
import os
from pathlib import Path
import PyQt5.QtWidgets
from Gui.llm_setup_window import LlmSetuWindow
from Managers.llm_manager import LlmManager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting LLM GUI application...")

    # Initialize the LLM manager
    llm_manager = LlmManager()

    # Create the main window
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    main_window = LlmSetuWindow(llm_manamer=llm_manager)
    main_window.show()

    # Start the application event loop
    sys.exit(app.exec_())

    # from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit
    # from PyQt5.QtWidgets import QVBoxLayout, QTableView
    # from PyQt5.QtWidgets import QApplication, QWidget

    # app = QApplication([])
    # window = QWidget()
    # query_layout = QHBoxLayout()
    # query_layout.addWidget(QLabel("Search:"))
    # query_layout.addWidget(QLineEdit()) 
    # main_layout = QVBoxLayout()
    # main_layout.addLayout(query_layout)
    # main_layout.addWidget(QTableView()) # Add another widget directly to the main layout
    # window.setLayout(main_layout)
    # window.show()
    # app.exec_()    
