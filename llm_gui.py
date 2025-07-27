import sys
import logging
import os
from pathlib import Path
from Managers.llm_manager import LlmManager

## Qt should be imported after openvino_genai to avoid conflicts
import PyQt5.QtWidgets
from Gui.llm_setup_window import LlmSetuWindow

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
