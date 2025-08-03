import logging
from pathlib import Path
import os
import sys
import openvino_genai as ov_genai  
from Managers.llm_manager import LlmManager
from Utils.model_utils import streamer

## Qt should be imported  after openvino_genai to avoid conflicts
import PyQt5
import PyQt5.QtWidgets
from PyQt5.QtCore import Qt
from Gui.out_log import OutLog

class LlmChatWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self, pipe: ov_genai.LLMPipeline, 
                 generation_config: ov_genai.GenerationConfig,  parent=None):
        super().__init__(parent)
        self.set_pipe(pipe)
        self.set_generation_config(generation_config)
        self.setWindowTitle("LLM Chat")
        self.setGeometry(300, 300, 800, 600)
        self.init_ui()

    def set_pipe(self, pipe):
        self.pipe = pipe
        if not self.pipe:
            logging.error("Failed to set pipeline. Pipeline is None.")
            return
        logging.info("Pipeline set successfully.")

    def set_generation_config(self, generation_config):
        self.generation_config = generation_config
        if not self.generation_config:
            logging.error("Failed to set generation config. Config is None.")
            return
        logging.info("Generation config set successfully.")


    def init_layouts(self):
        # Main layout for the chat window
        self.main_layout = PyQt5.QtWidgets.QVBoxLayout()
        self.button_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)

        
    def add_text_output_ui(self):
        # Text output area for displaying chat messages
        self.chat_output = PyQt5.QtWidgets.QTextEdit()
        self.chat_output.setReadOnly(True)
        self.main_layout.addWidget(self.chat_output)

        self.pompt_input = PyQt5.QtWidgets.QLineEdit()
        self.pompt_input.setPlaceholderText("Type your message here...")
        self.main_layout.addWidget(self.pompt_input)
        
        # Redirect logging to the text output area
        sys.stdout = OutLog(self.chat_output)
        sys.stderr = sys.stdout  # Redirect stderr to the same QTextEdit        

    def init_butons(self):
        # Button for sending messages
        self.send_button = PyQt5.QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        # Connect the returnPressed signal of the prompt input to the send button click event
        self.pompt_input.returnPressed.connect(self.on_send_clicked)
        
        # Button for canceling the chat
        self.cancel_button = PyQt5.QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

        self.clear_button = PyQt5.QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: self.chat_output.clear())
        
        # Add buttons to the main layout
        self.button_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.send_button)
        self.button_layout.addWidget(self.cancel_button)
        self.main_layout.addLayout(self.button_layout)


    def on_send_clicked(self):
        # Get the input text from the prompt input field
        input_text = self.pompt_input.text().strip()
        if not input_text:
            PyQt5.QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a message.")
            return
        
        # Clear the input field
        self.pompt_input.clear()
        
        # Log the input text
        logging.info(f"User input: {input_text}")
        
        # Display the user input in the chat output area
        # Apply style to user input
        user_text = f"You: {input_text}"
        styled_text = f'<div style="text-align: right;"><span style="color: blue; font-family: Courier New;">{user_text}</span></div><br>'
        self.chat_output.append(styled_text)
        
        
        # Generate response using the LLM pipeline
        try:
            self.pipe.generate(input_text, self.generation_config, streamer)
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}")
            self.chat_output.append(f'<span style="color: red;">Error: {e}</span>')

    def on_cancel_clicked(self):
        # Close the chat window
        self.close()
        logging.info("Chat window closed.")
        

    def combine_layouts(self):
        # Combine all layouts into the main layout
        self.main_layout.addLayout(self.button_layout)


    def init_ui(self):
        # Initialize UI components
        self.init_layouts()
        self.add_text_output_ui()
        self.init_butons()
        self.combine_layouts()

        # Set the main layout for the chat window
        central_widget = PyQt5.QtWidgets.QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)
        logging.info("LLM Chat Window initialized.")