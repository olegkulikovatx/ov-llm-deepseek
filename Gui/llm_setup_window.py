import logging
from pathlib import Path
import os
import sys
import PyQt5
import PyQt5.QtWidgets
from PyQt5.QtCore import Qt

class LlmSetuWindow(PyQt5.QtWidgets.QMainWindow):
    def __init__(self, llm_manamer, parent=None):
        super().__init__(parent)
        self.llm_manager = llm_manamer
        self.setWindowTitle("Setup LLM")
        self.setGeometry(100, 100, 800, 600)
        self.init_layouts()
        self.add_model_selection_ui()
        self.add_device_selection_ui()
        self.add_compression_options_ui()  
        self.add_temperature_ui()        
        self.add_button_ui()
        self.combine_layouts()
        self.add_text_output_ui()        
        self.init_ui()


    def init_layouts(self):
        # Add widget groups for model selection, device selection, and compression options
        self.main_layout = PyQt5.QtWidgets.QVBoxLayout()
        self.setup_layout = PyQt5.QtWidgets.QVBoxLayout()
        self.model_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.device_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.compression_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.temperature_layout = PyQt5.QtWidgets.QHBoxLayout()

        self.button_layout = PyQt5.QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)


    def combine_layouts(self):
        # Combine all layouts into the main layout
        self.setup_layout.addLayout(self.model_layout)
        self.setup_layout.addLayout(self.device_layout)
        self.setup_layout.addLayout(self.compression_layout)
        self.setup_layout.addLayout(self.temperature_layout)
        self.setup_layout.addStretch(1)  # Add stretch to fill space
        self.main_layout.addLayout(self.setup_layout)
        self.main_layout.addLayout(self.button_layout)

        # Set the main layout to the central widget
        central_widget = PyQt5.QtWidgets.QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)


    def add_device_selection_ui(self):
        # Device Selection Label
        self.device_label = PyQt5.QtWidgets.QLabel("Select Device:")
        self.device_label.move(50, 100)

        # Device Selection Dropdown
        self.device_dropdown = PyQt5.QtWidgets.QComboBox()
        self.device_dropdown.move(100, 100)
        self.device_dropdown.setGeometry(150, 100, 500, 30)
        devices = self.llm_manager.available_devices
        self.device_dropdown.addItems(devices)

        self.device_layout.addWidget(self.device_label)
        self.device_layout.addWidget(self.device_dropdown)


    def add_temperature_ui(self):
        # Temperature Label
        self.temperature_label = PyQt5.QtWidgets.QLabel("Set Temperature:")
        self.temperature_label.move(50, 200)

        # Temperature Input
        self.temperature_input = PyQt5.QtWidgets.QSlider(Qt.Horizontal) # type: ignore
        self.temperature_input.setRange(0, 100)  
        self.temperature_input.setSingleStep(1)
        self.temperature_input.setValue(int(self.llm_manager.temperature * 100))  # Scale to 0-100
        self.temperature_input.setGeometry(150, 200, 500, 30)
        self.temperature_layout.addWidget(self.temperature_label)
        self.temperature_layout.addWidget(self.temperature_input)
        # Temperature Value Label
        self.temperature_value_label = PyQt5.QtWidgets.QLabel(str(self.llm_manager.temperature))
        self.temperature_value_label.move(660, 200)
        self.temperature_layout.addWidget(self.temperature_value_label)

        # Connect slider value change to update label
        self.temperature_input.valueChanged.connect(self.update_temperature_value)

    def add_text_output_ui(self):
        # Add a text output area to display model information or results
        self.text_output = PyQt5.QtWidgets.QTextEdit()
        self.text_output.setReadOnly(True)
        self.main_layout.addWidget(self.text_output)
        # Redirect logging to the text output area
        logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(self.text_output)])
        logging.info("Text output area initialized for logging.")


    def update_temperature_value(self, value):
        # Update the temperature value label when the slider changes
        temperature = value / 100.0  # Scale back to 0-1
        self.temperature_value_label.setText(str(temperature))

    def add_button_ui(self):
        # Add buttons for actions like "Convert Model", "Load Model", etc.
        self.ok_button = PyQt5.QtWidgets.QPushButton("OK")
        self.cancel_button = PyQt5.QtWidgets.QPushButton("Cancel")
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)
        self.ok_button.clicked.connect(self.on_ok_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

    def on_ok_clicked(self):
        pass
        # Handle OK button click
        selected_model = self.model_dropdown.currentText()
        selected_device = self.device_dropdown.currentText()
        selected_compression = self.compression_dropdown.currentText()
        selected_temperature = self.temperature_input.value() / 100.0  # Scale back to 0-1
        
        logging.info(f"Selected Model: {selected_model}, Device: {selected_device}, "
                     f"Compression: {selected_compression}, Temperature: {selected_temperature}")
        
        # # Convert and compress the model
        self.llm_manager.active_model_id = selected_model
        self.llm_manager.active_compression_variant = selected_compression
        self.llm_manager.set_device(selected_device)
        self.llm_manager.set_temperature(selected_temperature)


    def on_cancel_clicked(self):
        # Handle Cancel button click
        logging.info("Setup cancelled by user.")
        self.close()


    def add_compression_options_ui(self):
        # Compression Options Label
        self.compression_label = PyQt5.QtWidgets.QLabel("Select Compression:")
        self.compression_label.move(50, 150)

        # Compression Options Dropdown
        self.compression_dropdown = PyQt5.QtWidgets.QComboBox()
        self.compression_dropdown.move(10, 150)
        self.compression_dropdown.setGeometry(150, 150, 500, 30)
        compression_options = self.llm_manager.compression_variants
        self.compression_dropdown.addItems(compression_options)

        self.compression_layout.addWidget(self.compression_label)
        self.compression_layout.addWidget(self.compression_dropdown)

    
    def add_model_selection_ui(self):
        # Model Selection Label
        self.model_label = PyQt5.QtWidgets.QLabel("Select Model:")
        self.model_label.move(50, 50)

        # Model Selection Dropdown
        self.model_dropdown = PyQt5.QtWidgets.QComboBox()
        self.model_dropdown.move(100, 50)
        self.model_dropdown.setGeometry(150, 50, 500, 30)
        models = self.llm_manager.get_available_models()
        self.model_dropdown.addItems(models)

        self.model_layout.addWidget(self.model_label)
        self.model_layout.addWidget(self.model_dropdown)


    def init_ui(self):
        # Initialize UI components here
        logging.info("LLM Setup Window initialized.")
        pass
