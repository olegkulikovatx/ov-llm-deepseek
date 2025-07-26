
import sys
import logging  
import os
from pathlib import Path

from Utils import model_utils

class LlmManager:
    def __init__(self):
        logging.info("Hello from llm_manager!")
        self.model_ids = ["DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B"]
        self.active_model_id = "DeepSeek-R1-Distill-Qwen-1.5B"
        self.compression_variants = ["INT4", "INT8", "FP16"]
        self.active_compression_variant = "INT4"
        self.ai_id = "deepseek-ai"
        self.available_devices = model_utils.get_devives()
        self.device_preference = ["GPU", "NPU", "CPU"]
        self.device = self._select_device()
        self.temperature = 0.7

    def _select_device(self):
        '''Select the best available device based on preference.'''
        for dev in self.device_preference:
            if dev in self.available_devices:
                logging.info(f"Using device: {dev}")
                return dev
        logging.error("No suitable device found for model inference.")
        return ""
    
    def set_device(self, device):
        '''Set the device for model inference.'''
        if device in self.available_devices:
            self.device = device
            logging.info(f"Device set to: {self.device}")
        else:
            logging.error(f"Device {device} is not available.")
            self.device = self._select_device()

    def set_temperature(self, temperature):
        '''Set the temperature for model generation.'''
        if 0 <= temperature <= 1:
            self.temperature = temperature
            logging.info(f"Temperature set to: {self.temperature}")
        else:
            logging.error("Temperature must be between 0 and 1.")

    def convert_and_compress_model(self, model_id=None, compression_variant=None):
        '''Convert and compress the model to the specified precision.'''
        if model_id is None:
            model_id = self.active_model_id
        if compression_variant is None:
            compression_variant = self.active_compression_variant
        
        model_path = Path(self.active_model_id + "-" + self.active_compression_variant + "-" + self.device)
        return model_utils.convert_and_compress_model(self.ai_id, model_id, model_path, compression_variant, use_preconverted=True)
    
    def get_available_models(self):
        '''Get the list of available models.'''
        return self.model_ids

    def test_hello(self):
        logging.info("Hello from test_llm_manager!")
        assert True