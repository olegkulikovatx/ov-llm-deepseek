import sys
import logging
import os

test_path = os.path.dirname(os.path.abspath(__file__))
if test_path not in sys.path:
    sys.path.append(test_path)

from Managers.llm_manager import LlmManager

def test_hello():
    logging.info("Hello from test_utils_model_utils!")
    assert True


def test_llm_manager_initialization():
    logging.info("Testing LlmManager initialization...")
    llm_manager = LlmManager()
    
    assert llm_manager.active_model_id == "DeepSeek-R1-Distill-Qwen-1.5B"
    assert llm_manager.active_compression_variant == "INT4"
    assert llm_manager.device in llm_manager.available_devices
    assert 0 <= llm_manager.temperature <= 1
    
    logging.info("LlmManager initialized successfully.")
    