import sys
import logging
import os

test_path = os.path.dirname(os.path.abspath(__file__))
if test_path not in sys.path:
    sys.path.append(test_path)

from Utils import model_utils

def test_hello():
    logging.info("Hello from test_utils_model_utils!")
    assert True
    
