'''
This script demonstrates how to use the OpenVINO GenAI library to load and generate text using a DeepSeek model.
It includes functions to convert and compress models, etc.
'''

from pathlib import Path
import logging

import openvino_genai as ov_genai

from Utils.model_utils import convert_and_compress_model, get_optimum_cli_command, streamer, get_model_size

def main():
    logging.info("Hello from llm-deepseek!")
    device = "NPU"  # or "CPU"
    #model_id = "DeepSeek-R1-Distill-Qwen-7B"
    ai_id  = "deepseek-ai"
    model_id = "DeepSeek-R1-Distill-Qwen-1.5B"
    compression_variant = "INT4"
    device = "NPU"  # "CPU"
    model_path = Path(model_id+"-" + compression_variant + "-" + device)

    convert_and_compress_model(ai_id, model_id, model_path, compression_variant, use_preconverted=True)
    model_size = get_model_size(model_path)
    logging.info(f"Model size: {model_size:.2f} MB")

    logging.info(f"Loading model from {model_path}\n")

    pipe = ov_genai.LLMPipeline(model_path, device)
    genai_chat_template = ""
    # genai_chat_template = "{% for message in messages %}{% if loop.first %}"
    # "{{ '<｜begin▁of▁sentence｜>' }}{% endif %}"
    # "{% if message['role'] == 'system' and message['content'] %}"
    # "{{ message['content'] }}{% elif message['role'] == 'user' %}"
    # "{{  '<｜User｜>' +  message['content'] }}{% elif message['role'] == 'assistant' %}"
    # "{{ '<｜Assistant｜>' +  message['content'] + '<｜end▁of▁sentence｜>' }}"
    # "{% endif %}{% if loop.last and add_generation_prompt and message['role'] != 'assitant' %}"
    # "{{ '<｜Assistant｜>' }}{% endif %}{% endfor %}"
    logging.info(f"Model has been loaded successfully")
    if genai_chat_template:
        pipe.get_tokenizer().set_chat_template(genai_chat_template)

    generation_config = ov_genai.GenerationConfig()
    generation_config.max_new_tokens = 256

    input_prompt = "Tell me about planet Mars."
    print(f"\nInput text: {input_prompt}")
    pipe.generate(input_prompt, generation_config, streamer)

    input_prompt = "Tell me about planet Earth."
    print(f"\nInput text: {input_prompt}")
    pipe.generate(input_prompt, generation_config, streamer)

    generation_config.max_new_tokens = 256

    input_prompt = "What is e-based logarithm of 5?"
    print(f"\nInput text: {input_prompt}")
    pipe.generate(input_prompt, generation_config, streamer)

    input_prompt = "solve the equation 2x^2 + 3x - 100 = 0"
    print(f"\nInput text: {input_prompt}")
    pipe.generate(input_prompt, generation_config, streamer)


def test_start():
    logging.info("Test start")
    assert(True)
    logging.info("Test start passed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()