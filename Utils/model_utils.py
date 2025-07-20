import sys
import logging
import os

'''
This module provides utility functions for model conversion, compression, and size retrieval.
It includes functions to convert and compress models using the OpenVINO GenAI library,
and to retrieve the size of a model in megabytes.
'''

compression_configs = {
    "DeepSeek-R1-Distill-Llama-8B": {
        "sym": True,
        "group_size": 128,
        "ratio": 0.8,
    },
    "DeepSeek-R1-Distill-Qwen-7B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "DeepSeek-R1-Distill-Qwen-14B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "DeepSeek-R1-Distill-Qwen-1.5B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "DeepSeek-R1-Distill-Qwen-32B": {"sym": True, "group_size": 128, "ratio": 1.0},
    "default": {
        "sym": False,
        "group_size": 128,
        "ratio": 0.8,
    },
}


int4_npu_config = {
    "sym": True,
    "group_size": -1,
    "ratio": 1.0,
}

def get_optimum_cli_command(model_id, weight_format, output_dir, compression_options=None, enable_awq=False, trust_remote_code=False):
    '''Generate the optimum-cli command for converting a model to OpenVINO format.
    Args:
        model_id (str): The model ID to be converted.
        weight_format (str): The weight format to convert the model to (e.g., "int4", "fp16").
        output_dir (Path): The directory where the converted model will be saved.
        compression_options (dict, optional): Options for model compression.
        enable_awq (bool, optional): Whether to enable AWQ compression.
        trust_remote_code (bool, optional): Whether to trust remote code execution.
    Returns:
        str: The command to run for model conversion.
    '''
    base_command = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format {}"
    command = base_command.format(model_id, weight_format)
    if compression_options:
        compression_args = " --group-size {} --ratio {}".format(compression_options["group_size"], compression_options["ratio"])
        if compression_options["sym"]:
            compression_args += " --sym"
        if enable_awq or compression_options.get("awq", False):
            compression_args += " --awq --dataset wikitext2 --num-samples 128"
            if compression_options.get("scale_estimation", False):
                compression_args += " --scale-estimation"
        if compression_options.get("all_layers", False):
            compression_args += " --all-layers"

        command = command + compression_args
    if trust_remote_code:
        command += " --trust-remote-code"

    command += " {}".format(output_dir)
    return command


def get_ov_model_hub_id(pt_model_id, precision):
    """
    Generate the OpenVINO model hub ID based on the AI ID, model ID, and precision.
    """
    OV_ORG = "OpenVINO"
    pt_model_name = pt_model_id.split("/")[-1]
    ov_model_name = pt_model_name + f"-{precision.lower()}-cw-ov"
    ov_model_hub_id = f"{OV_ORG}/{ov_model_name}"

    return ov_model_hub_id


#def convert_and_compress_model(model_id, model_config, precision, use_preconverted=False):
def convert_and_compress_model(ai_id, model_id, model_dir, precision, use_preconverted=False):
    '''Convert and compress a model to the specified precision and save it to the model directory.
    If the model is already converted and exists in the model directory, it will return the path.
    If use_preconverted is True, it will check for a preconverted model in the OpenVINO repo on Hugging Face.
    If the pre-converted model is not found, it will download a non converted model from Hugging Face
    and convert the model using the optimum-cli command. 
    Args:
        ai_id (str): The AI organization for the model ex. DeepSeek.
        model_id (str): The model ID to be converted.
        model_dir (Path): The directory where the converted model will be saved.
        precision (str): The precision to convert the model to (e.g., "INT4", "FP16").
        use_preconverted (bool): Whether to use a preconverted model from the OpenVINO Model Hub.
    Returns: model_dir (Path): The directory where the converted model is saved.
    '''
    
    from pathlib import Path
    import subprocess  # nosec - disable B404:import-subprocess check
    import platform

    pt_model_id = f"{ai_id}/{model_id}"
    pt_model_name = model_id
    remote_code = False
    if (model_dir / "openvino_model.xml").exists():
        logging.info(f"✅ {precision} {model_id} model already converted and can be found in {model_dir}")
        return model_dir
    if use_preconverted:
        ov_model_hub_id = get_ov_model_hub_id(pt_model_id, precision)
        logging.info(f"Checking for preconverted {precision} {model_id} model in OpenVINO Model Hub: {ov_model_hub_id}")
        import huggingface_hub as hf_hub

        hub_api = hf_hub.HfApi()
        if hub_api.repo_exists(ov_model_hub_id):
            logging.info(f"⌛Found preconverted {precision} {model_id}: {ov_model_hub_id}. Downloading model started. It may takes some time.")
            hf_hub.snapshot_download(ov_model_hub_id, local_dir=model_dir)
            logging.info(f"✅ {precision} {model_id} model downloaded and can be found in {model_dir}")
            return model_dir

    model_compression_params = {}
    if "INT4" in precision:
        model_compression_params = compression_configs.get(model_id, compression_configs["default"]) if not "NPU" in precision else int4_npu_config
    weight_format = precision.split("-")[0].lower()
    optimum_cli_command = get_optimum_cli_command(pt_model_id, weight_format, model_dir, model_compression_params, "AWQ" in precision, remote_code)
    logging.info(f"⌛ {model_id} conversion to {precision} started. It may takes some time.")
    logging.info("**Export command:**")
    logging.info(f"{optimum_cli_command}")
    subprocess.run(optimum_cli_command.split(" "), shell=(platform.system() == "Windows"), check=True)
    logging.info(f"✅ {precision} {model_id} model converted and can be found in {model_dir}")
    return model_dir


def get_model_size(model_dir):
    '''Get the size of the model in MB.'''
        
    file_name = "openvino_model.bin"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    file_path = model_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Model file {file_name} does not exist in {model_dir}.")
    file_size = os.path.getsize(file_path)

    return file_size / (1024 * 1024)  # Convert bytes to MB
        

def streamer(subword):
    print(subword, end="", flush=True)
    sys.stdout.flush()
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False
