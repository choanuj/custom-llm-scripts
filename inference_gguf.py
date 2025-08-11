import logging
import psutil
import torch
from llama_cpp import Llama

# Setup logging
logging.basicConfig(filename='inference.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def monitor_resources():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = 0
    if torch.backends.mps.is_available():
        gpu_usage = torch.mps.current_allocated_memory() / (1024 ** 2)  # MB
    logger.info(f"CPU: {cpu_usage}%, RAM: {ram_usage}%, GPU Mem: {gpu_usage} MB")
    return cpu_usage, ram_usage, gpu_usage

def run_inference(gguf_path, query, max_tokens=100, use_gpu=False):
    try:
        n_gpu_layers = -1 if use_gpu else 0  # -1 for all layers on GPU
        llm = Llama(gguf_path, n_gpu_layers=n_gpu_layers, verbose=False)
        monitor_resources()  # Log start resources
        output = llm(query, max_tokens=max_tokens, echo=True)
        response = output['choices'][0]['text']
        logger.info(f"Query: {query}\nResponse: {response}")
        monitor_resources()  # Log end resources
        return response
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        print("Troubleshooting: Ensure GGUF file is valid. Check GPU support (rebuild llama-cpp-python with Metal for Mac). Reduce max_tokens if OOM.")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gguf_path', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--max_tokens', default=100, type=int)
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()
    print(run_inference(args.gguf_path, args.query, args.max_tokens, args.use_gpu))