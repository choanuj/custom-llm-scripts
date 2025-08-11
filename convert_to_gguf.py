import torch
import json
import subprocess
import os
import logging
from llm_model import GPTLanguageModel  # Import from previous script
from transformers import PreTrainedModel, PretrainedConfig

# Setup logging
logging.basicConfig(filename='conversion.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTConfig(PretrainedConfig):
    model_type = "gpt2"  # Map to closest supported architecture (GPT-2 like)
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_hidden_layers = n_layer
        self.max_position_embeddings = block_size
        self.dropout = dropout

class GPTPreTrainedModel(PreTrainedModel):
    config_class = GPTConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = GPTLanguageModel(config.__dict__)
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def export_to_hf(model_path='trained_model.pth', config_path='model_config.pth', output_dir='hf_model'):
    try:
        config_dict = torch.load(config_path)
        config = GPTConfig(**config_dict)
        model = GPTPreTrainedModel(config)
        model.load_state_dict(torch.load(model_path))
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        # Save config.json with GPT-2 like params
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f)
        logger.info(f"Exported to HF format at {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise

def convert_to_gguf(hf_dir, gguf_output='model.gguf', quant_type='Q4_0', llama_cpp_path='llama.cpp'):
    try:
        # Assume llama.cpp is cloned and built: git clone https://github.com/ggerganov/llama.cpp; cd llama.cpp; make
        convert_script = os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')
        subprocess.run(['python', convert_script, hf_dir, '--outfile', gguf_output, '--outtype', 'f16'], check=True)
        # Quantize
        quant_exe = os.path.join(llama_cpp_path, 'llama-quantize')
        subprocess.run([quant_exe, gguf_output, f"{gguf_output}.{quant_type}.gguf", quant_type], check=True)
        logger.info(f"Converted to GGUF: {gguf_output}.{quant_type}.gguf")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e.stderr}")
        print("Troubleshooting: Ensure llama.cpp is built correctly. Check if model architecture matches supported types (e.g., GPT-2). Try manual conversion.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_type', default='Q4_0', help='Quantization type (e.g., Q4_0, Q5_K, F16)')
    args = parser.parse_args()
    hf_dir = export_to_hf()
    convert_to_gguf(hf_dir, quant_type=args.quant_type)