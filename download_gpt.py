import torch
from transformers import GPT2Model, GPT2Tokenizer
import struct
import os
from typing import Dict
import requests
import json

class GPT2Config:
    def __init__(self, config_dict: Dict):
        self.vocab_size = config_dict.get('vocab_size', 50257)
        self.n_positions = config_dict.get('n_positions', 1024)
        self.n_ctx = config_dict.get('n_ctx', 1024)
        self.n_embd = config_dict.get('n_embd', 768)
        self.n_layer = config_dict.get('n_layer', 12)
        self.n_head = config_dict.get('n_head', 12)
        self.activation_function = config_dict.get('activation_function', 'gelu')

class GPT2Model:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.weights = {}
        
    def load_weights_from_hf(self, model_name="gpt2"):
        """Download and load weights from Hugging Face"""
        print(f"Downloading {model_name} from Hugging Face...")
        
        # Create cache directory
        cache_dir = f"./{model_name}_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download config
        config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        config_path = os.path.join(cache_dir, "config.json")
        
        if not os.path.exists(config_path):
            print("Downloading config.json...")
            response = requests.get(config_url)
            response.raise_for_status()
            with open(config_path, 'w') as f:
                f.write(response.text)
        
        # Download tokenizer
        tokenizer_url = f"https://huggingface.co/{model_name}/resolve/main/tokenizer.json"
        tokenizer_path = os.path.join(cache_dir, "tokenizer.json")
        
        if not os.path.exists(tokenizer_path):
            print("Downloading tokenizer.json...")
            response = requests.get(tokenizer_url)
            response.raise_for_status()
            with open(tokenizer_path, 'w') as f:
                f.write(response.text)
        
        # Download vocab
        vocab_url = f"https://huggingface.co/{model_name}/resolve/main/vocab.json"
        vocab_path = os.path.join(cache_dir, "vocab.json")
        
        if not os.path.exists(vocab_path):
            print("Downloading vocab.json...")
            response = requests.get(vocab_url)
            response.raise_for_status()
            with open(vocab_path, 'w') as f:
                f.write(response.text)
        
        # Download merges
        merges_url = f"https://huggingface.co/{model_name}/resolve/main/merges.txt"
        merges_path = os.path.join(cache_dir, "merges.txt")
        
        if not os.path.exists(merges_path):
            print("Downloading merges.txt...")
            response = requests.get(merges_url)
            response.raise_for_status()
            with open(merges_path, 'w') as f:
                f.write(response.text)
        
        # Step 1: Download pytorch_model.bin from HuggingFace
        model_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"
        model_path = os.path.join(cache_dir, "pytorch_model.bin")
        
        if not os.path.exists(model_path):
            print("Downloading pytorch_model.bin... (this may take a while)")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            print("\nDownload complete!")
        
        # Step 2 & 3: Load pytorch_model.bin and convert tensors to NumPy arrays
        print("Loading and converting model weights...")
        success = self._safe_load_pytorch_weights(model_path)
        
        if not success:
            self._initialize_dummy_weights()
        
        return vocab_path, merges_path
    
def convert():

    with open("gpt2_cache/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Sort by ID (values), as vocab.json is {token: id}
    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])

    with open("gpt2_cache/vocab.txt", "w", encoding="utf-8") as out:
        for token, idx in sorted_vocab:
            out.write(token + "\n")

def save_matrix(tensor: torch.Tensor, path: str):
    tensor = tensor.cpu().contiguous().float()
    rows, cols = tensor.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", rows, cols))
        f.write(tensor.detach().numpy().astype("float32").tobytes())


def load_gpt2():
    config_dict = {
        'vocab_size': 50257,
        'n_positions': 1024,
        'n_ctx': 1024,
        'n_embd': 768,
        'n_layer': 12,
        'n_head': 12
    }
    
    config = GPT2Config(config_dict)
    model = GPT2Model(config)
    
    # Load weights from Hugging Face
    model.load_weights_from_hf("gpt2")

def extract_gpt2_weights(save_dir="gpt2_weights"):
    os.makedirs(save_dir, exist_ok=True)

    print("Loading GPT-2 model from Hugging Face...")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    # Save token and position embeddings
    save_matrix(model.wte.weight, f"{save_dir}/token_embedding.bin")
    save_matrix(model.wpe.weight, f"{save_dir}/position_embedding.bin")

    # Loop through layers
    for i, block in enumerate(model.h):
        prefix = f"{save_dir}/layer_{i}"

        # Attention weights
        save_matrix(block.attn.c_attn.weight, f"{prefix}_attn_c_attn_weight.bin")
        save_matrix(block.attn.c_attn.bias.unsqueeze(0), f"{prefix}_attn_c_attn_bias.bin")
        save_matrix(block.attn.c_proj.weight, f"{prefix}_attn_c_proj_weight.bin")
        save_matrix(block.attn.c_proj.bias.unsqueeze(0), f"{prefix}_attn_c_proj_bias.bin")

        # Feedforward
        save_matrix(block.mlp.c_fc.weight, f"{prefix}_mlp_fc_weight.bin")
        save_matrix(block.mlp.c_fc.bias.unsqueeze(0), f"{prefix}_mlp_fc_bias.bin")
        save_matrix(block.mlp.c_proj.weight, f"{prefix}_mlp_proj_weight.bin")
        save_matrix(block.mlp.c_proj.bias.unsqueeze(0), f"{prefix}_mlp_proj_bias.bin")

        # LayerNorms
        save_matrix(block.ln_1.weight.unsqueeze(0), f"{prefix}_ln1_weight.bin")
        save_matrix(block.ln_1.bias.unsqueeze(0), f"{prefix}_ln1_bias.bin")
        save_matrix(block.ln_2.weight.unsqueeze(0), f"{prefix}_ln2_weight.bin")
        save_matrix(block.ln_2.bias.unsqueeze(0), f"{prefix}_ln2_bias.bin")

    # Final LayerNorm
    save_matrix(model.ln_f.weight.unsqueeze(0), f"{save_dir}/final_ln_weight.bin")
    save_matrix(model.ln_f.bias.unsqueeze(0), f"{save_dir}/final_ln_bias.bin")

    print(f"✅ Weights saved to: {save_dir}/")

if __name__ == "__main__":
    load_gpt2()
    convert()
    extract_gpt2_weights()
