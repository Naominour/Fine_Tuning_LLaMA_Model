import os
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from llama31 import Transformer, ModelArgs

def create_small_llama_from_hf():
    """Create small LLaMA model from Hugging Face, matching GPT2 size"""
    
    print("Creating small LLaMA configuration...")
    # Create small LLaMA config matching GPT2 size
    small_config = LlamaConfig(
        hidden_size=768,        # Same as GPT2
        num_hidden_layers=12,   # Same as GPT2
        num_attention_heads=12, # Same as GPT2
        intermediate_size=2048, # Adjusted for size
        vocab_size=50257       # Same as GPT2
    )
    
    print("Loading pretrained LLaMA...")
    # You'll need to login with your Hugging Face token first
    from huggingface_hub import login
    login()  # This will prompt for your token
    
    # Load the original model first
    original_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Create small model
    small_model = LlamaForCausalLM(small_config)
    
    # Map weights where possible
    print("Mapping weights...")
    small_state_dict = small_model.state_dict()
    original_state_dict = original_model.state_dict()
    
    mapped_state_dict = {}
    for key in small_state_dict.keys():
        if key in original_state_dict:
            if small_state_dict[key].shape == original_state_dict[key].shape:
                mapped_state_dict[key] = original_state_dict[key]
            else:
                # For layers that don't match, take a slice of the original
                orig_tensor = original_state_dict[key]
                if len(small_state_dict[key].shape) == 2:
                    mapped_state_dict[key] = orig_tensor[:small_state_dict[key].shape[0], 
                                                       :small_state_dict[key].shape[1]]
                else:
                    mapped_state_dict[key] = orig_tensor[:small_state_dict[key].shape[0]]
    
    # Save to our format
    base_dir = "/content/drive/MyDrive/Llama_Medical_LLM"
    small_llama_dir = os.path.join(base_dir, "small_llama/model")
    os.makedirs(small_llama_dir, exist_ok=True)
    
    # Convert to our format
    config = {
        "dim": 768,            # Same as GPT2
        "n_layers": 12,        # Same as GPT2
        "n_heads": 12,         # Same as GPT2
        "n_kv_heads": 12,
        "vocab_size": 50257,   # Same as GPT2
        "multiple_of": 256,
        "norm_eps": 1e-5,
        "max_seq_len": 1024,   # Same as GPT2
        "max_batch_size": 32
    }
    
    # Initialize our model
    model_args = ModelArgs(**config)
    our_model = Transformer(model_args)
    
    # Map the weights
    our_state_dict = our_model.state_dict()
    converted_state_dict = {}
    
    # Add mapping logic here for each layer
    layer_mapping = {
        'model.embed_tokens.weight': 'tok_embeddings.weight',
        'model.norm.weight': 'norm.weight',
        'lm_head.weight': 'output.weight'
    }
    
    # Map layer weights
    for i in range(12):  # For each layer
        layer_mapping.update({
            f'model.layers.{i}.self_attn.q_proj.weight': f'h.{i}.attention.wq.weight',
            f'model.layers.{i}.self_attn.k_proj.weight': f'h.{i}.attention.wk.weight',
            f'model.layers.{i}.self_attn.v_proj.weight': f'h.{i}.attention.wv.weight',
            f'model.layers.{i}.self_attn.o_proj.weight': f'h.{i}.attention.wo.weight',
            f'model.layers.{i}.mlp.gate_proj.weight': f'h.{i}.feed_forward.w1.weight',
            f'model.layers.{i}.mlp.up_proj.weight': f'h.{i}.feed_forward.w2.weight',
            f'model.layers.{i}.mlp.down_proj.weight': f'h.{i}.feed_forward.w3.weight',
            f'model.layers.{i}.input_layernorm.weight': f'h.{i}.attention_norm.weight',
            f'model.layers.{i}.post_attention_layernorm.weight': f'h.{i}.ffn_norm.weight',
        })
    
    for hf_name, our_name in layer_mapping.items():
        if hf_name in mapped_state_dict:
            converted_state_dict[our_name] = mapped_state_dict[hf_name]
    
    # Save everything
    print("Saving model...")
    torch.save(converted_state_dict, os.path.join(small_llama_dir, "consolidated.00.pth"))
    
    with open(os.path.join(small_llama_dir, "params.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    print("Saved small LLaMA model!")
    return our_model

if __name__ == "__main__":
    create_small_llama_from_hf()