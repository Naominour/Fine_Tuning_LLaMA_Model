import os
import glob
import fire
import time
import json
import math
import torch
import mlflow
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, TypedDict
from torch import nn
from tokenizer import Tokenizer
# -----------------------------------------------------------------------------
# ModelArgs


@dataclass
class ModelArgs:
    """
    A dataclass for storing and configuring model hyperparameters.
    """
    dim: int = 4096  # Dimensionality of model embeddings
    n_layers: int = 32  # Number of transformer layers
    n_heads: int = 32  # Number of attention heads
    n_kv_heads: Optional[int] = None  # Number of key-value heads (default: equal to n_heads)
    vocab_size: int = -1  # Size of the vocabulary
    multiple_of: int = 256  # SwiGLU hidden layer size must be a multiple of this value
    ffn_dim_multiplier: Optional[float] = None  # Multiplier for feedforward network size
    norm_eps: float = 1e-5  # Epsilon for RMS normalization
    rope_theta: float = 500000  # Maximum RoPE context length
    use_scaled_rope: bool = False  # Use scaled rotary positional embeddings
    max_batch_size: int = 8  # Maximum batch size
    max_seq_len: int = 2048  # Maximum sequence length
    flash: bool = False  # Use flash attention if True

    def __init__(self, **kwargs):
        """
        Initialize model arguments and validate constraints.
        Args:
            kwargs: Keyword arguments for overriding default parameters.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads  # Default to n_heads if not specified
        assert self.n_kv_heads <= self.n_heads, "n_kv_heads must not exceed n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.dim % self.n_heads == 0, "Embedding dimension must be divisible by n_heads"

# -----------------------------------------------------------------------------
# Transformer Components

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square (RMS) normalization layer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm layer.
        Args:
            dim: Dimension of the input features.
            eps: Small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scaling factor

    def _norm(self, x):
        """
        Compute the RMS normalization.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor):
    """
    Scale rotary positional embeddings (RoPE).
    Args:
        freqs: Input frequency tensor.
    Returns:
        Scaled frequencies.
    """
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    """
    Precompute complex frequencies for RoPE embeddings.
    Args:
        dim: Dimension of the embeddings.
        end: Sequence length for precomputing.
        theta: Base scaling factor for RoPE.
        use_scaled: Whether to use scaled RoPE.
    Returns:
        Precomputed complex frequencies.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real


def apply_rotary_emb(x, freqs_cis):
    """
    Apply rotary positional embeddings to input tensor.
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim).
        freqs_cis: Complex frequencies for RoPE.
    Returns:
        Tensor with RoPE applied.
    """
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    return x_out2.flatten(3).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value pairs for grouped-query attention.
    Args:
        x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim).
        n_rep: Number of repetitions.
    Returns:
        Tensor with repeated key-value pairs.
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class KVCache(nn.Module):
    """
    Key-Value cache for transformer attention layers.
    """
    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        """
        Initialize the KVCache.
        Args:
            batch_size: Batch size.
            seq_length: Sequence length.
            n_kv_heads: Number of key-value heads.
            head_dim: Dimension of each head.
            dtype: Data type of the cache.
            device: Device for the cache.
        """
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        """
        Update the cache with new key-value pairs.
        Args:
            start_pos: Starting position for the update.
            xk: New key tensor.
            xv: New value tensor.
        Returns:
            Updated key and value tensors.
        """
        seqlen = xk.size(1)
        self.cache_k[:, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv

class Attention(nn.Module):
    """
    Implements multi-head attention with optional flash attention and key-value caching.
    """

    def __init__(self, args: ModelArgs):
        """
        Initialize the attention module.
        Args:
            args: An instance of ModelArgs containing model hyperparameters.
        """
        super().__init__()
        self.flash = args.flash  # Use flash attention if True
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # Key-value heads
        model_parallel_size = 1  # Model parallel size (default: 1 GPU)
        self.n_local_heads = args.n_heads // model_parallel_size  # Local attention heads
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size  # Local key-value heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # Number of repetitions for key-value heads
        self.head_dim = args.dim // args.n_heads  # Dimension per attention head

        # Linear layers for query, key, value, and output projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # Query projection
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Key projection
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # Value projection
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # Output projection

        # Key-value cache, initialized during inference
        self.cache = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass for multi-head attention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            start_pos: Start position for caching.
            freqs_cis: Precomputed rotary positional embeddings.
            mask: Optional attention mask.
        Returns:
            Projected output tensor of shape (batch_size, seq_len, dim).
        """
        bsz, seqlen, _ = x.shape

        # Compute query, key, and value projections
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)  # Reshape query
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # Reshape key
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)  # Reshape value

        # Apply rotary positional embeddings to query and key
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        # Update key-value cache during inference
        if self.cache is not None:
            xk, xv = self.cache.update(start_pos, xk, xv)

        # Repeat key-value heads for grouped-query attention (GQA)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Transpose to make heads the batch dimension
        xq, xk, xv = (x.transpose(1, 2) for x in (xq, xk, xv))

        # Compute attention scores and apply attention
        if self.flash:
            # Flash attention for efficient computation
            output = F.scaled_dot_product_attention(xq, xk, xv, mask)
        else:
            # Standard scaled dot-product attention
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # Apply mask
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # Normalize scores
            output = torch.matmul(scores, xv)  # Compute weighted sum of values

        # Concatenate all heads and apply output projection
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        proj = self.wo(output)
        return proj


class FeedForward(nn.Module):
    """
    Implements the feedforward network (FFN) with SwiGLU activation.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the feedforward module.
        Args:
            dim: Input and output dimension.
            hidden_dim: Base hidden dimension.
            multiple_of: Hidden dimension must be a multiple of this value.
            ffn_dim_multiplier: Optional multiplier for hidden dimension.
        """
        super().__init__()

        # Adjust hidden dimension based on multiplier and ensure it's a multiple of `multiple_of`
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Linear layers for FFN
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # First linear layer
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # Output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # Second linear layer for SwiGLU

    def forward(self, x):
        """
        Forward pass for feedforward network.
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))  # Apply SwiGLU and output projection


class Transformer(nn.Module):
    """
    Implements the transformer model, consisting of multiple transformer blocks 
    with token embeddings, output projection, and RMS normalization.
    """

    def __init__(self, params: ModelArgs):
        """
        Initialize the Transformer model.
        Args:
            params: An instance of ModelArgs containing model hyperparameters.
        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size  # Size of the vocabulary
        self.n_layers = params.n_layers  # Number of transformer blocks

        # Token embeddings and transformer layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # Token embedding layer
        self.layers = nn.ModuleList(
            TransformerBlock(params) for _ in range(params.n_layers)
        )

        # Final normalization and output projection
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # Output projection layer

        # Precompute rotary positional embeddings
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,  # Double the sequence length for interpolation
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward method for inference, redirects to `forward_inference`.
        Args:
            tokens: Input token indices of shape (batch_size, seq_len).
            start_pos: Starting position for attention.
        Returns:
            Output logits of shape (batch_size, seq_len, vocab_size).
        """
        return self.forward_inference(tokens, start_pos)

    def forward_inference(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward pass for inference mode.
        Args:
            tokens: Input token indices of shape (batch_size, seq_len).
            start_pos: Starting position for attention.
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # Get token embeddings
        self.freqs_cis = self.freqs_cis.to(h.device)  # Ensure positional embeddings are on the correct device
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]  # Select relevant positional embeddings

        # Create a causal mask for attention
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)  # Mask upper triangle for causal attention
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # Apply final normalization and output projection
        h = self.norm(h)
        output = self.output(h).float()
        return output

    def forward_loss(self, inputs: torch.Tensor, targets: torch.Tensor, ignore_index=-100):
        """
        Compute the loss for training mode.
        Args:
            inputs: Input token indices of shape (batch_size, seq_len).
            targets: Target token indices of shape (batch_size, seq_len).
            ignore_index: Index to ignore in loss computation.
        Returns:
            Cross-entropy loss value.
        """
        _bsz, seqlen = inputs.shape
        h = self.tok_embeddings(inputs)  # Get token embeddings
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=inputs.device)
        mask = torch.triu(mask, diagonal=1).type_as(h)  # Causal mask
        start_pos = -1  # Disable key-value caching logic
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.output(h).float()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            input=logits.transpose(1, 2),
            target=targets,
            reduction="mean",
            ignore_index=ignore_index,
        )
        return loss

    def configure_optimizers(self, learning_rate, weight_decay=0.0, betas=(0.9, 0.97), device_type='cuda'):
        """
        Configure and initialize the optimizer for training.
        Args:
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for regularization.
            betas: Betas for AdamW optimizer.
            device_type: Type of device (e.g., "cuda").
        Returns:
            Optimizer instance.
        """
        train_params = []

        finetune_type = "all"
        if finetune_type == "rmsnorm":
            # Train only RMSNorm parameters
            for name, param in self.named_parameters():
                if "norm" in name:
                    train_params.append(param)
        elif finetune_type == "all":
            # Train all parameters
            for param in self.parameters():
                train_params.append(param)
        elif finetune_type == "all_no_pos":
            # Exclude positional embeddings and output weights
            for name, param in self.named_parameters():
                if name == "output.weight":
                    continue  # Exclude output weight
                elif name == "tok_embeddings.weight":
                    param.requires_grad = False  # Freeze token embeddings
                else:
                    train_params.append(param)

        # Log parameter counts
        print("Number of parameters: ", sum(p.numel() for p in self.parameters()))
        print("Number of trainable parameters: ", sum(p.numel() for p in train_params))

        # Initialize optimizer
        fused_available = True  # Fused optimizers for CUDA
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(train_params, lr=learning_rate, betas=betas, **extra_args)
        return optimizer
# -----------------------------------------------------------------------------
class Llama:
    """
    Wrapper for the Llama model, providing methods for model loading and text generation.
    """

    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        flash: bool = False,
        model_parallel_size: Optional[int] = 1,
        seed: int = 1,
    ) -> "Llama":
        """
        Build and initialize the Llama model from a checkpoint and tokenizer.
        Args:
            ckpt_dir: Directory containing model checkpoint files.
            tokenizer_path: Path to the tokenizer model file.
            max_seq_len: Maximum sequence length for the model.
            max_batch_size: Maximum batch size for inference.
            flash: Use flash attention if True.
            model_parallel_size: Number of parallel processes for model.
            seed: Random seed for reproducibility.
        Returns:
            An instance of the Llama class.
        """
        # Validate input parameters
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        # Set the device and random seed
        local_rank = 0
        torch.cuda.set_device(local_rank)
        torch.manual_seed(seed)

        # Load checkpoint files
        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"No checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints), "Mismatch between model_parallel_size and number of checkpoints."
        ckpt_path = checkpoints[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Load model parameters
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        # Initialize model arguments and tokenizer
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            flash=flash,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words, "Vocab size mismatch between model and tokenizer."

        # Set default tensor type based on hardware support
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        # Initialize and load the transformer model
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        """
        Initialize the Llama instance.
        Args:
            model: A Transformer model instance.
            tokenizer: A Tokenizer instance.
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        sample_rng: torch.Generator,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on input prompts.
        Args:
            prompt_tokens: List of tokenized prompts, each represented as a list of integers.
            sample_rng: Random number generator for sampling.
            max_gen_len: Maximum length of the generated sequence.
            temperature: Sampling temperature for randomness control.
            top_p: Top-p probability threshold for nucleus sampling.
            echo: Include prompt tokens in the output if True.
        Returns:
            A tuple containing:
                - Generated token sequences for each prompt.
                - Optional log probabilities of generated tokens (if implemented).
        """
        params = self.model.params
        bsz = len(prompt_tokens)  # Batch size
        assert bsz <= params.max_batch_size, f"Batch size {bsz} exceeds max batch size {params.max_batch_size}."

        # Determine prompt and total lengths
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len, "Prompt length exceeds max_seq_len."
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # Install KV cache in all attention layers
        for block in self.model.layers:
            layer_dtype = block.attention.wq.weight.dtype
            layer_device = block.attention.wq.weight.device
            block.attention.cache = KVCache(
                batch_size=bsz,
                seq_length=total_len,
                n_kv_heads=params.n_kv_heads,
                head_dim=params.dim // params.n_heads,
                dtype=layer_dtype,
                device=layer_device,
            )

        # Initialize token tensor with padding
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0  # Initialize start position for token generation
        eos_reached = torch.tensor([False] * bsz, device="cuda")  # End-of-sequence flags
        input_text_mask = tokens != pad_id  # Mask for non-padding tokens

        # Handle edge case where prompt length equals total length
        if min_prompt_len == total_len:
            logits = self.model.forward_inference(tokens, prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))  # Tokens indicating stop

        for cur_pos in range(min_prompt_len, total_len):
            # Generate logits for the next token
            logits = self.model.forward_inference(tokens[:, prev_pos:cur_pos], prev_pos)

            # Sample the next token based on temperature and top-p
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p, sample_rng)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)

            # Replace token only if it has been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            # Check for end-of-sequence tokens
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos  # Update previous position
            if all(eos_reached):  # Break if all sequences have ended
                break
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            """
            Process generated token sequences for each batch instance.
            Steps:
                1. Adjust sequence start position based on whether the prompt is echoed.
                2. Truncate sequence to the maximum generation length.
                3. Cut off the sequence at the first stop token, if any.
            """
            # Determine the start position (skip prompt tokens if `echo` is False)
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]

            # Truncate at the first occurrence of a stop token
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)  # Find stop token position
                    toks = toks[:eos_idx]  # Truncate at the stop token
                except ValueError:
                    pass  # Continue if the stop token is not found

            out_tokens.append(toks)  # Add processed tokens to the output list

        # Clear the KV cache in all attention layers to free memory
        for block in self.model.layers:
            block.attention.cache = None

        return out_tokens  # Return processed output tokens

    def text_completion(
        self,
        prompts: List[str],
        sample_rng: torch.Generator,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ):
        """
        Generate text completions for input prompts.
        Args:
            prompts: List of input strings to complete.
            sample_rng: Random number generator for token sampling.
            temperature: Sampling temperature to control randomness (default: 0.6).
            top_p: Top-p threshold for nucleus sampling (default: 0.9).
            max_gen_len: Maximum generation length (default: model's max sequence length - 1).
            echo: Include prompt in the output if True (default: False).
        Returns:
            List of dictionaries containing generated text for each prompt.
        """
        if max_gen_len is None:
            # Default to model's maximum sequence length minus one
            max_gen_len = self.model.params.max_seq_len - 1

        # Encode prompts into tokenized format (add BOS, exclude EOS)
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        # Generate token sequences for the prompts
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        # Decode generated tokens back into strings
        completions = [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
        return completions

def sample_top_p(probs, p, generator):
    """
    Perform nucleus (top-p) sampling to select the next token.
    Args:
        probs: Tensor of token probabilities (batch_size, vocab_size).
        p: Cumulative probability threshold for top-p sampling.
        generator: Random number generator for sampling.
    Returns:
        Tensor of selected token indices (batch_size, 1).
    """
    # Sort probabilities in descending order and compute cumulative sums
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Mask tokens that exceed the top-p cumulative probability
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0  # Set masked probabilities to zero

    # Normalize remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # Sample a token from the adjusted distribution
    next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)

    # Map sampled indices back to the original vocabulary
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# -----------------------------------------------------------------------------
def _peek_data_shard(filename):
    """
    Reads the header of a data shard file to extract metadata.
    Args:
        filename: Path to the data shard file.
    Returns:
        ntok: Number of tokens in the file as claimed in the header.
    """
    with open(filename, "rb") as f:
        # Read the first 256 int32 integers (header)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)  # Exit if the magic number does not match
    assert header[1] == 7, "Unsupported version"
    ntok = header[2]  # Extract the number of tokens
    return ntok  # Return the token count from the header

def _load_data_shard(filename):
    """
    Loads a data shard from disk, including both metadata and tokens.

    Args:
        filename: Path to the data shard file.

    Returns:
        tokens: Numpy array of tokenized data.
    """
    with open(filename, "rb") as f:
        # Read the header
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240801, "Magic number mismatch in the data .bin file"
        assert header[1] == 7, "Unsupported version"
        ntok = header[2]  # Number of tokens (claimed in header)
        
        # Read the remaining data as uint32 (tokens)
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "Number of tokens read does not match the header"
    return tokens

class DistributedShardedDataLoader:
    """
    A data loader that handles distributed and sharded datasets.

    Features:
        - Distributed: Works with multiple processes in DDP.
        - Sharded: Supports datasets split across multiple shard files.
        - Sequential: Iterates over the data in order; shuffling should be handled during dataset creation.

    Args:
        filename_pattern: Glob pattern to match data shard files.
        B: Batch size (number of sequences per batch).
        T: Sequence length (number of tokens per sequence).
        process_rank: Rank of the current process in DDP.
        num_processes: Total number of processes in DDP.
    """
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B  # Batch size
        self.T = T  # Sequence length

        # Discover and sort all shard files matching the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"Did not find any files that match the pattern {filename_pattern}"

        # Validate all shards and count total tokens
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # Ensure each shard has enough tokens for distributed processing
            assert shard_ntok >= num_processes * B * T + 1, "Shard has insufficient tokens for distributed batch"
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: Total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # Initialize loader state
        self.current_shard = None
        self.reset()

    def reset(self):
        """
        Reset the data loader to the start of the first shard.
        """
        if self.current_shard != 0:
            self.current_shard = 0  # Reset to the first shard
            self.tokens = _load_data_shard(self.files[self.current_shard])  # Load the first shard
        self.current_position = self.process_rank * self.B * self.T  # Reset position pointer

    def advance(self):
        """
        Advance to the next data shard and reset the position pointer.
        """
        self.current_shard = (self.current_shard + 1) % len(self.files)  # Cycle through shards
        self.current_position = self.process_rank * self.B * self.T  # Reset position pointer
        self.tokens = _load_data_shard(self.files[self.current_shard])  # Load the next shard

    def next_batch(self):
        """
        Load the next batch of data.

        Returns:
            x: Input tensor of shape (B, T).
            y: Target tensor of shape (B, T).
        """
        B, T = self.B, self.T
        # Extract a buffer of tokens for the batch
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf, dtype=torch.long)  # Convert to PyTorch tensor
        x = (buf[:-1]).view(B, T)  # Inputs: All except the last token
        y = (buf[1:]).view(B, T)  # Targets: All except the first token

        # Advance the position pointer
        self.current_position += B * T * self.num_processes

        # If the pointer exceeds the current shard, move to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x, y  # Return input and target tensors

# -----------------------------------------------------------------------------
def main(
    ckpt_dir: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B",
    tokenizer_path: str = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model",
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = 64,
    max_gen_len: int = 64,
    max_batch_size: int = 1,
    flash: bool = True,
    total_steps: int = 10000,
):
    """
    Main function for training and generating text using a Llama model.

    Args:
        ckpt_dir: Path to the model checkpoint directory.
        tokenizer_path: Path to the tokenizer model.
        temperature: Sampling temperature to control randomness (default: 1.0).
        top_p: Top-p probability threshold for nucleus sampling (default: 0.9).
        max_seq_len: Maximum input sequence length (default: 64).
        max_gen_len: Maximum length for generated sequences (default: 64).
        max_batch_size: Maximum batch size for training and inference (default: 1).
        flash: Enable flash attention for improved performance (default: True).
        total_steps: Total number of training steps (default: 10,000).
    """
    # Set default tensor type and device
    torch.set_default_dtype(torch.float16)
    torch.set_default_device('cuda')

    # Initialize the distributed and sharded data loader
    data_loader = DistributedShardedDataLoader(
        filename_pattern="/content/drive/MyDrive/Llama_Medical_LLM/output_data/*_test.bin",
        B=max_batch_size, 
        T=max_seq_len,
        process_rank=0,  # Single-process setup
        num_processes=1,
    )

    # Build the Llama model and tokenizer
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        flash=flash,
    )

    model = llama.model  # Extract the transformer model from the wrapper

    # Enable training for specific layers (e.g., normalization and query weights)
    trainable_params = []
    for name, param in model.named_parameters():
        if "norm" in name or "wq" in name:  # Train normalization and query layers
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False  # Freeze other layers

    model.train()  # Set the model to training mode
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)  # Initialize optimizer

    train_loss_values = []  # List to store training loss values
    accumulation_steps = 8  # Gradient accumulation steps

    # Start MLflow for logging
    mlflow.start_run()
    mlflow.log_param("total_steps", total_steps)
    mlflow.log_param("batch_size", max_batch_size)

    # Training loop
    for step in range(total_steps):
        optimizer.zero_grad()
        accumulated_loss = 0

        # Perform gradient accumulation
        for _ in range(accumulation_steps):
            x, y = data_loader.next_batch()  # Get next batch
            x, y = x.cuda(), y.cuda()  # Move data to GPU
            
            loss = model.forward_loss(x, y) / accumulation_steps  # Compute loss
            accumulated_loss += loss.item()
            loss.backward()  # Backpropagate gradients
            
            del x, y  # Clear data to save memory
            torch.cuda.empty_cache()

        optimizer.step()  # Update model parameters
        train_loss_values.append(accumulated_loss)  # Store loss

        # Log and print training loss periodically
        if step % 500 == 0:
            mlflow.log_metric("Train Loss", accumulated_loss, step=step)
            print(f"Step {step} - Train Loss: {accumulated_loss}")
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Save the trained model checkpoint
    model_checkpoint_path = "output_data/trained_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, model_checkpoint_path)

    mlflow.end_run()  # End MLflow logging

    # Switch model to evaluation mode for text generation
    model.eval()
    prompts = [
        "A 45-year-old woman was diagnosed with",  # Example medical prompt
        "A patient with breast cancer is undergoing"  # Example medical prompt
    ]

    # Initialize a random number generator for sampling
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)  # Set seed for reproducibility

    # Generate text completions
    results = llama.text_completion(
        prompts,
        sample_rng=sample_rng,
        max_gen_len=max_seq_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Print the prompts and corresponding completions
    for prompt, result in zip(prompts, results):
        print(prompt, end="")
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    # Use the Fire library to expose the `main` function as a CLI
    fire.Fire(main)
