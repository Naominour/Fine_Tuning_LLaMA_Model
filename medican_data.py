"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is Llama 3.1 Tokenizer (with tiktoken).

The output is written to a newly created tinystories/ folder.
The script prints:

Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to /home/ubuntu/nano-llama31/tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to /home/ubuntu/nano-llama31/tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of uint32 numbers indicating the token ids.

The .bin file sizes are:
3.4G    /home/ubuntu/nano-llama31/tinystories/TinyStories_train.bin
72M     /home/ubuntu/nano-llama31/tinystories/TinyStories_val.bin
"""

import os
import glob
import json
import random
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tokenizer import Tokenizer
# -----------------------------------------------------------------------------

def write_datafile(filename, toks):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint32
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801 # magic
    header[1] = 7 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    toks_np = np.array(toks, dtype=np.uint32)
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def process_shard(shard_index, csv_filename, tokenizer_path, text_column="Question", answer_column="Answer", split="split"):
    # create tokenizer and encode function within the process
    tokenizer = Tokenizer(tokenizer_path)
    def encode(x):
        return tokenizer.encode(x, bos=True, eos=True)

    data = pd.read_csv(csv_filename)
    data = data[data['split'] == split ]

    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []

    for _, row in data.dropna(subset=[text_column, answer_column]).iterrows():
        
        text =  f"Q: {row[text_column]}\nA: {row[answer_column]}"
        tokens = encode(text)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize(tokenizer_path, data_directory, output_directory):
    csv_files = glob(os.path.join(data_directory, "*.csv"))
    # shard 0 will be the val split, rest is train
    '''
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename, tokenizer_path)
                       for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())

        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens)
'''
    for split_name in ["train", "test"]:
        print(f"Tokenizing {split_name} split with {len(csv_files)} files...")
        with ProcessPoolExecutor() as executor:
            futures = []
            for file in csv_files:
                futures.append(
                    executor.submit(process_shard, file, tokenizer_path, "Question", "Answer", split=split_name)
                )
                
                for future in as_completed(futures):
                    all_tokens = future.result()
                    
                    # Create a unique output filename based on the file name and split
                    base_name = os.path.basename(file).replace(".csv", "")
                    output_filename = os.path.join(output_directory, f"{base_name}_{split_name}.bin")
                    
                    # Save tokens to a unique shard file
                    write_datafile(output_filename, all_tokens)
                    print(f"Saved {split_name} shard to {output_filename}")

# Assuming DATA_CACHE_DIR is your chosen directory for tokenized output
if __name__ == "__main__":
    DATA_CACHE_DIR = "G:\\My Drive\\Medical_LLM\\output_data"  # Define your output directory
    tokenizer_path = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model"
    data_directory = "G:\\My Drive\\Medical_LLM\\input_data"  # Directory with your CSV files
    
    tokenize(tokenizer_path, data_directory, DATA_CACHE_DIR)
