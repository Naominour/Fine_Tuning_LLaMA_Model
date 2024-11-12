"""
Tokenizes medical datasets into binary format for LLaMA training.
"""

import os
from glob import glob
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tokenizer import Tokenizer

def write_datafile(filename, toks):
    """Saves token data as a .bin file"""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801
    header[1] = 7
    header[2] = len(toks)
    toks_np = np.array(toks, dtype=np.uint32)
    print(f"Writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def process_file(csv_filename, tokenizer, split_value):
    """Process a single CSV file for a specific split"""
    print(f"Processing {csv_filename} for split '{split_value}'")
    
    try:
        # Read CSV
        data = pd.read_csv(csv_filename)
        
        # Check columns
        if 'Question' not in data.columns or 'Answer' not in data.columns or 'split' not in data.columns:
            print(f"Missing required columns in {csv_filename}. Found columns: {data.columns.tolist()}")
            return []
        
        # Filter by split value
        split_data = data[data['split'] == split_value].copy()
        print(f"Found {len(split_data)} examples for split '{split_value}'")
        
        if len(split_data) == 0:
            return []
            
        # Convert to list and shuffle
        data_list = []
        for _, row in split_data.iterrows():
            if pd.notna(row['Question']) and pd.notna(row['Answer']):
                data_list.append((row['Question'], row['Answer']))
        
        random.shuffle(data_list)
        
        # Tokenize
        all_tokens = []
        for question, answer in data_list:
            text = f"Q: {question}\nA: {answer}"
            try:
                tokens = tokenizer.encode(text, bos=True, eos=True)
                all_tokens.extend(tokens)
            except Exception as e:
                print(f"Error tokenizing: {str(e)}")
                continue
                
        return all_tokens
        
    except Exception as e:
        print(f"Error processing {csv_filename}: {str(e)}")
        return []

def tokenize(tokenizer_path, data_directory, output_directory):
    """Tokenize all medical datasets"""
    os.makedirs(output_directory, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(tokenizer_path)
    
    # Get all CSV files
    csv_files = sorted(glob(os.path.join(data_directory, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_directory}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Process each split
    for split_value in ['train', 'test']:
        print(f"\nProcessing {split_value} split...")
        all_tokens = []
        
        # Process each file
        for csv_file in tqdm(csv_files, desc=f"Processing files for {split_value}"):
            tokens = process_file(csv_file, tokenizer, split_value)
            if tokens:
                all_tokens.extend(tokens)
        
        if all_tokens:
            output_filename = os.path.join(output_directory, f"medical_dataset_{split_value}.bin")
            write_datafile(output_filename, all_tokens)
            print(f"Saved {len(all_tokens)} tokens to {output_filename}")
        else:
            print(f"No data found for {split_value} split")

if __name__ == "__main__":
    output_directory = "/content/drive/MyDrive/Llama_Medical_LLM/output_data"
    tokenizer_path = "/content/drive/MyDrive/Llama_Medical_LLM/Llama3.1-8B/tokenizer.model"
    data_directory = "/content/drive/MyDrive/Medical_LLM/input_data"
    
    print("Starting tokenization process...")
    print(f"Input directory: {data_directory}")
    print(f"Output directory: {output_directory}")
    
    # Let's check the content of one file first
    csv_files = sorted(glob(os.path.join(data_directory, "*.csv")))
    if csv_files:
        sample_df = pd.read_csv(csv_files[0])
        print("\nSample data structure:")
        print(f"Columns: {sample_df.columns.tolist()}")
        if 'split' in sample_df.columns:
            print(f"Unique split values: {sample_df['split'].unique()}")
    
    tokenize(tokenizer_path, data_directory, output_directory)