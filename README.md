# Fine-Tuning GPT Model for Medical Applications

This repository contains the implementation of a Transformer-based language model designed for generating medical text. It supports fine-tuning, evaluation, and deployment, leveraging modern deep learning techniques for accurate and efficient text generation.

![Deep Learning](https://img.shields.io/badge/Skill-Deep%20Learning-yellow)
![PyTorch](https://img.shields.io/badge/Skill-PyTorch-blueviolet)
![Transformers](https://img.shields.io/badge/Skill-Deep%20Learning-orange)
![Generative AI](https://img.shields.io/badge/Skill-Generative%20AI-yellow)
![LLaMA](https://img.shields.io/badge/Skill-GPT-blue)
![Fine Tuning Pretrained Models](https://img.shields.io/badge/Skill-Fine%20Tuning%20Pretrained%20Models-red)
![Model Deployment](https://img.shields.io/badge/Skill-Model%20Deployment-purpule)
![Tokenization](https://img.shields.io/badge/Skill-Tokenization-blue)
![Experiment Tracking Mlflow](https://img.shields.io/badge/Skill-Experiment%20Tracking%20Mlflow-yellow)
![Data Augmentation](https://img.shields.io/badge/Skill-Data%20Augmentation-red)
![Version Control](https://img.shields.io/badge/Skill-Version%20Control-white)
![High Performance Computing](https://img.shields.io/badge/Skill-High%20Performance%20Computing-pink)
![Python Programming](https://img.shields.io/badge/Skill-Python%20Programming-blue)

## Table of Contents
- Key Features
- Project Overview
- Architecture 
- Setup and Installation
- How to Run the Project
- Model Deployment

## Key Features
**• Transformer Architecture:** Utilizes a custom Transformer-based architecture with scalable rotary embeddings and multi-head self-attention.
**• Medical Domain Adaptation:** Fine-tuned for medical text generation using distributed and sharded datasets.
**• Experiment Tracking:** Logs training progress, metrics, and parameters using MLflow.
**• Custom Data Loader:** Efficient distributed, sharded data loading for handling large datasets.
**• Deployment: Supports** text generation via a pre-trained model through customizable inference methods.

## Project Overview
This project provides tools for **fine-tuning and deploying a Transformer-based language model**. It is specifically optimized for medical text generation with the following key components:

◉ **Training:** Fine-tune the model with a large medical dataset.

◉ **Evaluation:** Monitor performance using validation loss and generate sample outputs.

◉ **Deployment:** Deploy the model to generate medical text for various use cases.

## Architecture
#### Data Flow
**Data Loader:** Efficiently loads tokenized data from distributed, sharded files for training and validation.
**Training Loop:** Optimizes model parameters through gradient accumulation and periodic checkpointing.
**Validation Loop:** Evaluates model loss during training and ensures convergence.

#### Model Flow
**Input:** Tokenized text input.
**Transformer Blocks:** 
  - RMSNorm for normalization.
  - Multi-head self-attention with rotary positional embeddings.
  - Feed-forward layers with SwiGLU activations.
**Output:** Predicted token probabilities and logits.

#### Deployment Flow
**Model Loading:** Load a fine-tuned model checkpoint for inference.
**Prompt Encoding:** Tokenize the input text prompt.
**Text Generation:** Generate completions using nucleus sampling (top-p) and temperature adjustment, and decode generated tokens into human-readable text.

## Installation and Setup
**Prerequisites**
• Python 3.8+
• GPU with CUDA (optional but recommended)
• Libraries: torch, mlflow, fire, numpy, tiktoken, matplotlib.

**Setup Instructions**
Clone the repository:
```
git clone https://github.com/Naominour/Fine_Tuning_LLaMA_Model.git
cd Fine_Tuning_LLaMA_Model
```
Install dependencies:
```
pip install -r requirements.txt
```

Download or prepare the medical dataset and place it in the output_data directory.

## How to Run the Project
Let's begin with the official Llama 3.1 code release from Meta, which acts as our reference. This turns out to not be trivial because Meta's official repo does not seem to include documentation or instructions on how to actually use the models once you download them. But let's try:

Download the official **llama-models** repo, e.g. inside this project's directory is ok:
```
git clone https://github.com/meta-llama/llama-models.git
```
Download a model, e.g. the Llama 3.1 8B (base) model:
```
cd llama-models/models/llama3_1
chmod u+x download.sh
./download.sh
```
You'll have to enter a "URL from the email". For this you have to request access to Llama 3.1 here. Then when it asks which model, let's enter meta-llama-3.1-8b, and then again one more time meta-llama-3.1-8b to indicate the base model instead of the instruct model. This will download about 16GB of data into ./Meta-Llama-3.1-8B - 16GB because we have ~8B params in 2 bytes/param (bfloat16).

Now we set up our environment, best to create a new conda env, e.g.:
```
conda create -n llama31 python=3.10
conda activate llama31
```
Don't use a too recent Python (e.g. 3.12) because I think PyTorch support is still not 100% there. Now let's go back to the llama-models directory and install it. This will install the llama-models package which we can use to load the model:
```
cd ../../
pip install -r requirements.txt
pip install -e .
```
And now let's run the generation script:
```
cd ../
pip install fire
torchrun --nnodes 1 --nproc_per_node 1 reference.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model
```

**1. Model Deployment**
1. Load a trained checkpoint:

```
model = Llama.build(
    ckpt_dir="models/",
    tokenizer_path="tokenizer.model",
    max_seq_len=128,
    max_batch_size=1,
)
```

2. Generate predictions:
   
```
results = model.text_completion(
    prompts=["The patient was diagnosed with"],
    max_gen_len=50,
    temperature=0.7,
    top_p=0.9,
)
```
3. Sample Output
<img src="src\result.png" style="width:1000px;">

