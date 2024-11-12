import torch
from llama31 import Llama

def test_medical_model(
    ckpt_dir: str = "/content/drive/MyDrive/Llama_Medical_LLM/Llama3.1-8B",
    tokenizer_path: str = "/content/drive/MyDrive/Llama_Medical_LLM/Llama3.1-8B/tokenizer.model",
    max_seq_len: int = 256,
    max_gen_len: int = 256,
):
    print("Loading model...")
    
    # Initialize model
    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        flash=torch.cuda.is_available(),
    )
    
    def ask_medical_question(question: str) -> str:
        """Ask a medical question and get the response"""
        prompt = f"Q: {question}\nA:"
        
        sample_rng = torch.Generator(device='cuda')
        sample_rng.manual_seed(1337)
        
        result = model.text_completion(
            [prompt],
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
            temperature=0.7,
            top_p=0.9,
        )
        
        return result[0]['generation'].strip()
    
    # Interactive mode
    print("\nMedical Question-Answering System")
    print("Type 'quit' to exit")
    print("---------------------------------")
    
    while True:
        question = input("\nEnter your medical question: ")
        if question.lower() == 'quit':
            break
            
        print("\nGenerating response...")
        answer = ask_medical_question(question)
        print("\nAnswer:", answer)
        
    print("\nTesting complete!")

if __name__ == "__main__":
    test_medical_model()