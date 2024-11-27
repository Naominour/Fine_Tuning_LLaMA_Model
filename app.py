from flask import Flask, request, render_template, jsonify
import torch
from pathlib import Path
from llama31 import Llama

# Define Flask app
app = Flask(__name__)

# Load the LLaMA model and tokenizer
def load_model():
    checkpoint_path = "output_data/trained_model.pth"
    ckpt_dir = "G:\My Drive\Llama_Medical_LLM\Llama3.1-8B/"
    tokenizer_path = "G:\My Drive\Llama_Medical_LLM\Llama3.1-8B\tokenizer.model"
    
    # Initialize the LLaMA model
    llama = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=2048,
        max_batch_size=8,
        flash=False
    )
    
    # Load the checkpoint to the CPU
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    llama.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move the model to GPU
    llama.model.to('cuda')
    llama.model.eval()
    return llama

# Load the model globally
print("Loading model...")
llama = load_model()
print("Model loaded successfully!")

# Flask routes
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical LLM</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background-color: #f4f4f9; }
            .container { max-width: 600px; margin: auto; background: white; padding: 20px; border-radius: 8px; }
            textarea, input { width: 100%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #result { margin-top: 20px; background: #e9ecef; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Medical Question Answering</h1>
            <form id="queryForm">
                <textarea id="prompt" placeholder="Ask a medical question..." rows="5"></textarea>
                <button type="button" onclick="askQuestion()">Generate Response</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            function askQuestion() {
                const prompt = document.getElementById('prompt').value;
                if (!prompt.trim()) {
                    alert('Please enter a question!');
                    return;
                }
                document.getElementById('result').innerHTML = 'Generating response...';
                fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('result').innerHTML = 'Error: ' + data.error;
                    } else {
                        document.getElementById('result').innerHTML = '<b>Response:</b><br>' + data.response;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('result').innerHTML = 'Error: Unable to generate response.';
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        # Generate text
        sample_rng = torch.Generator(device='cuda')
        sample_rng.manual_seed(1337)
        results = llama.text_completion(
            [prompt],
            sample_rng=sample_rng,
            max_gen_len=128,  # Adjust as needed
            temperature=0.6,
            top_p=0.9,
        )
        response = results[0]['generation']
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
