

from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch

app = Flask(__name__, template_folder='templates')

# '/' means it is home route
@app.route('/', methods=['GET', 'POST'])
def home_page():
    return render_template("index.html")
@app.route('/adding',methods=['GET','POST'])
def start():
    if request.method == 'POST':
        question = request.form['question']

        model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
        tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

        def generate_response(question, max_length=50, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95):
            input_ids = tokenizer.encode(question, return_tensors="pt")

            generated_output = model.generate(
                input_ids,
                max_length=100,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
            response = tokenizer.decode(generated_output[0], skip_special_tokens=True)
            return response

        def post_process_response(generated_text):
            cleaned_text = " ".join(sorted(set(generated_text.split()), key=generated_text.split().index))
            return cleaned_text

        question1 = question
        response = generate_response(question1)
        cleaned_response = post_process_response(response)
        print("Generated Response:", cleaned_response)

        return jsonify({'response': cleaned_response})
    else:
        return render_template("index.html")
    
