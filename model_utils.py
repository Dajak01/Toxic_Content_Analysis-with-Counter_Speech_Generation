import pandas as pd
import tensorflow as tf
import torch
from transformers import BertTokenizer, TFBertForSequenceClassification, GPT2Tokenizer
from torch.utils.data import Dataset
from torch.nn.functional import softmax
import requests
import json
import openai


# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the MT-CONAN dataset globally to avoid reloading
mt_conan_data = pd.read_csv('MT-CONAN.csv')


def analyze_toxicity_with_bert(text):
   # Load dataset and search for a matching comment 
    data = pd.read_csv('train_dataset.csv')
    matched_data = data[data['comment_text'].str.contains(text, case=False, na=False)]

    if not matched_data.empty:
        prediction = matched_data.iloc[0]['toxicity']
        # Map prediction to toxicity levels
        if prediction == 1:
            return "Somewhat Toxic"
        elif prediction == 2:
            return "Toxic"
        elif prediction in [3, 4]:
            return "Very Toxic"
        else:
            return "Non-Toxic"
    else:
       # If no match, use OpenAI for analysis
        level, _ = analyze_with_openai(text, "sk-proj-5V1UsF8sgJqRmQ87CgtQT3BlbkFJV8WjkYtSa8PZfpoOUil6")
        return level  # Returning only the toxicity level from OpenAI



def generate_counter_response(text, toxicity_level):
   # Only generate a counter response if the text is toxic
    if toxicity_level == "Non-Toxic":
        return "No counter response, this is not a toxic comment."
    else:
        # Search for the exact match in HATE_SPEECH column
        potential_matches = mt_conan_data[mt_conan_data['HATE_SPEECH'].str.contains(text, case=False, na=False)]
        if not potential_matches.empty:
            return potential_matches.iloc[0]['COUNTER_NARRATIVE']
        else:
            # Fallback to OpenAI's API if no matches found
            return fetch_openai_response(text)
            

def fetch_openai_response(text):
    api_key = "sk-proj-5V1UsF8sgJqRmQ87CgtQT3BlbkFJV8WjkYtSa8PZfpoOUil6"  
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",  
        "messages": [
            {"role": "user", "content": f"Please provide a polite and constructive response to this comment: {text}"}
        ],
        "temperature": 0.5,
        "max_tokens": 150  # Adjust max_tokens as needed
    }
    url = "https://api.openai.com/v1/chat/completions"  # Correct endpoint for chat models
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_data = response.json()
        if 'choices' in response_data and response_data['choices']:
            return response_data['choices'][0]['message']['content']
        else:
            return "No completion found in response."
    else:
        error_message = response_data.get('error', {}).get('message', 'Unknown error occurred.')
        return f"Failed to fetch data: {error_message}"



def analyze_with_openai(text, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_text = f"""
    Analyze the toxicity of the following statement. Classify the toxicity level (Non-Toxic, Somewhat Toxic, Toxic, Very Toxic), and suggest a polite and constructive counter-response.
    Statement: '{text}'
    Example:
    Statement: 'You are always such a disappointment!'
    Level: Toxic
    Counter Response: 'It's tough to hear that. Let's discuss what's really bothering you to address the root of the issue.'
    """
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": prompt_text}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        response_message = response_data['choices'][0]['message']['content']
        return parse_openai_response(response_message)
    else:
        return "Failed to fetch data: " + response.text, None

def parse_openai_response(response_message):
    lines = response_message.strip().split('\n')
    level_line = next((line for line in lines if "Level:" in line), "Level: Not available")
    counter_response_index = lines.index(level_line) + 1 if "Level:" in level_line else -1
    level = level_line.split("Level:")[1].strip() if "Level:" in level_line else "Not available"
    counter_response = " ".join(lines[counter_response_index:]) if counter_response_index != -1 else "Counter response not available."
    return level, counter_response