# # another API for bakend
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
# ========== Load embeddings and model ==========
df_embed = pd.read_csv("dataSet_embedding_MiniLM.csv")
df_embed['Embedding'] = df_embed['Embedding'].apply(eval).apply(np.array)

df_full = pd.read_csv("dataSet.csv")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ========== Set up GROQ ==========
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DEFAULT_MODEL = "llama3-8b-8192"

# ========== Functions ==========
def clean_text(text):
    return text.replace('"', '').replace("'", "").strip()

def get_embedding(sentence):
    return model.encode(sentence)

def get_top_k_similar_sentences(input_sentence, k=3):
    input_sentence = clean_text(input_sentence)
    df_embed_cleaned = df_embed.copy()
    df_embed_cleaned['Question_clean'] = df_embed_cleaned['Question'].apply(clean_text)

    input_emb = get_embedding(input_sentence)
    similarities = cosine_similarity([input_emb], list(df_embed['Embedding']))[0]

    top_k_indices = similarities.argsort()[-k:][::-1]
    top_scores = similarities[top_k_indices]

    results = df_embed.iloc[top_k_indices].copy()
    results['Similarity'] = top_scores
    results = results.merge(df_full[['Question', 'Response', 'Context']], on='Question', how='left')
    return results[['Question', 'Response', 'Context', 'Similarity']]

def build_assistant_prompt(top_matches_df):
    prompt = "Simiralr questions with their answers and contexts:\n\n"
    for _, row in top_matches_df.iterrows():
        prompt += f"Question: {row['Question']}\n"
        prompt += f"Answer: {row['Response']}\n"
        prompt += f"Context: {row['Context']}\n\n"
    return prompt

def assistant(content: str):
    return {"role": "assistant", "content": content}

def user(content: str):
    return {"role": "user", "content": content}

def system(content: str):
    return {"role": "system", "content": content}

def chat_completion(messages: List[Dict], model=DEFAULT_MODEL, temperature: float = 0.0, top_p: float = 0.9) -> str:
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content

# ========== Flask App ==========
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    input_question = data.get("text", "")
    if not input_question:
        return jsonify({"error": "No question provided"}), 400

    top_matches = get_top_k_similar_sentences(input_question)
    assistant_content = build_assistant_prompt(top_matches)

    prompt_template = """You are a question answering chatbot.\nYou are tasked to answer the following question using the same language as the question itself."""

    messages = [
        system(prompt_template),
        user(f"I have the following question: {input_question}"),
        assistant(assistant_content),
        user(f"The question is:\n{input_question}\n\nPlease answer directly and clearly. DO NOT include any notes, explanations, or repeat the examples above.")
    ]

    answer = chat_completion(messages)
    return jsonify({"response": answer})

if __name__ == '__main__':
    app.run(port=5000)
