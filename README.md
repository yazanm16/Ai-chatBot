# 🤖 AI Chatbot - PTUK Educational Platform

This is the Python-based backend service for the **AI Chatbot** of the PTUK Educational Platform. It provides semantic question-answering capabilities using sentence embeddings and the LLaMA model via the Groq API.

---

## 🚀 Overview

The chatbot backend leverages advanced semantic similarity techniques to analyze user queries in both Arabic and English by referencing a curated dataset of question-answer pairs. It then generates contextually accurate and intelligent responses using the LLaMA large language model, deployed on Groq hardware. This system employs a few-shot learning methodology to enhance performance and ensure adaptability across diverse topics. Seamless integration with the primary Node.js backend is achieved through efficient HTTP API communication.

---

## 🧠 AI Stack

- **SentenceTransformer** (`paraphrase-multilingual-MiniLM-L12-v2`) for semantic embedding
- **Groq API** to access LLaMA models (`llama3-8b-8192`)
- **Cosine Similarity** (via `sklearn`) to match top-k similar questions
- **Flask** to serve a RESTful endpoint

---

## 📁 File Structure

```
ai_chatbot/
├── dataSet.csv                       # Full dataset with questions, answers, context
├── dataSet_embedding_MiniLM.csv     # Precomputed embeddings for each question(search how to converte dataSet.csv to dataSet_embedding_MiniLM.csv )
├── chatbot.py                        # Main Flask app (provided above)
```

---

## ⚙️ How It Works

1. Receives a POST request with a question to `/generate`
2. Embeds the question using MiniLM
3. Finds top 3 similar questions from the dataset using cosine similarity
4. Prepares a prompt with similar Q&A pairs + context
5. Sends prompt to Groq's LLaMA model to get a high-quality response

---

## 🔐 Environment Variables

Create a `.env` or set in your system:
```bash
GROQ_API_KEY=your_groq_api_key
```

---

## 🧪 Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask server:
```bash
python chatbot.py
```

3. Send a POST request to:
```
http://localhost:5000/generate
```
With JSON body:
```json
{
  "text": "ما الهدف من التعليم الجامعي؟"
}
```

---

## 🔗 Endpoint

- `POST /generate` → returns smart AI response in the same language as input

