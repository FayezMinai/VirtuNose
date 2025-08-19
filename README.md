# VirtuNose
A Streamlit concierge chatbot that recommends niche &amp; designer fragrances using Gemini 2.0 Flash (REST) with LangChain conversation memory and a luxury stylist system prompt.
## Personalized Fragrance Concierge — Streamlit + LangChain + Gemini (REST)

A concierge-style fragrance recommendation chatbot built with **Streamlit**, **LangChain** (for conversation memory), and the **Gemini 2.0 Flash** REST API.

---


## Features
- Single-file app: `app.py` (Streamlit UI + LangChain memory + Gemini REST)
- System prompt tailored for a luxury fragrance stylist
- Conversation history (windowed) to retain context
- Simulated streaming of replies in the UI


---


## Requirements
- Python 3.10+
- A Gemini API key (create a key and note it as `GEMINI_API_KEY`)


---


## Quick Start


```bash
# 1) Clone your repo (or create a new one) and navigate into it


# 2) Create and populate .env from the example
cp .env.example .env
# then edit .env and paste your real key


# 3) Install dependencies
pip install -r requirements.txt


# 4) Run the app
streamlit run app.py
