import os
import time
import json
from typing import Any, Dict, Iterable

import requests
import streamlit as st
from dotenv import load_dotenv

# LangChain (memory + chain)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Compatibility: try both import paths for Base LLM class across LangChain versions
try:  # LangChain < 0.2 legacy path
    from langchain.llms.base import LLM  # type: ignore
except Exception:  # LangChain >= 0.2
    from langchain_core.language_models.llms import LLM  # type: ignore


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    st.error("GEMINI_API_KEY not found. Create a .env file next to app.py with `GEMINI_API_KEY=YOUR_KEY_HERE`.")
    st.stop()

GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={API_KEY}"
)


# ---------------------------
# System Prompt (embedded)
# ---------------------------
SYSTEM_PROMPT = (
    "You are an AI luxury fragrance expert and friendly stylist. Your role is to recommend niche "
    "and designer perfumes to users based on their existing collection and preferred scent notes. "
    "Ask first what fragrances they own, analyse common notes via online databases (Fragrantica, "
    "Parfumo, Reddit) using LangChain search tools, then ask which notes they love most. Recommend 3–5 "
    "perfumes each time, with:\n– Brand & fragrance name\n– Key notes\n– Vibe (occasion, season, masculine–feminine scale)\n– Why it matches their preferences\nEnd every answer with a follow-up question to keep the conversation going. Maintain a concierge-style tone: "
    "luxurious, elegant, yet friendly."
)


# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Fragrance Concierge (Gemini REST)", page_icon="💎", layout="centered")
st.title("💎 Luxury Fragrance Concierge — Gemini 2.0 Flash (REST)")
st.caption("Streamlit · LangChain (memory) · Gemini REST API")


# ---------------------------
# Sidebar (Settings)
# ---------------------------
with st.sidebar:
    st.subheader("Response Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Max output tokens", 128, 2048, 512, 64)
    st.markdown("---")
    st.write("**Tip:** Start by listing a few fragrances you already own.")


# ---------------------------
# Minimal Gemini REST wrapper as an LLM
# ---------------------------
class GeminiHTTP(LLM):
    """LangChain-compatible LLM that calls Gemini REST `generateContent`.

    Note: This uses non-streaming HTTP. We simulate streaming in the UI.
    """

    api_url: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "gemini-http"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_url": self.api_url,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
        }

    def _call(self, prompt: str, stop: Iterable[str] | None = None, run_manager: Any | None = None, **kwargs: Any) -> str:  
        # Respect stop sequences (basic client-side truncation)
        if stop:
            for s in stop:
                if s in prompt:
                    prompt = prompt.split(s)[0]

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": int(self.max_output_tokens),
            },
        }

        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(self.api_url, headers=headers, data=json.dumps(payload), timeout=60)
        except requests.RequestException as e:
            raise Exception(f"Network error calling Gemini: {e}")

        if resp.status_code != 200:
            # Try to surface helpful error info from the body
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise Exception(f"{resp.status_code} {body}")

        data = resp.json()
        # Parse the first candidate's text
        try:
            candidates = data.get("candidates", [])
            parts = candidates[0]["content"].get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            return text.strip() or "(No content returned.)"
        except Exception as e:
            raise Exception(f"Unexpected Gemini response format: {e}\nFull body: {data}")


# ---------------------------
# Prompt + Memory + Chain
# ---------------------------
prompt_tmpl = PromptTemplate(
    input_variables=["system_prompt", "history", "input"],
    template=(
        "System:\n{system_prompt}\n\n"
        "Conversation so far:\n{history}\n\n"
        "User:\n{input}\n\n"
        "Assistant:"
    ),
)

# Windowed memory to keep token usage in check
memory = ConversationBufferWindowMemory(
    k=6,  # keep the last 6 exchanges
    memory_key="history",
    input_key="input",
    ai_prefix="Assistant",
    human_prefix="User",
    return_messages=False,  # we want a text block for the prompt
)

# Instantiate the HTTP-backed LLM
llm = GeminiHTTP(
    api_url=GEMINI_API_URL,
    temperature=float(temperature),
    top_p=float(top_p),
    max_output_tokens=int(max_tokens),
)

chain = LLMChain(
    llm=llm,
    prompt=prompt_tmpl,
    memory=memory,
    verbose=False,
)


# ---------------------------
# Session state for UI chat log
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

if "last_input" not in st.session_state:
    st.session_state.last_input = None


# ---------------------------
# Helper: simulate token streaming in UI
# ---------------------------
def stream_to_placeholder(text: str, placeholder: st.delta_generator.DeltaGenerator, delay: float = 0.01, chunk_size: int = 24) -> None:
    shown = ""
    for i in range(0, len(text), chunk_size):
        shown += text[i : i + chunk_size]
        placeholder.markdown(shown)
        time.sleep(delay)


# ---------------------------
# Render prior chat
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])  # safe; Streamlit handles Markdown rendering


# ---------------------------
# Chat input + response
# ---------------------------
user_input = st.chat_input("Tell me a few fragrances you own, or ask for recommendations…")

if user_input:
    # Guard against duplicate sends on reruns / double-enter
    if st.session_state.last_input == user_input:
        user_input = None
    else:
        st.session_state.last_input = user_input

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare assistant streaming container
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        try:
            # Run the chain; memory auto-updates
            full_text = chain.predict(system_prompt=SYSTEM_PROMPT, input=user_input)
        except Exception as e:
            full_text = f"Sorry—something went wrong: {e}"

        # Simulate streaming to the UI
        stream_to_placeholder(full_text, stream_placeholder)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_text})


# ---------------------------
# Initial nudge if no messages yet
# ---------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Welcome to your **Luxury Fragrance Concierge**. ✨\n\n"
            "To begin, tell me a few fragrances you already own. I’ll spot common notes, ask what you love most, "
            "and then curate 3–5 refined recommendations tailored to your taste."
        )
