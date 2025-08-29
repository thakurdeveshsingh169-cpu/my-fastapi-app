from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import time
from typing import Dict, List
import langid
import re
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Load environment variables
load_dotenv()
# -------------------------
# Removed TogetherAI key as requested.
# Add your Groq key (GROQ_API_KEY) in your .env file or replace the default string below.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# -------------------------

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyCTYSeHgh3-zMOvoILD1WjBLj-sZl9PVDs")

# Usage limits
image_limit_per_day = 5
question_limit_per_day = 15
ip_usage_tracker: Dict[str, Dict[str, int]] = {}
chat_history: Dict[str, List[Dict[str, str]]] = {}
last_answer: Dict[str, str] = {}   # 🆕 Store last answer per IP for PDF

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")

class Question(BaseModel):
    question: str

# Convert verbal math phrases to LaTeX
def convert_to_latex_math(text: str) -> str:
    replacements = {
        r'\b1\s+upon\s+2\b': r'\\frac{1}{2}',
        r'\b1\s+upon\s+3\b': r'\\frac{1}{3}',
        r'\bsquare\s+root\s+of\s+(\w+)': r'\\sqrt{\1}',
        r'\bcube\s+root\s+of\s+(\w+)': r'\\sqrt[3]{\1}',
        r'\bx\s+square\b': r'x^2',
        r'\bx\s+cube\b': r'x^3',
        r'\bx\s+power\s+(\d+)': r'x^{\1}',
        r'\bupon\b': '/',
        r'\btimes\b': r'\\times ',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def detect_language(text: str) -> str:
    lang, _ = langid.classify(text)
    return lang

def reset_if_new_day(ip: str):
    now = time.localtime()
    today = f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}"
    if ip not in ip_usage_tracker or ip_usage_tracker[ip]['date'] != today:
        ip_usage_tracker[ip] = {'count': 0, 'img_count': 0, 'date': today}
        chat_history[ip] = []

def fetch_youtube_videos(query: str, max_results: int = 1):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        items = response.json().get("items", [])
        return [{
            "title": item["snippet"]["title"],
            "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
            "videoId": item["id"]["videoId"]
        } for item in items]
    return []

def ensure_token_safe_response(full_text: str, max_tokens: int = 1500) -> str:
    if len(full_text) / 4 > max_tokens:
        return f"🔍 Summary due to length:\n{summarize_text(full_text, max_tokens)}"
    return full_text

def summarize_text(text: str, max_tokens: int = 1000) -> str:
    words = text.split()
    estimated_limit = max_tokens * 0.75
    return ' '.join(words[:int(estimated_limit)]) + '... (summary)'

# -------------------------
# Helper: call Groq API (grok-1)
# -------------------------
def ask_grok_api(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = 0.7):
    """
    Send chat completion request to Groq (OpenAI-compatible endpoint).
    Returns string response or raises/returns error string.
    """
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        # raise for status to get exceptions on 4xx/5xx
        res.raise_for_status()
        data = res.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            # Return server response for debugging
            return f"❌ No valid choice in Groq response: {data}"
    except requests.exceptions.RequestException as e:
        return f"❌ Error calling Groq API: {str(e)}"
# -------------------------

@app.post("/ask")
async def ask_question(data: Question, request: Request):
    prompt = data.question.strip()
    prompt_lower = prompt.lower()
    ip = request.client.host
    reset_if_new_day(ip)

    if ip_usage_tracker[ip]['count'] >= question_limit_per_day:
        return {"answer": "❌Hey Dear User!, You have reached your Daily limit of 15 questions. Try again tomorrow."}

    ip_usage_tracker[ip]['count'] += 1
    detected_lang = detect_language(prompt)

    founder_keywords = [
        "founder of", "who is your founder", "who founded",
        "who is your owner", "owner of", "who made desh ai",
        "who created desh ai by dsr", "who developed desh ai", "who made you",
        "who made Desh ai", "who made Deshai", "who created you",
        "who create you", "who creates you", "who creates you Desh ai",
        "who created you Deshai"
    ]
    if any(kw in prompt_lower for kw in founder_keywords):
        reply = "𝕯𝖊𝕾𝖍 𝗔𝐢 is founded by 𝕯𝖊𝖛𝖊𝖘𝖍 𝚂𝚒𝚗𝚐𝚑 𝕽𝖆𝖏𝖕𝖚𝖙, a 15-year-old boy from Jaunpur, U.P."
        last_answer[ip] = reply  # 🆕 Save last answer
        return {"answer": reply, "youtube_videos": fetch_youtube_videos(prompt)}

    # -------------------------
    # Image generation removed as requested (no TogetherAI calls remain).
    # We preserve the ip_usage_tracker structure (img_count remains unused),
    # but we do NOT call any image API here.
    # -------------------------

    explain_keywords = ["explain", "describe", "in brief", "long", "elaborate"]
    max_tokens = 650 if any(kw in prompt_lower for kw in explain_keywords) else 1500

    # System prompt: preserve existing behavior
    system_prompt = {
        "role": "system",
        "content": (
            f"You are a smart assistant. The user is speaking in '{detected_lang}' language. Reply in the same language. "
            "Add 5 to 20 meaningful emojis based on tone or topic. Use them naturally between answer. "
            "Preserve user clarity and support LaTeX formatting if present."
        )
    }

    messages = [system_prompt] + chat_history.get(ip, [])[-10:] + [{"role": "user", "content": prompt}]

    # Call Groq API instead of TogetherAI
    try:
        reply = ask_grok_api(messages=messages, max_tokens=max_tokens, temperature=0.7)
        # reply now either string answer or error string starting with ❌
        # Keep existing post-processing
        reply = ensure_token_safe_response(reply, max_tokens)
        reply = convert_to_latex_math(reply)
        if any(x in reply for x in ['\\(', '\\)', '\\[', '\\]', '^', '\\frac', '\\sqrt']):
            reply = reply.strip()
        # update history
        chat_history[ip] = messages + [{"role": "assistant", "content": reply}]
        last_answer[ip] = reply  # 🆕 Save last answer
        return {"answer": reply, "youtube_videos": fetch_youtube_videos(prompt)}
    except Exception as e:
        return {"error": "Error Fetching Answer", "details": str(e), "youtube_videos": fetch_youtube_videos(prompt)}

# 🆕 New PDF route (unchanged)
@app.get("/download-pdf")
async def download_pdf(request: Request):
    ip = request.client.host
    if ip not in last_answer or not last_answer[ip].strip():
        return {"error": "❌ No answer available to convert into PDF."}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(last_answer[ip], styles["Normal"])]
    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=chat_answer.pdf"
    })