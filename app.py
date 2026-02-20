from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os, requests, time, langid, re, io
from typing import Dict, List
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# Load environment variables
load_dotenv()

# -------------------------
# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_KEY")
HF_API_KEY = os.getenv("HF_API_KEY", "YOUR_HF_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_YT_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

# -------------------------
# Usage Limits & Storage
image_limit_per_day = 5
question_limit_per_day = 200
ip_usage_tracker: Dict[str, Dict[str, int]] = {}
chat_history: Dict[str, List[Dict[str, str]]] = {}
last_answer: Dict[str, str] = {}  # Store last answer per IP for PDF & TTS

# -------------------------
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

# -------------------------
# Utility functions
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
# Groq API helper
def ask_grok_api(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = 0.7):
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
        res.raise_for_status()
        data = res.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return "❌ Seems like server issue, Try after a while"
    except requests.exceptions.RequestException as e:
        return f"❌ Seems like server issue, Try after a while: {str(e)}"

# -------------------------
# Hugging Face Image helper
def generate_image_hf(prompt: str, ip: str):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {"width": 1024, "height": 768, "num_inference_steps": 30}
    }
    res = requests.post(HF_IMAGE_URL, headers=headers, json=payload, timeout=60)
    if res.status_code == 200:
        image_bytes = res.content
        file_name = f"generated_{int(time.time())}.png"
        file_path = f"static/{file_name}"
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        ip_usage_tracker[ip]['img_count'] += 1
        notice = (
            "🖼 Your image is ready:\n"
            f"[📥 Click to View Image](/static/{file_name})\n\n"
            f"⚠ Limit: {image_limit_per_day} per day"
        )
        reply = notice
        last_answer[ip] = reply
        return {"answer": reply, "youtube_videos": fetch_youtube_videos(prompt)}
    return {"error": f"Image generation failed: {res.text}"}

# -------------------------
# Main Chat Route
@app.post("/ask")
async def ask_question(data: Question, request: Request):
    prompt = data.question.strip()
    ip = request.client.host
    reset_if_new_day(ip)

    if ip_usage_tracker[ip]['count'] >= question_limit_per_day:
        return {"answer": f"❌ Limit reached ({question_limit_per_day}/day)"}

    ip_usage_tracker[ip]['count'] += 1
    detected_lang = detect_language(prompt)
    prompt_lower = prompt.lower()

    founder_keywords = [
        "founder of", "who is your founder", "who made desh ai", "who created you", "Creates you", "created you" , "founded you" , "your founder" , "makes you" , "CEO of desh Ai" , "owner of Desh Ai" 
    ]
    if any(kw in prompt_lower for kw in founder_keywords):
        reply = "Hello! 😊 I'm 𝕯𝖊𝖘𝖍 𝐀𝖎 (A Cloud Based Data trained Artificial Intelligence) .
I'm proudly created by a 16-year-old innovator, 𝕯𝖊𝖛𝖊𝖘𝖍 𝕾𝖎𝖓𝖌𝖍 𝕽𝖆𝖏𝖕𝖚𝖙, from Jaunpur, Uttar Pradesh, India.
We(Whole Team 𝕯𝖊𝖘𝖍 𝐀𝖎) warmly welcome you and truly appreciate your Interest to know about me. Our mission is to make technology smarter, more helpful, and accessible for everyone. We believe innovation has no age limit, and big dreams can start small.
If you would like to visit our very first office in Noida, you are most welcome. We would be happy to connect with you. For any queries, collaborations, suggestions, or support, feel free to contact us through our Official email :- deshai.by.teamdsr@gmail.com .
Thank you for being a part of our journey.
We look forward to growing, learning, and innovating together with you."
        last_answer[ip] = reply
        return {"answer": reply, "youtube_videos": fetch_youtube_videos(prompt)}

    if any(word in prompt_lower for word in ["draw", "image", "picture", "generate"]):
        if ip_usage_tracker[ip]['img_count'] >= image_limit_per_day:
            return {"answer": f"❌ Image limit reached ({image_limit_per_day}/day)"}
        return generate_image_hf(prompt, ip)

    explain_keywords = ["explain", "describe", "in brief", "long", "elaborate"]
    max_tokens = 650 if any(kw in prompt_lower for kw in explain_keywords) else 1500

    system_prompt = {
        "role": "system",
        "content": (
            f"You are Desh AI. Reply in {detected_lang} language, using emojis naturally."
        )
    }

    messages = [system_prompt] + chat_history.get(ip, [])[-10:] + [{"role": "user", "content": prompt}]
    try:
        reply = ask_grok_api(messages, max_tokens=max_tokens)
        reply = ensure_token_safe_response(convert_to_latex_math(reply))
        chat_history[ip] = messages + [{"role": "assistant", "content": reply}]
        last_answer[ip] = reply
        return {"answer": reply, "youtube_videos": fetch_youtube_videos(prompt)}
    except Exception as e:
        return {"error": f"Error fetching answer: {str(e)}"}

# -------------------------
# 🆕 TTS Route (gTTS integration)
@app.post("/tts")
async def text_to_speech(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    lang = detect_language(text)
    if lang not in ["en", "hi"]:
        lang = "en"

    filename = "tts_output.mp3"
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    return FileResponse(filename, media_type="audio/mpeg", filename=filename)

# -------------------------
# PDF Route
@app.get("/download-pdf")
async def download_pdf(request: Request):
    ip = request.client.host
    if ip not in last_answer or not last_answer[ip].strip():
        return {"error": "❌ No answer available for PDF."}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(last_answer[ip], styles["Normal"])]
    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=chat_answer.pdf"}
    )

@app.get("/numpuzz")
def serve_numpuzz():
    return FileResponse("static/numpuzz.html")

@app.get("/snake")
def serve_numpuzz():
    return FileResponse("static/snake.html")

@app.get("/calculator")
def serve_numpuzz():
    return FileResponse("static/calculator.html")


@app.get("/BMI")
def serve_numpuzz():
    return FileResponse("static/BMI.html")



@app.get("/Dictionary")
def serve_numpuzz():
    return FileResponse("static/Dictionary.html")


@app.get("/desh.html")
def serve_numpuzz():
    return FileResponse("static/desh.html")



@app.get("/Tic")
def serve_numpuzz():
    return FileResponse("static/Tic.html")

@app.get("/Tac")
def serve_numpuzz():
    return FileResponse("static/Tac.html")

@app.get("/50")
def serve_numpuzz():
    return FileResponse("static/50.html")

@app.get("/neon")
def serve_numpuzz():
    return FileResponse("static/neon.html")

@app.get("/waves")
def serve_numpuzz():
    return FileResponse("static/wave.html")

















