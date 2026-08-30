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
from gtts import gTTS
from mistralai import Mistral

# Load environment variables
load_dotenv()

# -------------------------
# API Keys & Clients
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Initialize Mistral Client safely
client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

if not MISTRAL_API_KEY:
    print("⚠️ WARNING: MISTRAL_API_KEY is missing in Environment Variables!")

HF_IMAGE_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

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
    if not YOUTUBE_API_KEY:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results
    }
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            items = response.json().get("items", [])
            return [{
                "title": item["snippet"]["title"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "videoId": item["id"]["videoId"]
            } for item in items]
    except Exception:
        pass
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
# Mistral API Helper
def ask_grok_api(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = 0.7):
    if not client:
        return "❌ Server configuration error: MISTRAL_API_KEY is not set on host environment."
        
    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
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
        "founder of", "who is your founder", "who made desh ai", "who created you", "creates you", "created you" , "founded you" , "your founder" , "makes you" , "ceo of desh ai" , "owner of desh ai" 
    ]
    if any(kw in prompt_lower for kw in founder_keywords):
        reply = """The Vision Behind 𝕯𝖊𝖘𝖍 𝐀𝖎: This platform is a cutting-edge fully Aí-driven system established in 2025 to democratize advanced technology. Led by 𝗦𝗵𝗿𝗲𝘆𝗮 𝗦𝗶𝗻𝗴𝗵 (𝙲𝙴𝙾), 𝗔𝗵𝗮𝗮𝗻 𝗦𝗶𝗻𝗴𝗵 (𝙲𝚘-𝙵𝚘𝚞𝚗𝚍𝚎𝚛) and 𝗗𝗲𝘃𝗲𝘀𝗵 𝗦𝗶𝗻𝗴𝗵 (Here is a review of your FastAPI application code, along with key observations, potential bug fixes, and optimization recommendations.

---

## 🔍 Code Review & Optimization Highlights

### 1. **Mistral Client Re-initialization (Performance Fix)**
In `ask_grok_api`, the `client = Mistral(api_key=MISTRAL_API_KEY)` call runs inside every single function call. Instantiating the client once globally at module load saves unnecessary setup overhead per HTTP request.

### 2. **Blocking Network Calls (`requests` in Async Routes)**
FastAPI endpoints defined with `async def` run on the main asyncio event loop. Using synchronous `requests.get()` or `requests.post()` inside an `async def` route (e.g., in `/ask`, `/tts`, and HF image generation) **blocks the entire server event loop** for all connected users while waiting for the HTTP response.
* **Fix Options:** Either convert network calls to `httpx.AsyncClient` or define those FastAPI endpoints with standard synchronous functions (`def` instead of `async def`), which forces FastAPI to run them in a background thread pool.

### 3. **Blocking File I/O (`gTTS.save`)**
In `/tts`, `tts.save(filename)` writes directly to disk synchronously. Writing to a static local file name (`tts_output.mp3`) also creates a **race condition** if multiple users request TTS at the exact same time (one request overwrites the file while another is reading it).
* **Fix:** Use `io.BytesIO()` to stream the audio directly from memory rather than writing to a shared file on disk.

### 4. **PDF Generation Memory Streaming**
The PDF download route correctly uses `io.BytesIO()` with `StreamingResponse`. However, plain `Paragraph` objects in ReportLab do not automatically parse HTML/Markdown formatting like standard line breaks or raw markdown symbols. Wrapping long response text in basic standard styles works well for simple text.

---

## 🛠 Refactored Code

Here is the updated code incorporating these stability and performance enhancements:

```python
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
from gtts import gTTS
from mistralai import Mistral  # Official Mistral SDK

# Load environment variables
load_dotenv()

# -------------------------
# API Keys & Clients
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Prevent invalid default keys
if not MISTRAL_API_KEY or MISTRAL_API_KEY == "YOUR_KEY":
    print("⚠️ WARNING: MISTRAL_API_KEY is missing or invalid in Environment Variables!")
    mistral_client = None
else:
    # Initialize client once globally
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

HF_IMAGE_URL = "[https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell](https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell)"

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
    if not YOUTUBE_API_KEY:
        return []
    url = "[https://www.googleapis.com/youtube/v3/search](https://www.googleapis.com/youtube/v3/search)"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "key": YOUTUBE_API_KEY,
        "maxResults": max_results
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            items = response.json().get("items", [])
            return [{
                "title": item["snippet"]["title"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "videoId": item["id"]["videoId"]
            } for item in items]
    except Exception:
        pass
    return []

def summarize_text(text: str, max_tokens: int = 1000) -> str:
    words = text.split()
    estimated_limit = max_tokens * 0.75
    return ' '.join(words[:int(estimated_limit)]) + '... (summary)'

def ensure_token_safe_response(full_text: str, max_tokens: int = 1500) -> str:
    if len(full_text) / 4 > max_tokens:
        return f"🔍 Summary due to length:\n{summarize_text(full_text, max_tokens)}"
    return full_text

# -------------------------
# Mistral API Helper
def ask_grok_api(messages: List[Dict[str, str]], max_tokens: int = 1500, temperature: float = 0.7):
    if not mistral_client:
        return "❌ Server configuration error: MISTRAL_API_KEY is not set on host environment."
        
    try:
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
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
        
        # Ensure static folder exists
        os.makedirs("static", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(image_bytes)

        ip_usage_tracker[ip]['img_count'] += 1
        notice = (
            "🖼 Your image is ready:\n"
            f"[📥 Click to View Image](/static/{file_name})\n\n"
            f"⚠ Limit: {image_limit_per_day} per day"
        )
        last_answer[ip] = notice
        return {"answer": notice, "youtube_videos": fetch_youtube_videos(prompt)}
    return {"error": f"Image generation failed: {res.text}"}

# -------------------------
# Main Chat Route (Using synchronous def to run blocking operations in worker threads)
@app.post("/ask")
def ask_question(data: Question, request: Request):
    prompt = data.question.strip()
    ip = request.client.host
    reset_if_new_day(ip)

    if ip_usage_tracker[ip]['count'] >= question_limit_per_day:
        return {"answer": f"❌ Limit reached ({question_limit_per_day}/day)"}

    ip_usage_tracker[ip]['count'] += 1
    detected_lang = detect_language(prompt)
    prompt_lower = prompt.lower()

    founder_keywords = [
        "founder of", "who is your founder", "who made desh ai", "who created you", 
        "creates you", "created you", "founded you", "your founder", "makes you", 
        "ceo of desh ai", "owner of desh ai" 
    ]
    if any(kw in prompt_lower for kw in founder_keywords):
        reply = (
            "The Vision Behind 𝕯𝖊𝖘𝖍 𝐀𝖎: This platform is a cutting-edge fully AI-driven system "
            "established in 2025 to democratize advanced technology. Led by 𝗦𝗵𝗿𝗲𝘆𝗮 𝗦𝗶𝗻𝗴𝗵 (CEO), "
            "𝗔𝗵𝗮𝗮𝗻 𝗦𝗶𝗻𝗴𝗵 (Co-Founder) and 𝗗𝗲𝘃𝗲𝘀𝗵 𝗦𝗶𝗻𝗴𝗵 (Founder & Managing Director), the company "
            "has evolved into a powerhouse of digital innovation.\n\n"
            "Leadership and Board:\nThe strategic direction is spearheaded by a dynamic duo. "
            "Shreya Singh serves as the CEO and primary architect of the vision and scaling strategies. "
            "Devesh Singh is the Founder & Managing Director and the technical force driving the core architecture "
            "and integration. Whereas Ahaan Singh (Co-Founder) gives his best contribution with Devesh & whole Team DSR "
            "in making DBMS & AI's data training.\n\n"
            "Core Capabilities and Innovations:\nThe platform distinguishes itself through a suite of integrated tools "
            "designed for utility and entertainment. It features high-performance conversational engines with voice synthesis "
            "and real-time response capabilities. The multi-functional utility suite includes dynamic PDF solutions for generators "
            "and editors, and creative tools like FaceTalk in 𝕯𝖊𝖘𝖍 𝐀𝖎 Pro.\n\n"
            "Interactive Entertainment and API:\nUnique projects like the Hand Cricket game utilize pattern-recognition for "
            "a personalized experience. The platform leverages top-tier models ensuring sophisticated language processing for all users.\n\n"
            "The 2025 Milestone:\nFounded during a pivotal year for artificial intelligence, the platform focuses on clean UI "
            "and functional web-based tools to bridge the gap between complex coding and daily needs."
        )
        last_answer[ip] = reply
        return {"answer": reply, "youtube_videos": fetch_youtube_videos(prompt)}

    if any(word in prompt_lower for word in ["shhahshahshdhdhhsh"]):
        if ip_usage_tracker[ip]['img_count'] >= image_limit_per_day:
            return {"answer": f"❌ Image limit reached ({image_limit_per_day}/day)"}
        return generate_image_hf(prompt, ip)

    explain_keywords = ["explain", "describe", "in brief", "long", "elaborate"]
    max_tokens = 2650 if any(kw in prompt_lower for kw in explain_keywords) else 1500

    system_prompt = {
        "role": "system",
        "content": f"You are 𝕯𝖊𝖘𝖍 𝐀𝖎. Reply in {detected_lang} language, using emojis naturally."
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
# TTS Route (Streams in-memory MP3 without writing to shared disk file)
@app.post("/tts")
def text_to_speech(data: dict):
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    lang = detect_language(text)
    if lang not in ["en", "hi"]:
        lang = "en"

    mp3_fp = io.BytesIO()
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    return StreamingResponse(mp3_fp, media_type="audio/mpeg", headers={"Content-Disposition": "inline; filename=tts_output.mp3"})

# -------------------------
# PDF Route
@app.get("/download-pdf")
def download_pdf(request: Request):
    ip = request.client.host
    if ip not in last_answer or not last_answer[ip].strip():
        return JSONResponse({"error": "❌ No answer available for PDF."}, status_code=400)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(last_answer[ip].replace('\n', '<br/>'), styles["Normal"])]
    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=chat_answer.pdf"}
    )

# Static file routes
@app.get("/numpuzz")
def serve_numpuzz(): return FileResponse("static/numpuzz.html")

@app.get("/snake")
def serve_snake(): return FileResponse("static/snake.html")

@app.get("/calculator")
def serve_calculator(): return FileResponse("static/calculator.html")

@app.get("/BMI")
def serve_bmi(): return FileResponse("static/BMI.html")

@app.get("/Dictionary")
def serve_dictionary(): return FileResponse("static/Dictionary.html")

@app.get("/desh.html")
def serve_desh(): return FileResponse("static/desh.html")

@app.get("/Tic")
def serve_tic(): return FileResponse("static/Tic.html")

@app.get("/Tac")
def serve_tac(): return FileResponse("static/Tac.html")

@app.get("/50")
def serve_50(): return FileResponse("static/50.html")

@app.get("/neon")
def serve_neon(): return FileResponse("static/neon.html")

@app.get("/waves")
def serve_waves(): return FileResponse("static/waves.html")

@app.get("/dot")
def serve_dot(): return FileResponse("static/grid.html")

@app.get("/pdf")
def serve_pdf(): return FileResponse("static/pdf.html")

@app.get("/smart")
def serve_smart(): return FileResponse("static/smart.html")

@app.get("/study")
def serve_study(): return FileResponse("static/study.html")

@app.get("/vsics")
def serve_vsics(): return FileResponse("static/vsics.html")

@app.get("/cam")
def serve_cam(): return FileResponse("static/cam.html")

@app.get("/mag")
def serve_mag(): return FileResponse("static/mag.html")

