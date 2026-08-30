from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import os
import requests
import time
import langid
import re
import io

from typing import Dict, List

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from gtts import gTTS

# IMPORTANT:
# Mistral SDK v2.x uses this import path.
from mistralai.client import Mistral


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# API KEYS & CLIENTS
# ============================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# Mistral client
# Do NOT crash the whole server if the API key is missing.
if not MISTRAL_API_KEY or MISTRAL_API_KEY == "YOUR_KEY":
    print(
        "WARNING: MISTRAL_API_KEY is missing or invalid "
        "in Environment Variables."
    )
    mistral_client = None
else:
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        print("Mistral client initialized successfully.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Mistral client: {e}")
        mistral_client = None


# ============================================================
# EXTERNAL API CONFIGURATION
# ============================================================

HF_IMAGE_URL = (
    "https://api-inference.huggingface.co/models/"
    "black-forest-labs/FLUX.1-schnell"
)


# ============================================================
# USAGE LIMITS & IN-MEMORY STORAGE
# ============================================================

image_limit_per_day = 5
question_limit_per_day = 200

ip_usage_tracker: Dict[str, Dict[str, int]] = {}
chat_history: Dict[str, List[Dict[str, str]]] = {}

# Store last answer per IP for PDF & TTS-related functionality.
last_answer: Dict[str, str] = {}


# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Make sure static directory exists before mounting it.
os.makedirs("static", exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def serve_homepage():
    return FileResponse("static/index.html")


# ============================================================
# REQUEST MODELS
# ============================================================

class Question(BaseModel):
    question: str


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def convert_to_latex_math(text: str) -> str:
    """
    Converts common spoken mathematical expressions
    into simple LaTeX-compatible notation.
    """

    replacements = {
        r"\b1\s+upon\s+2\b": r"\\frac{1}{2}",
        r"\b1\s+upon\s+3\b": r"\\frac{1}{3}",
        r"\bsquare\s+root\s+of\s+(\w+)": r"\\sqrt{\1}",
        r"\bcube\s+root\s+of\s+(\w+)": r"\\sqrt[3]{\1}",
        r"\bx\s+square\b": r"x^2",
        r"\bx\s+cube\b": r"x^3",
        r"\bx\s+power\s+(\d+)": r"x^{\1}",
        r"\bupon\b": "/",
        r"\btimes\b": r"\\times ",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE
        )

    return text


def detect_language(text: str) -> str:
    """
    Detect the language of the user's question.
    """

    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return "en"


def reset_if_new_day(ip: str):
    """
    Reset question/image usage and chat history
    when a new day starts.
    """

    now = time.localtime()

    today = (
        f"{now.tm_year}-"
        f"{now.tm_mon}-"
        f"{now.tm_mday}"
    )

    if (
        ip not in ip_usage_tracker
        or ip_usage_tracker[ip].get("date") != today
    ):
        ip_usage_tracker[ip] = {
            "count": 0,
            "img_count": 0,
            "date": today
        }

        chat_history[ip] = []

        # Remove old answer as well so a new day starts cleanly.
        last_answer.pop(ip, None)


# ============================================================
# YOUTUBE SEARCH
# ============================================================

def fetch_youtube_videos(
    query: str,
    max_results: int = 1
):
    """
    Fetch YouTube videos using YouTube Data API.
    Returns an empty list if the API key is unavailable
    or the request fails.
    """

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
        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        if response.status_code != 200:
            return []

        items = response.json().get("items", [])

        videos = []

        for item in items:
            video_id = item.get("id", {}).get("videoId")

            snippet = item.get("snippet", {})

            thumbnails = snippet.get(
                "thumbnails",
                {}
            )

            medium_thumbnail = thumbnails.get(
                "medium",
                {}
            ).get("url")

            if not video_id:
                continue

            videos.append({
                "title": snippet.get(
                    "title",
                    "YouTube Video"
                ),
                "thumbnail": medium_thumbnail,
                "videoId": video_id
            })

        return videos

    except Exception as e:
        print(f"YouTube API error: {e}")
        return []


# ============================================================
# TEXT LENGTH / TOKEN SAFETY
# ============================================================

def summarize_text(
    text: str,
    max_tokens: int = 1000
) -> str:

    words = text.split()

    estimated_limit = int(max_tokens * 0.75)

    shortened = words[:estimated_limit]

    if not shortened:
        return ""

    return (
        " ".join(shortened)
        + "... (summary)"
    )


def ensure_token_safe_response(
    full_text: str,
    max_tokens: int = 1500
) -> str:

    if not full_text:
        return ""

    # Rough estimate:
    # approximately 4 characters per token.
    estimated_tokens = len(full_text) / 4

    if estimated_tokens > max_tokens:
        return (
            "🔍 Summary due to length:\n"
            + summarize_text(
                full_text,
                max_tokens
            )
        )

    return full_text


# ============================================================
# MISTRAL API HELPER
# ============================================================

def ask_mistral_api(
    messages: List[Dict[str, str]],
    max_tokens: int = 1500,
    temperature: float = 0.7
):
    """
    Send chat messages to Mistral.

    This replaces the previous Grok/Groq helper while
    preserving the same role inside the application.
    """

    if not mistral_client:
        return (
            "❌ Server configuration error: "
            "MISTRAL_API_KEY is not set correctly "
            "on the host environment."
        )

    try:
        response = mistral_client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not response.choices:
            return (
                "❌ Mistral returned an empty response."
            )

        content = response.choices[0].message.content

        if content is None:
            return (
                "❌ Mistral returned an empty response."
            )

        return str(content)

    except Exception as e:
        print(f"Mistral API error: {e}")

        return (
            "❌ Seems like server issue, "
            "Try after a while: "
            f"{str(e)}"
        )


# ============================================================
# HUGGING FACE IMAGE GENERATION
# ============================================================

def generate_image_hf(
    prompt: str,
    ip: str
):
    """
    Generate an image using Hugging Face FLUX.
    """

    if not HF_API_KEY:
        return {
            "error": (
                "❌ HF_API_KEY is not configured "
                "on the server."
            )
        }

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": 1024,
            "height": 768,
            "num_inference_steps": 30
        }
    }

    try:
        res = requests.post(
            HF_IMAGE_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if res.status_code == 200:

            image_bytes = res.content

            os.makedirs(
                "static",
                exist_ok=True
            )

            file_name = (
                f"generated_{int(time.time())}.png"
            )

            file_path = os.path.join(
                "static",
                file_name
            )

            with open(
                file_path,
                "wb"
            ) as f:
                f.write(image_bytes)

            # Increment image usage only after
            # successful image generation.
            ip_usage_tracker[ip]["img_count"] += 1

            notice = (
                "🖼 Your image is ready:\n"
                f"[📥 Click to View Image]"
                f"(/static/{file_name})\n\n"
                f"⚠ Limit: "
                f"{image_limit_per_day} per day"
            )

            last_answer[ip] = notice

            return {
                "answer": notice,
                "youtube_videos":
                    fetch_youtube_videos(prompt)
            }

        # Hugging Face can return useful JSON error details.
        try:
            error_details = res.json()
        except Exception:
            error_details = res.text

        return {
            "error": (
                "❌ Image generation failed: "
                f"{error_details}"
            )
        }

    except requests.Timeout:
        return {
            "error": (
                "❌ Image generation timed out. "
                "Please try again."
            )
        }

    except Exception as e:
        print(
            f"Hugging Face image error: {e}"
        )

        return {
            "error": (
                "❌ Image generation failed: "
                f"{str(e)}"
            )
        }


# ============================================================
# MAIN CHAT ROUTE
# ============================================================

@app.post("/ask")
def ask_question(
    data: Question,
    request: Request
):

    prompt = data.question.strip()

    if not prompt:
        return {
            "answer": "❌ Please enter a question."
        }

    # Safely obtain client IP.
    client = request.client

    if client is not None:
        ip = client.host
    else:
        ip = "unknown"

    reset_if_new_day(ip)

    # --------------------------------------------------------
    # QUESTION LIMIT
    # --------------------------------------------------------

    if (
        ip_usage_tracker[ip]["count"]
        >= question_limit_per_day
    ):
        return {
            "answer": (
                f"❌ Limit reached "
                f"({question_limit_per_day}/day)"
            )
        }

    # Count the request.
    ip_usage_tracker[ip]["count"] += 1

    # --------------------------------------------------------
    # LANGUAGE DETECTION
    # --------------------------------------------------------

    detected_lang = detect_language(prompt)

    prompt_lower = prompt.lower()

    # --------------------------------------------------------
    # FOUNDER / ABOUT DESH AI RESPONSE
    # --------------------------------------------------------

    founder_keywords = [
        "founder of",
        "who is your founder",
        "who made desh ai",
        "who created you",
        "creates you",
        "created you",
        "founded you",
        "your founder",
        "makes you",
        "ceo of desh ai",
        "owner of desh ai"
    ]

    if any(
        kw in prompt_lower
        for kw in founder_keywords
    ):

        reply = (
            "The Vision Behind 𝕯𝖊𝖘𝖍 𝐀𝖎: "
            "This platform is a cutting-edge "
            "fully AI-driven system established "
            "in 2025 to democratize advanced "
            "technology. Led by 𝗦𝗵𝗿𝗲𝘆𝗮 𝗦𝗶𝗻𝗴𝗵 "
            "(CEO), 𝗔𝗵𝗮𝗮𝗻 𝗦𝗶𝗻𝗴𝗵 (Co-Founder) "
            "and 𝗗𝗲𝘃𝗲𝘀𝗵 𝗦𝗶𝗻𝗴𝗵 "
            "(Founder & Managing Director), "
            "the company has evolved into a "
            "powerhouse of digital innovation.\n\n"

            "Leadership and Board:\n"
            "The strategic direction is spearheaded "
            "by a dynamic duo. Shreya Singh serves "
            "as the CEO and primary architect of "
            "the vision and scaling strategies. "
            "Devesh Singh is the Founder & Managing "
            "Director and the technical force driving "
            "the core architecture and integration. "
            "Whereas Ahaan Singh (Co-Founder) gives "
            "his best contribution with Devesh & "
            "whole Team DSR in making DBMS & AI's "
            "data training.\n\n"

            "Core Capabilities and Innovations:\n"
            "The platform distinguishes itself "
            "through a suite of integrated tools "
            "designed for utility and entertainment. "
            "It features high-performance "
            "conversational engines with voice "
            "synthesis and real-time response "
            "capabilities. The multi-functional "
            "utility suite includes dynamic PDF "
            "solutions for generators and editors, "
            "and creative tools like FaceTalk in "
            "𝕯𝖊𝖘𝖍 𝐀𝖎 Pro.\n\n"

            "Interactive Entertainment and API:\n"
            "Unique projects like the Hand Cricket "
            "game utilize pattern-recognition for "
            "a personalized experience. The platform "
            "leverages top-tier models ensuring "
            "sophisticated language processing "
            "for all users.\n\n"

            "The 2025 Milestone:\n"
            "Founded during a pivotal year for "
            "artificial intelligence, the platform "
            "focuses on clean UI and functional "
            "web-based tools to bridge the gap "
            "between complex coding and daily needs."
        )

        last_answer[ip] = reply

        return {
            "answer": reply,
            "youtube_videos":
                fetch_youtube_videos(prompt)
        }

    # --------------------------------------------------------
    # IMAGE GENERATION TRIGGER
    # --------------------------------------------------------

    if any(
        word in prompt_lower
        for word in ["shhahshahshdhdhhsh"]
    ):

        if (
            ip_usage_tracker[ip]["img_count"]
            >= image_limit_per_day
        ):
            return {
                "answer": (
                    f"❌ Image limit reached "
                    f"({image_limit_per_day}/day)"
                )
            }

        return generate_image_hf(
            prompt,
            ip
        )

    # --------------------------------------------------------
    # RESPONSE LENGTH
    # --------------------------------------------------------

    explain_keywords = [
        "explain",
        "describe",
        "in brief",
        "long",
        "elaborate"
    ]

    if any(
        kw in prompt_lower
        for kw in explain_keywords
    ):
        max_tokens = 2650
    else:
        max_tokens = 1500

    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    system_prompt = {
        "role": "system",
        "content": (
            "You are 𝕯𝖊𝖘𝖍 𝐀𝖎. "
            f"Reply in {detected_lang} language, "
            "using emojis naturally."
        )
    }

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    previous_messages = chat_history.get(
        ip,
        []
    )[-10:]

    messages = (
        [system_prompt]
        + previous_messages
        + [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # --------------------------------------------------------
    # MISTRAL REQUEST
    # --------------------------------------------------------

    try:

        reply = ask_mistral_api(
            messages,
            max_tokens=max_tokens,
            temperature=0.7
        )

        reply = convert_to_latex_math(
            reply
        )

        reply = ensure_token_safe_response(
            reply,
            max_tokens=max_tokens
        )

        # Save assistant response in history.
        chat_history[ip] = (
            messages
            + [
                {
                    "role": "assistant",
                    "content": reply
                }
            ]
        )

        last_answer[ip] = reply

        return {
            "answer": reply,
            "youtube_videos":
                fetch_youtube_videos(prompt)
        }

    except Exception as e:

        print(
            f"Error in /ask route: {e}"
        )

        return {
            "error": (
                f"Error fetching answer: {str(e)}"
            )
        }


# ============================================================
# TEXT-TO-SPEECH ROUTE
# ============================================================

@app.post("/tts")
def text_to_speech(data: dict):

    text = data.get(
        "text",
        ""
    )

    if not text:
        return JSONResponse(
            {
                "error":
                    "No text provided"
            },
            status_code=400
        )

    lang = detect_language(text)

    # gTTS support used by your existing system.
    if lang not in ["en", "hi"]:
        lang = "en"

    try:

        mp3_fp = io.BytesIO()

        tts = gTTS(
            text=text,
            lang=lang,
            slow=False
        )

        tts.write_to_fp(
            mp3_fp
        )

        mp3_fp.seek(0)

        return StreamingResponse(
            mp3_fp,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition":
                    "inline; "
                    "filename=tts_output.mp3"
            }
        )

    except Exception as e:

        print(
            f"TTS error: {e}"
        )

        return JSONResponse(
            {
                "error":
                    f"TTS generation failed: {str(e)}"
            },
            status_code=500
        )


# ============================================================
# PDF ROUTE
# ============================================================

@app.get("/download-pdf")
def download_pdf(
    request: Request
):

    client = request.client

    if client is not None:
        ip = client.host
    else:
        ip = "unknown"

    if (
        ip not in last_answer
        or not last_answer[ip].strip()
    ):
        return JSONResponse(
            {
                "error":
                    "❌ No answer available for PDF."
            },
            status_code=400
        )

    try:

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer
        )

        styles = getSampleStyleSheet()

        # Escape HTML-sensitive characters so
        # generated answers don't accidentally
        # break ReportLab Paragraph parsing.
        answer_text = last_answer[ip]

        answer_text = (
            answer_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        story = [
            Paragraph(
                answer_text,
                styles["Normal"]
            )
        ]

        doc.build(story)

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    "attachment; "
                    "filename=chat_answer.pdf"
            }
        )

    except Exception as e:

        print(
            f"PDF generation error: {e}"
        )

        return JSONResponse(
            {
                "error":
                    f"PDF generation failed: {str(e)}"
            },
            status_code=500
        )


# ============================================================
# STATIC PAGE ROUTES
# ============================================================

@app.get("/numpuzz")
def serve_numpuzz():
    return FileResponse(
        "static/numpuzz.html"
    )


@app.get("/snake")
def serve_snake():
    return FileResponse(
        "static/snake.html"
    )


@app.get("/calculator")
def serve_calculator():
    return FileResponse(
        "static/calculator.html"
    )


@app.get("/BMI")
def serve_bmi():
    return FileResponse(
        "static/BMI.html"
    )


@app.get("/Dictionary")
def serve_dictionary():
    return FileResponse(
        "static/Dictionary.html"
    )


@app.get("/desh.html")
def serve_desh():
    return FileResponse(
        "static/desh.html"
    )


@app.get("/Tic")
def serve_tic():
    return FileResponse(
        "static/Tic.html"
    )


@app.get("/Tac")
def serve_tac():
    return FileResponse(
        "static/Tac.html"
    )


@app.get("/50")
def serve_50():
    return FileResponse(
        "static/50.html"
    )


@app.get("/neon")
def serve_neon():
    return FileResponse(
        "static/neon.html"
    )


@app.get("/waves")
def serve_waves():
    return FileResponse(
        "static/waves.html"
    )


@app.get("/dot")
def serve_dot():
    return FileResponse(
        "static/grid.html"
    )


@app.get("/pdf")
def serve_pdf():
    return FileResponse(
        "static/pdf.html"
    )


@app.get("/smart")
def serve_smart():
    return FileResponse(
        "static/smart.html"
    )


@app.get("/study")
def serve_study():
    return FileResponse(
        "static/study.html"
    )


@app.get("/vsics")
def serve_vsics():
    return FileResponse(
        "static/vsics.html"
    )


@app.get("/cam")
def serve_cam():
    return FileResponse(
        "static/cam.html"
    )


@app.get("/mag")
def serve_mag():
    return FileResponse(
        "static/mag.html"
        )
