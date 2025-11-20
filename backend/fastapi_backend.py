import os, pytesseract, uuid, io, asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
from deep_translator import GoogleTranslator

from .extract_module import extract_text
from .text_to_speech import text_to_speech

# ----------------- Tesseract -----------------
TESSDATA_DIR = os.getenv("TESSDATA_PREFIX", "./tessdata")
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

# ----------------- FastAPI -----------------
app = FastAPI(title="ಚಿತ್ರವಚಕ API", version="1.3.5")

# ----------------- Security + CORS Middleware -----------------
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# ✅ Make sure these match your deployed frontend URLs
origins = [
    "https://magnificent-lamington-9d1886.netlify.app",  # ✅ your current live frontend
    "https://chitravachaka-production.up.railway.app",  # your backend domain
    "http://localhost:3000",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # ✅ ensures fetch + preflight requests succeed
)

# ----------------- Static folders -----------------
os.makedirs("static/audio", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# ----------------- Async helpers -----------------
async def async_tts(text, lang, filename):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, text_to_speech, text, lang, filename)

async def async_translate(text, target_lang):
    loop = asyncio.get_event_loop()
    translator = GoogleTranslator(source='kn', target=target_lang)
    return await loop.run_in_executor(None, translator.translate, text)

# ----------------- Endpoints -----------------
@app.get("/")
async def root():
    return {"message": "ಚಿತ್ರವಚಕ API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    upload_filename = f"{uuid.uuid4().hex}.jpg"
    upload_path = f"static/uploads/{upload_filename}"
    with open(upload_path, "wb") as f:
        f.write(contents)

    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB':
        image = image.convert('RGB')

    text_kn = extract_text(image, upload_filename)
    if not text_kn.strip():
        return JSONResponse({
            "image_url": f"/static/uploads/{upload_filename}",
            "text_kn": "",
            "audio_kn": None,
            "text_en": "",
            "audio_en": None,
            "text_hi": "",
            "audio_hi": None,
            "error": "No text found"
        })

    # Audio files
    audio_kn_file = f"static/audio/{uuid.uuid4().hex}_kn.mp3"
    audio_en_file = f"static/audio/{uuid.uuid4().hex}_en.mp3"
    audio_hi_file = f"static/audio/{uuid.uuid4().hex}_hi.mp3"

    # Run translation + TTS concurrently
    trans_en_task = asyncio.create_task(async_translate(text_kn, 'en'))
    trans_hi_task = asyncio.create_task(async_translate(text_kn, 'hi'))
    tts_kn_task = asyncio.create_task(async_tts(text_kn, 'kn', audio_kn_file))

    text_en, text_hi = await asyncio.gather(trans_en_task, trans_hi_task)
    tts_en_task = asyncio.create_task(async_tts(text_en, 'en', audio_en_file))
    tts_hi_task = asyncio.create_task(async_tts(text_hi, 'hi', audio_hi_file))
    await asyncio.gather(tts_kn_task, tts_en_task, tts_hi_task)

    return {
        "image_url": f"/static/uploads/{upload_filename}",
        "text_kn": text_kn,
        "audio_kn": f"/{audio_kn_file}",
        "text_en": text_en,
        "audio_en": f"/{audio_en_file}",
        "text_hi": text_hi,
        "audio_hi": f"/{audio_hi_file}",
        "error": None
    }

# ----------------- Serve static files -----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------- Run locally / Railway -----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("backend.fastapi_backend:app", host="0.0.0.0", port=port)
