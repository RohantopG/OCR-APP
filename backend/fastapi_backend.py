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

# ----------------- Security + CORS -----------------
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Static Folders (ABSOLUTE PATH FIX) -----------------
BASE_DIR = os.getcwd()
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


# ----------------- Async Helpers -----------------
async def async_tts(text, lang, filename):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, text_to_speech, text, lang, filename)

async def async_translate(text, target_lang):
    loop = asyncio.get_event_loop()
    translator = GoogleTranslator(source="kn", target=target_lang)
    return await loop.run_in_executor(None, translator.translate, text)


# ----------------- Endpoints -----------------
@app.get("/")
async def root():
    return {"message": "ಚಿತ್ರವಚಕ API is running!", "status": "healthy"}

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/process/")
async def process(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save image
    upload_filename = f"{uuid.uuid4().hex}.jpg"
    upload_path = os.path.join(UPLOADS_DIR, upload_filename)
    with open(upload_path, "wb") as f:
        f.write(contents)

    image = Image.open(io.BytesIO(contents))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # OCR
    text_kn = extract_text(image, upload_filename)

    if not text_kn.strip():
        return {
            "image_url": f"/static/uploads/{upload_filename}",
            "text_kn": "",
            "audio_kn": None,
            "text_en": "",
            "audio_en": None,
            "text_hi": "",
            "audio_hi": None,
            "error": "No text found"
        }

    # Audio paths
    audio_kn_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4().hex}_kn.mp3")
    audio_en_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4().hex}_en.mp3")
    audio_hi_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4().hex}_hi.mp3")

    # Translate + TTS (concurrent)
    trans_en_task = asyncio.create_task(async_translate(text_kn, "en"))
    trans_hi_task = asyncio.create_task(async_translate(text_kn, "hi"))
    tts_kn_task = asyncio.create_task(async_tts(text_kn, "kn", audio_kn_file))

    text_en, text_hi = await asyncio.gather(trans_en_task, trans_hi_task)

    tts_en_task = asyncio.create_task(async_tts(text_en, "en", audio_en_file))
    tts_hi_task = asyncio.create_task(async_tts(text_hi, "hi", audio_hi_file))
    await asyncio.gather(tts_kn_task, tts_en_task, tts_hi_task)

    return {
        "image_url": f"/static/uploads/{upload_filename}",
        "text_kn": text_kn,
        "audio_kn": f"/static/audio/{os.path.basename(audio_kn_file)}",
        "text_en": text_en,
        "audio_en": f"/static/audio/{os.path.basename(audio_en_file)}",
        "text_hi": text_hi,
        "audio_hi": f"/static/audio/{os.path.basename(audio_hi_file)}",
        "error": None
    }


# ----------------- Static Mount (FIXED) -----------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ----------------- Run -----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("backend.fastapi_backend:app", host="0.0.0.0", port=port)
