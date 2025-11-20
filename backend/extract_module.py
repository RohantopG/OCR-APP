import os, sys, shutil
from datetime import datetime
from typing import Union

import cv2
import numpy as np
from PIL import Image
import pytesseract

# -------------------------
# TESSERACT SETUP
# -------------------------
_TESSERACT_CONFIGURED = False
def configure_tesseract():
    global _TESSERACT_CONFIGURED
    if _TESSERACT_CONFIGURED:
        return
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if sys.platform == "win32" and os.path.exists(win_path):
        pytesseract.pytesseract.tesseract_cmd = win_path
        os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
    elif shutil.which("tesseract"):
        # ✅ Ensure correct binary path on Linux (Railway/Docker)
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"
        os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
    else:
        raise OSError("Tesseract not found. Install Tesseract and ensure it's in PATH.")
    
    # ✅ Preserve your Render/local tessdata override
    os.environ["TESSDATA_PREFIX"] = os.path.join(os.path.dirname(__file__), "tessdata")

    print(f"[INFO] Tesseract configured: {pytesseract.pytesseract.tesseract_cmd}")
    _TESSERACT_CONFIGURED = True

# Wordlist support
BASE_DIR = os.path.dirname(__file__)
WORDLIST_SOURCE = os.path.join(BASE_DIR, "kannada_wordList_with_freq.txt")
TESSERACT_WORDLIST = os.path.join(BASE_DIR, "kannada_words.txt")
if not os.path.exists(TESSERACT_WORDLIST) and os.path.exists(WORDLIST_SOURCE):
    with open(WORDLIST_SOURCE, "r", encoding="utf-8") as fin, open(TESSERACT_WORDLIST, "w", encoding="utf-8") as fout:
        for line in fin:
            w = line.strip().split(" ")[0]
            if w:
                fout.write(w + "\n")
    print("[INFO] Created Tesseract wordlist:", TESSERACT_WORDLIST)

# -------------------------
# UTILS
# -------------------------
def pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr

def cv2_to_pil(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)) if img_cv.ndim == 3 else Image.fromarray(img_cv)

def ensure_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def upscale_for_dpi(img: np.ndarray, min_height=1200) -> np.ndarray:
    h, w = img.shape[:2]
    if h >= min_height: return img
    scale = min_height / h
    return cv2.resize(img, (int(w*scale), int(min_height)), interpolation=cv2.INTER_CUBIC)

def unsharp_mask(img: np.ndarray, amount=1.2, ksize=(5,5)) -> np.ndarray:
    blur = cv2.GaussianBlur(img, ksize, 0)
    return cv2.addWeighted(img, 1+amount, blur, -amount, 0)

def remove_borders(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(255 - thresh)
    if coords is None: return img
    x, y, w0, h0 = cv2.boundingRect(coords)
    pad = 4
    return img[max(0,y-pad):min(h,y+h0+pad), max(0,x-pad):min(w,x+w0+pad)]

def morphological_clean(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)

def gamma_correction(img: np.ndarray, gamma=1.0) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

# -------------------------
# OCR
# -------------------------
def tesseract_confidence_and_text(img_for_tess: Union[np.ndarray, Image.Image], lang="kan", config="--psm 6 --oem 3"):
    pil = cv2_to_pil(img_for_tess) if isinstance(img_for_tess, np.ndarray) else img_for_tess
    try:
        data = pytesseract.image_to_data(pil, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print("[WARN] pytesseract.image_to_data failed:", e)
        return -1, ""
    texts, confs = [], []
    for i, txt in enumerate(data.get('text', [])):
        t = (txt or "").strip()
        if t: texts.append(t)
        try: c = float(data['conf'][i]); confs.append(c if c>=0 else 0)
        except: pass
    return (sum(confs)/len(confs) if confs else -1, " ".join(texts).strip())

def extract_text(image_source: Union[str, np.ndarray, Image.Image], image_name="temp.jpg", lang="kan") -> str:
    configure_tesseract()

    # Load image
    if isinstance(image_source, str):
        img = pil_to_cv2(Image.open(image_source).convert("RGB"))
    elif isinstance(image_source, Image.Image):
        img = pil_to_cv2(image_source.convert("RGB"))
    else:
        img = image_source.copy()

    gray = ensure_grayscale(img)
    gray = remove_borders(gray)
    gray = upscale_for_dpi(gray, 1200)

    # Variants for better accuracy
    variants = []

    # Variant A: Histogram Equalization + Unsharp
    a = cv2.equalizeHist(gray)
    a = unsharp_mask(a, amount=1.2, ksize=(3,3))
    variants.append(("hist_unsharp", a))

    # Variant B: CLAHE + Median Blur
    b = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    b = cv2.medianBlur(b, 3)
    variants.append(("clahe_median", b))

    # Variant C: Bilateral Filter + Adaptive Threshold + Morphology
    c = cv2.bilateralFilter(gray, 9, 75, 75)
    c = cv2.medianBlur(c, 3)
    block = max(11, int(min(c.shape[:2]) / 16) | 1)
    c = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 2)
    c = morphological_clean(c)
    variants.append(("bilateral_adapt_morph", c))

    # Variant D: Gamma Correction + Histogram
    d = gamma_correction(gray, gamma=0.85)
    d = cv2.equalizeHist(d)
    variants.append(("gamma_hist", d))

    # OCR scanning
    best={"conf":-9999,"text":""}
    base_wordlist=f' --user-words "{TESSERACT_WORDLIST}"' if os.path.exists(TESSERACT_WORDLIST) else ""

    # PSMs and OEMs
    psm_list = [6, 4, 3, 12]  # Added 12 for sparse text
    oem_list = [3, 1]          # LSTM and LSTM+legacy

    for name, var in variants:
        proc = ensure_grayscale(var)
        for psm in psm_list:
            for oem in oem_list:
                config = f"--oem {oem} --psm {psm} --dpi 300" + base_wordlist
                avg_conf, txt = tesseract_confidence_and_text(proc, lang=lang, config=config)
                txt_clean = txt.strip() if txt else ""
                score = (avg_conf if avg_conf>0 else 0) + min(len(txt_clean),300)/300*5
                if score > best["conf"] and txt_clean:
                    best.update({"conf": score, "text": txt_clean})

    return best["text"]
