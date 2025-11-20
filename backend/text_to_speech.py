# backend/text_to_speech.py
from gtts import gTTS
from io import BytesIO
import time
import os

def text_to_speech(text: str, lang: str = 'kn', filename: str = None) -> bytes:
    """
    Convert text into speech using the gTTS API.
    Retries on failure to handle potential rate-limiting.
    Returns the audio bytes. If a filename is provided, it also saves the audio.
    """
    if not text.strip():
        raise ValueError("❌ No text provided for TTS")

    retries = 3
    for attempt in range(retries):
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Create bytes in memory
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            audio_bytes = fp.read()

            # Save to file if filename provided
            if filename:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'wb') as f:
                    f.write(audio_bytes)
                print(f"[TTS] Audio saved to: {filename}")
            
            return audio_bytes

        except Exception as e:
            print(f"[TTS] Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(1.5)  # Wait before the next attempt
            else:
                error_message = (
                    "Failed to generate audio with gTTS after several attempts. "
                    "This can be due to network issues or rate-limiting by Google's API. "
                    "Please wait a moment and try again."
                )
                raise ConnectionError(f"❌ {error_message} Original error: {e}")