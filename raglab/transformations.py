from PIL import Image
from pdf2image import convert_from_path
from pydub import AudioSegment, silence
import pytesseract
import os

def extract_text_from_pdf(pdf_path: str, lang='fra') -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for page_number, image in enumerate(pages):
        text = pytesseract.image_to_string(image, lang=lang)
        full_text += text

    return full_text

def convert_voice_to_text(voice_file, model="openai/whisper-medium"):
    whisper = pipeline("automatic-speech-recognition", model=model)
    text = whisper(voice_file.read())
    del whisper
    return text