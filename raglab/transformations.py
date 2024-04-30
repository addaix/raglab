from PIL import Image
from pdf2image import convert_from_path
from pydub import AudioSegment, silence
import pytesseract
import os

def extract_text_from_pdf(pdf_path: str, lang='fr') -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for page_number, image in enumerate(pages):
        text = pytesseract.image_to_string(image, lang=lang)
        full_text += text

    return full_text



def remove_silence_from_audio(audio_file_path, silence_threshold=-50.0, chunk_size=10):
    audio = AudioSegment.from_file(audio_file_path)

    non_silent_parts = silence.detect_nonsilent(
        audio, min_silence_len=chunk_size, silence_thresh=silence_threshold
    )

    processed_audio = AudioSegment.silent(duration=0)

    for start, end in non_silent_parts:
        processed_audio += audio[start:end]

    return processed_audio