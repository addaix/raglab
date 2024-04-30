from PIL import Image
from pdf2image import convert_from_path
from pydub import AudioSegment, silence
import pytesseract

def extract_text_from_pdf(pdf_path: str) -> str:
    pages = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    for page_number, image in enumerate(pages):
        text = pytesseract.image_to_string(image, lang='eng')
        full_text += text
        image.save(f'page_{page_number}.jpg')

    with open('extracted_text.txt', 'w', encoding='utf-8') as file:
        file.write(full_text)


def remove_silence_from_audio(audio_file_path, silence_threshold=-50.0, chunk_size=10):
    audio = AudioSegment.from_file(audio_file_path)

    non_silent_parts = silence.detect_nonsilent(
        audio, min_silence_len=chunk_size, silence_thresh=silence_threshold
    )

    processed_audio = AudioSegment.silent(duration=0)

    for start, end in non_silent_parts:
        processed_audio += audio[start:end]

    return processed_audio