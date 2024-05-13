'''Audio related utils'''

def remove_silence_from_audio(audio_file_path, silence_threshold=-50.0, chunk_size=10):
    audio = AudioSegment.from_file(audio_file_path)

    non_silent_parts = silence.detect_nonsilent(
        audio, min_silence_len=chunk_size, silence_thresh=silence_threshold
    )

    processed_audio = AudioSegment.silent(duration=0)

    for start, end in non_silent_parts:
        processed_audio += audio[start:end]

    return processed_audio