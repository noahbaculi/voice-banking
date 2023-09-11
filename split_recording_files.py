from pydub import AudioSegment, silence

num_phrases = 38
full_recording_path = r"recording/Recording_090923_12-22PM.wav"
full_recording = AudioSegment.from_wav(full_recording_path)
non_silent_clips = silence.split_on_silence(
    full_recording,
    min_silence_len=1000,
    silence_thresh=-100,
    seek_step=100,
    keep_silence=200,
)

assert (
    len(non_silent_clips) == num_phrases
), f"{len(non_silent_clips) = } should equal {num_phrases = }. Try adjusting the silence detection parameters."

# breakpoint()
for idx, non_silent_section in enumerate(non_silent_clips):
    phrase_number = idx + 1
    non_silent_section.export(f"dataset/wavs/phrase_{phrase_number}.wav")
