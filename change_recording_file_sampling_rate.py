import os
from typing import Final

import librosa
import soundfile

if __name__ == "__main__":
    PWD: Final[str] = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(PWD, "dataset")
    DESIRED_SAMPLE_RATE: Final[int] = 22050

    for phrase_num in range(1, 38 + 1):
        input_path = os.path.join(dataset_path, "wavs_sample_rate_44100", f"phrase_{phrase_num}.wav")
        audio, input_sample_rate = librosa.load(input_path, sr=None)

        output_path = os.path.join(dataset_path, f"wavs", f"phrase_{phrase_num}.wav")
        soundfile.write(output_path, audio, DESIRED_SAMPLE_RATE)

        print(f"Resampled {input_path} @ {input_sample_rate} Hz -> {output_path} @ {DESIRED_SAMPLE_RATE} Hz")

        _, check_output_sample_rate = librosa.load(output_path, sr=None)
        assert (
            check_output_sample_rate == DESIRED_SAMPLE_RATE
        ), "The output sample rate does not match the desired sample rate"
