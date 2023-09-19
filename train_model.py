import os
from typing import Literal

import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.tacotron_config import TacotronConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.tacotron import Tacotron
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    PWD = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(PWD, "output")

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.txt", path=os.path.join(PWD, "dataset")
    )

    train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.03)

    MODEL: Literal["GlowTTS", "Tacotron"] = "GlowTTS"

    if MODEL == "GlowTTS":
        config = GlowTTSConfig(
            batch_size=1,
            eval_batch_size=16,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=1000,
            text_cleaner="phoneme_cleaners",
            use_phonemes=True,
            phoneme_language="en-us",
            phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
            print_step=25,
            print_eval=False,
            mixed_precision=True,
            output_path=output_path,
            datasets=[dataset_config],
        )
    else:
        config = TacotronConfig()

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    if MODEL == "GlowTTS":
        model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    else:
        model = Tacotron(config, ap, tokenizer)

    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()
