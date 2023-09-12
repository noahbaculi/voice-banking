import os
import torch
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples


assert torch.cuda.is_available(), "CUDA is not available"

# we use the same path as this script as our training folder.
output_path = os.path.dirname(os.path.abspath(__file__))

# dataset config for one of the pre-defined datasets
dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="", path=os.path.join(output_path, "dataset"))

breakpoint()

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
