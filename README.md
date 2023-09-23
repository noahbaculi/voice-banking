# voice-banking

## Process

1. Create individual `.wav` files for each phrase.

```shell
python split_recording_files.py
```

2. Train model

```shell
python train_model.py
```

3. Synthesize speech

```shell
tts --text "Hi there" --out_path output.wav --model_name "tts_models/multilingual/multi-dataset/bark"
tts --text "Hi there" --out_path output.wav --model_path output/run-September-23-2023_08+49AM-3968b26/best_model.pth --config_path output/run-September-23-2023_08+49AM-3968b26/config.json
``````
