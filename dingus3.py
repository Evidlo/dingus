#!/usr/bin/env python3

import auditok
from piper import PiperVoice, download_voices
from pathlib import Path
import ollama
import whisper

# --- language model setup ---

# MODEL = 'tinyllama'
# MODEL = 'qwen3:4b'
# MODEL = 'phi3.5'
# MODEL = 'ministral-3:3b'
MODEL = 'llama3.2:3b'

# Download language model if it isn't already
ollama.pull(MODEL)

SYSTEM = 'You are a helpful assistant running on a HAM radio repeater giving short responses, but willing to talk about any topic.  \
Respond with one or a few sentences with no output styling. Only if you are asked, your callsign is KD9FMW.'

# --- TTS model setup ---

VOICEDIR = Path('voices')

VOICE = 'kokoro-v0_19'
# VOICE = 'en_US-lessac-medium'
# VOICE = 'en_US-lj-medium'
# VOICE = 'en_US-mv2-medium'
# VOICE = 'en_UK-cori-high'

# Download voice model and setup
path = (VOICEDIR / VOICE).with_suffix('.onnx')
if not path.exists():
    download_voices.download_voice(VOICE, VOICEDIR)
tts = PiperVoice.load(path)

# --- STT model setup ---
stt = whisper.load_model("small")

# audio must contain this word to trigger response
TRIGGER_WORD = 'avocado'

# set up wakeword detection
rec = auditok.Recorder(input='input_double.wav', sr=16000, sw=2, ch=1)
# rec = auditok.Recorder(None, sr=16000, sw=2, ch=1)

# wait for detected audio
for region in auditok.split(rec, min_dur=1, max_silence=2, max_dur=100, eth=55):
    region.save('region.wav')
    transcribed = stt.transcribe('region.wav', language='en', fp16=False)['text']

    # attempt to filter out noise recognized erroneously as short phrases
    if len(transcribed.strip().split(' ')) < 3:
        continue

    # wait for trigger word
    if not TRIGGER_WORD in transcribed:
        continue

    # strip out the trigger word
    transcribed = transcribed.split(TRIGGER_WORD, 1)[1]

    print('<<<', transcribed)

    response = ollama.generate(
        model=MODEL,
        system=SYSTEM,
        prompt=transcribed,
        options={
            # 'temperature': 0.9, # Higher for more creativity
            # 'top_k': 50,        # Sample from the top 50 tokens
            # 'num_predict': 100,    # Response length
        },
        stream=False
    )['response']

    # add callsign
    response += ' KD9FMW'

    print(response)

    import wave
    with wave.open("output.wav", "wb") as wav_file:
        tts.synthesize_wav(response, wav_file)

    # player = auditok.player_for(rec)
    # for chunk in tts.synthesize(response):
    #     player.play(chunk.audio_float_array)
