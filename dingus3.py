#!/usr/bin/env python3

import auditok
import ollama
import queue
import requests
import threading
import time
import urllib.parse
import wave
from faster_whisper import WhisperModel
from pathlib import Path
from piper import PiperVoice, download_voices
from subprocess import run

# --- language model setup ---

# MODEL = 'tinyllama'
# MODEL = 'qwen3:4b'
# MODEL = 'qwen2:0.5b'
# MODEL = 'qwen3:1.7b'
MODEL = 'qwen3.5:0.8b'

# Download language model if it isn't already
ollama.pull(MODEL)

SYSTEM = 'You are a helpful assistant running on a HAM radio repeater giving short responses, \
but willing to talk about any topic.  \
Respond with one or a few sentences with no output styling. Only if you are asked, your callsign is KD9FMW.'

# --- TTS model setup ---

VOICEDIR = Path('voices')

# VOICE = 'en_US-lessac-low'
# VOICE = 'en_US-ryan-high'
VOICE = 'en_US-lessac-medium'

# Download voice model and setup
path = (VOICEDIR / VOICE).with_suffix('.onnx')
if not path.exists():
    VOICEDIR.mkdir(parents=True, exist_ok=True)
    download_voices.download_voice(VOICE, VOICEDIR)
tts = PiperVoice.load(path)

# --- STT model setup ---

# distil-small.en is ~1.4x faster but hears the trigger word as "Avocato",
# so the wakeword never matches
# STT_MODEL = 'distil-small.en'
STT_MODEL = 'small'

stt = WhisperModel(STT_MODEL, device='cpu', compute_type='int8')

# --- transcript mirroring ---

# received audio is prefixed '<', spoken responses '>'
LOGFILE = Path('/srv/www/recognized.txt')
MATRIX_ROOM = '!PTZyXwJHptPcxTojOK:matrix.org'
MATRIX_API = 'https://matrix.org/_matrix/client/v3'
MATRIX_TOKEN = Path('~/.local/matrix_token').expanduser()

token = MATRIX_TOKEN.read_text().strip() if MATRIX_TOKEN.exists() else ''
room = urllib.parse.quote(MATRIX_ROOM, safe='')
outbox = queue.Queue()


def post_to_matrix():
    """Drain the outbox so a slow homeserver never stalls recognition."""
    while True:
        line = outbox.get()
        try:
            posted = requests.put(
                f'{MATRIX_API}/rooms/{room}/send/m.room.message/{time.time_ns()}',
                headers={'Authorization': f'Bearer {token}'},
                json={'msgtype': 'm.text', 'body': line},
                timeout=10,
            )
            posted.raise_for_status()
        except Exception as e:
            print('matrix send failed:', e)


threading.Thread(target=post_to_matrix, daemon=True).start()


def mirror(line):
    """Send one transcript line to the console, the log file and the Matrix room."""
    print(line, flush=True)
    with LOGFILE.open('a') as log:
        log.write(line + '\n')

    # posting stays disabled until an access token is installed
    if token:
        outbox.put(line)


print(f'matrix posting to {MATRIX_ROOM}:', bool(token))

# audio must contain this word to trigger response
TRIGGER_WORD = 'avocado'

# the 440/880 pair acknowledges the trigger word so the speaker knows it was heard;
# the 440 alone leads each response, giving VOX time to key up before speech starts
ACK_TONES = 'play -n -c1 synth sin 440 fade h 0.1 .4 .1 : synth sin 880 fade h 0.1 .2 0.1'
VOX_TONE = 'play -n -c1 synth sin 440 fade h 0.1 .4 .1'

# set up wakeword detection
# rec = auditok.Recorder(input='input_double.wav', sr=16000, sw=2, ch=1)
source = None # microphone

# wait for detected audio
for region in auditok.split(source, sw=2, ch=1, sr=16000, min_dur=1, max_silence=2, max_dur=100, eth=55):
    region.save('region.wav')
    # vad_filter runs Silero first, which drops repeater Morse and other non-speech
    segments, _ = stt.transcribe('region.wav', language='en', beam_size=1, vad_filter=True)
    transcribed = ' '.join(segment.text for segment in segments).strip()

    # attempt to filter out noise recognized erroneously as short phrases
    if len(transcribed.split(' ')) < 3:
        continue

    mirror(f'< {transcribed}')

    # whisper capitalizes sentence-initial words, so match the trigger case insensitively
    spoken = transcribed.lower()
    if TRIGGER_WORD not in spoken:
        continue

    # acknowledge the trigger word
    run(ACK_TONES, shell=True)

    # strip out the trigger word and everything before it
    prompt = transcribed[spoken.index(TRIGGER_WORD) + len(TRIGGER_WORD):].lstrip(' ,.')

    response = ollama.generate(
        model=MODEL,
        system=SYSTEM,
        prompt=prompt,
        think=False,
        stream=False,
        options={
            # 'temperature': 0.9, # Higher for more creativity
            # 'num_predict': 100, # Response length
        },
    )['response'].strip()

    # add callsign, spelled as letters so espeak does not read "eff" as e-f-f
    response += ' KD9FMW'

    mirror(f'> {response}')

    with wave.open('response.wav', 'wb') as wav_file:
        tts.synthesize_wav(response, wav_file)

    run(VOX_TONE, shell=True)
    run('aplay -q response.wav', shell=True)
