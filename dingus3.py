#!/usr/bin/env python3

import auditok
import numpy as np
import ollama
import queue
import requests
import shutil
import threading
import time
import urllib.parse
import wave
from faster_whisper import WhisperModel
from pathlib import Path
from piper import PiperVoice, download_voices
from silero_vad import load_silero_vad, get_speech_timestamps
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

# STT_MODEL = 'small'
STT_MODEL = 'distil-small.en'

stt = WhisperModel(STT_MODEL, device='cpu', compute_type='int8')

# --- voice activity detection ---

vad = load_silero_vad()

# silero is recurrent, and a full-scale transient (a tongue click, a squelch
# burst) poisons its state for the remainder of a call, hiding speech that
# follows.  Check short chunks with the state reset between them instead.
VAD_CHUNK = 2 * 16000


def has_speech(region):
    """True when any chunk of the region contains speech."""
    samples = region.numpy()[0] / 32768
    for start in range(0, samples.size, VAD_CHUNK):
        chunk = samples[start:start + VAD_CHUNK]
        vad.reset_states()
        if chunk.size > 4000 and get_speech_timestamps(chunk, vad, sampling_rate=16000):
            return True

    return False


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

# audio must contain one of these words to trigger a response;
# distil-small.en hears "avocado" as "avocato" often enough to accept both
TRIGGER_WORDS = ('avocado', 'avocato')

# every detected region is kept for debugging and tuning; regions carrying a
# trigger word are saved a second time under the same timestamp
RECORDINGS = Path('recordings')
RECORDINGS.mkdir(parents=True, exist_ok=True)

# fixed paths to the newest of each, so the most recent audio can be grabbed
# without looking up a timestamp
LAST_VOICE = 'last_voice.wav'
LAST_TRIGGER = 'last_voice_trigger.wav'

# audio captured while the assistant is talking is its own response coming
# back -- over a repeater it returns cleanly enough to transcribe.  Waiting out
# the buffer afterwards is not enough for a long response, so mute the capture
# device for the duration and give the tail time to pass before unmuting.
MIC = '@DEFAULT_SOURCE@'
PLAYBACK_TAIL = 2

# the 440/880 pair acknowledges the trigger word so the speaker knows it was heard;
# the 440 alone leads each response, giving VOX time to key up before speech starts
ACK_TONES = 'play -n -c1 synth sin 440 fade h 0.1 .4 .1 : synth sin 880 fade h 0.1 .2 0.1'
VOX_TONE = 'play -n -c1 synth sin 440 fade h 0.1 .4 .1'

# set up wakeword detection
# rec = auditok.Recorder(input='input_double.wav', sr=16000, sw=2, ch=1)
source = None # microphone

# a crash during playback would otherwise leave the microphone muted
run(f'pactl set-source-mute {MIC} 0', shell=True)

# wall clock time until which captured audio is discarded
deaf_until = 0

# wait for detected audio
for region in auditok.split(source, sw=2, ch=1, sr=16000, min_dur=1, max_silence=2, max_dur=100, eth=55):
    # auditok buffers whatever arrived while this loop was busy speaking
    if time.time() < deaf_until:
        continue

    # auditok only chunks on energy, so silence and repeater tones reach here;
    # silero decides what is speech, and nothing else is written to disk
    if not has_speech(region):
        continue

    stamp = time.strftime('%Y%m%d-%H%M%S')
    activity = str(RECORDINGS / f'activity_{stamp}.wav')
    region.save(activity)
    shutil.copy(activity, LAST_VOICE)
    # vad_filter would re-run silero over the whole region and blank exactly the
    # transient-preceded speech has_speech() recovers, so leave it off
    segments, _ = stt.transcribe(activity, language='en', beam_size=1, vad_filter=False)
    transcribed = ' '.join(segment.text for segment in segments).strip()

    # attempt to filter out noise recognized erroneously as short phrases
    if len(transcribed.split(' ')) < 3:
        continue

    mirror(f'< {transcribed}')

    # whisper capitalizes sentence-initial words, so match the trigger case insensitively
    spoken = transcribed.lower()
    trigger = next((word for word in TRIGGER_WORDS if word in spoken), None)
    if trigger is None:
        continue

    triggered = str(RECORDINGS / f'trigger_{stamp}.wav')
    region.save(triggered)
    shutil.copy(triggered, LAST_TRIGGER)

    # acknowledge the trigger word
    run(ACK_TONES, shell=True)

    # strip out the trigger word and everything before it
    prompt = transcribed[spoken.index(trigger) + len(trigger):].lstrip(' ,.')

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

    run(f'pactl set-source-mute {MIC} 1', shell=True)
    run(VOX_TONE, shell=True)
    run('aplay -q response.wav', shell=True)
    time.sleep(PLAYBACK_TAIL)
    run(f'pactl set-source-mute {MIC} 0', shell=True)

    # whatever auditok buffered before the mute took effect
    deaf_until = time.time() + PLAYBACK_TAIL
