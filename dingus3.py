#!/usr/bin/env python3

import auditok
import numpy as np
import ollama
import requests
import shutil
import time
import urllib.parse
import wave
from auditok.util import DataValidator
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from pathlib import Path
from piper import PiperVoice, download_voices
from silero_vad import load_silero_vad, get_speech_timestamps
from subprocess import run


# --- background workers ---

def worker(fn):
    """Decorate fn to run on its own serial background thread; calls return at once."""
    pool = ThreadPoolExecutor(max_workers=1)
    return lambda *a, **k: pool.submit(fn, *a, **k)


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

# a silero validator, not raw energy, decides which windows auditok keeps as speech
vad = load_silero_vad()

class SileroValidator(DataValidator):
    # reset per window so a transient can't poison silero's recurrent state
    def is_valid(self, data):
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
        vad.reset_states()
        return bool(get_speech_timestamps(
            samples, vad, sampling_rate=16000, threshold=0.1
        ))


# --- transcript mirroring ---

# received audio is prefixed '<', spoken responses '>'
LOGFILE = Path('/srv/www/recognized.txt')
MATRIX_ROOM = '!PTZyXwJHptPcxTojOK:matrix.org'
MATRIX_API = 'https://matrix.org/_matrix/client/v3'
MATRIX_TOKEN = Path('~/.local/matrix_token').expanduser()

token = MATRIX_TOKEN.read_text().strip() if MATRIX_TOKEN.exists() else ''
room = urllib.parse.quote(MATRIX_ROOM, safe='')


@worker
def post_to_matrix(line):
    """Send one line to Matrix off-thread so a slow homeserver never stalls recognition."""
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


def mirror(line):
    """Send one transcript line to the console, the log file and the Matrix room."""
    print(line, flush=True)
    with LOGFILE.open('a') as log:
        log.write(line + '\n')

    # posting stays disabled until an access token is installed
    if token:
        post_to_matrix(line)


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

# the assistant's own speech returns over the repeater; mute capture during playback, discard the tail after
MIC = '@DEFAULT_SOURCE@'
PLAYBACK_TAIL = 2

# the 440/880 pair acknowledges the trigger word so the speaker knows it was heard;
# the 440 alone leads each response, giving VOX time to key up before speech starts
ACK_TONES = 'play -n -c1 synth sin 440 fade h 0.1 .4 .1 : synth sin 880 fade h 0.1 .2 0.1'
VOX_TONE = 'play -n -c1 synth sin 440 fade h 0.1 .4 .1'

# audio source
# source = auditok.Recorder(input='input_double.wav', sr=16000, sw=2, ch=1)
source = None # microphone

# a crash during playback would otherwise leave the microphone muted
run(f'pactl set-source-mute {MIC} 0', shell=True)

# capture skips, and respond drops, anything from before this wall-clock time
mute_until = 0


@worker
def transcribe(region, captured_at):
    """Transcribe one region and, when it carries the trigger word, answer it."""
    global mute_until
    # drop what was buffered during the last exchange (the assistant's own echo)
    if captured_at < mute_until:
        return

    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(captured_at))
    # leave vad_filter off: it would blank the transient-preceded speech the validator recovers
    segments, _ = stt.transcribe(
        (region.samples[0] / 32768).astype(np.float32),
        language='en', beam_size=1, vad_filter=False
    )
    transcribed = ' '.join(segment.text for segment in segments).strip()

    # drop noise misrecognized as a short phrase
    if len(transcribed.split(' ')) < 3:
        return

    mirror(f'< {transcribed}')

    for trigger in TRIGGER_WORDS:
        # trigger word detected
        if trigger in transcribed.lower():
            # save triggered audio
            region.save(path:=str(RECORDINGS / f'trigger_{stamp}.wav'))
            shutil.copy(path, LAST_TRIGGER)
            # strip the trigger word and everything before it
            respond(transcribed[transcribed.index(trigger) + len(trigger):].lstrip(' ,.'))
            break

    # save audio
    region.save(path:=str(RECORDINGS / f'activity_{stamp}.wav'))
    shutil.copy(path, LAST_VOICE)


@worker
def respond(transcribed):
    global mute_until

    run(ACK_TONES, shell=True)  # acknowledge the trigger word


    response = ollama.generate(
        model=MODEL,
        system=SYSTEM,
        prompt=transcribed,
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

    # mute input while responding
    mute_until = time.time() + PLAYBACK_TAIL


# detect voice activity blocks
regions = auditok.split(
    source, sw=2, ch=1, sr=16000, aw=0.5,
    min_dur=1, max_silence=2, max_dur=100,
    validator=SileroValidator()
)
for region in regions:
    if time.time() < mute_until:
        continue
    transcribe(region, time.time())
