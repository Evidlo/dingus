#!/bin/bash
# Inject a wav file into dingus' microphone stream for testing.
#
#   ./inject.sh test.wav
#
# Works by parking dingus' capture stream on a null sink's monitor, playing the
# file into that sink, then handing the stream back to the real microphone.

WAV=${1:-test.wav}
SINK=inject

# auditok closes a region after max_silence seconds of quiet
TRAILING_SILENCE=3

if [ ! -f "$WAV" ]; then
    echo "no such file: $WAV" >&2
    exit 1
fi

# a null sink's monitor stands in for the microphone
pactl list short sinks | grep -qP "^\d+\s+$SINK\s" ||
    pactl load-module module-null-sink sink_name=$SINK \
        sink_properties=device.description=$SINK >/dev/null

# find dingus' capture stream and remember which source it came from
read -r STREAM ORIGINAL < <(pactl list source-outputs | awk '
    /^Source Output #/            {stream = substr($3, 2)}
    /^[[:space:]]*Source: /       {source = $2}
    /application\.name = .*python/ {print stream, source; exit}')

if [ -z "$STREAM" ]; then
    echo "dingus is not capturing audio -- is the service running?" >&2
    exit 1
fi

echo "injecting $WAV into stream $STREAM (returns to source $ORIGINAL)"
pactl move-source-output "$STREAM" "$SINK.monitor"
paplay -d "$SINK" "$WAV"
sleep $TRAILING_SILENCE
pactl move-source-output "$STREAM" "$ORIGINAL"
echo "done"
