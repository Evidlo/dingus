# Labelled audio samples

Ground truth for tuning the VAD, the STT model and the wakeword list.  These
are copies, kept in git so they survive `recordings/` being cleared.

One wav, one txt of the same name:

    activity_20260827-220738.wav
    activity_20260827-220738.txt

The txt holds what is actually audible, lowercase and unpunctuated so it can
be compared against a transcript directly.  Audio that is not speech, or that
cannot be made out, is described in brackets instead:

    radio radio radio
    [repeater morse]
    [people making animal noises]
    kd9s [something] listening [hard to hear]

Brackets never appear in real speech, so a label containing one is a sample no
transcript should be expected to match -- useful as a negative for the VAD.
