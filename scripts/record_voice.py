"""Record the spoken request the public app replays: one question, spoken, transcribed, answered.

    python scripts/record_voice.py

1. data/voice/question.wav holds the spoken question. The committed file was voiced with Windows' built-in
   speech synthesiser (System.Speech, not a model): Muse Voice Transcribe heard it word for word. If the file
   is missing, it is voiced with Meta's open MMS-TTS (facebook/mms-tts-eng) instead.
2. Muse Voice Transcribe turns the WAV into text (one live call, about $0.0003).
3. Muse Spark answers the transcript through the checked tools, on a fresh site at 6 pm on 21 June 2017.

Both responses land in data/recordings/, so the replay-only app runs the same path with no key.
The spend guard in headroom/meta/client.py applies.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headroom.agent.operator import SYSTEM, VOICE_REQUEST_AT, run  # noqa: E402
from headroom.agent.site import ReplaySite  # noqa: E402
from headroom.config import DATA  # noqa: E402
from headroom.meta.client import ModelClient, spent_usd, transcribe  # noqa: E402

WAV = DATA / "voice" / "question.wav"
SPOKEN = "Can we charge six fleet vans tonight? They plug in at seven p m, and each needs forty kilowatt hours by six a m."


def synthesize(path: Path) -> None:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, VitsModel

    model, tok = VitsModel.from_pretrained("facebook/mms-tts-eng"), AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    torch.manual_seed(7)  # the VITS sampler is random: fix it so the file is reproducible
    with torch.no_grad():
        audio = model(**tok(SPOKEN, return_tensors="pt")).waveform[0].numpy()
    pcm = (np.clip(audio / max(np.abs(audio).max(), 1e-6) * 0.9, -1, 1) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(model.config.sampling_rate)
        w.writeframes(pcm.tobytes())


def main() -> None:
    print(f"spent so far=${spent_usd():.4f}")
    if not WAV.exists():
        synthesize(WAV)
    text = transcribe(WAV.read_bytes(), mode="auto")
    print("transcript:", text)
    site = ReplaySite(month="2017-06", n_ports=24)
    site.now_local = VOICE_REQUEST_AT
    turn = run(site, text, history=[{"role": "system", "content": SYSTEM}], client=ModelClient("spark", mode="auto"))
    print("tools:", [s.tool for s in turn.steps if s.kind == "tool"])
    print("answer:", turn.answer)
    print("unsupported:", turn.unsupported)
    print(f"spent now=${spent_usd():.4f}")


if __name__ == "__main__":
    main()
