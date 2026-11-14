"""Voiceprints: tells speakers apart by the sound of their voice, consistently for a whole session.

Each finished sentence from Deepgram gets its slice of the band's audio run through
SpeechBrain's ECAPA speaker model (CPU). The resulting voiceprint is compared with the
voices heard so far: close enough to one of them -> same person, otherwise a new speaker.
Sentences spoken with the wrist raised and clearly loud teach Cast which voice is the wearer.
"""
import threading

import numpy as np

SAME_VOICE = 0.42        # cosine similarity above this: same person (ECAPA: same ~0.5-0.8, different ~0-0.35)
MIN_SECONDS = 0.7        # shorter slices give unreliable voiceprints
SAMPLE_RATE = 16000


class Voices:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="data/models/ecapa", run_opts={"device": "cpu"})
        self.lock = threading.Lock()
        self.speakers = []        # [{"id": int, "centroid": np.ndarray, "n": int}]
        self.wearer_id = None
        self.wearer_votes = {}

    def embed(self, pcm16):
        import torch
        wav = torch.from_numpy(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
        with torch.no_grad():
            e = self.model.encode_batch(wav).squeeze().numpy()
        return e / (np.linalg.norm(e) or 1.0)

    def identify(self, pcm16):
        """Returns (speaker id, similarity) or (None, 0) if the slice is too short."""
        if len(pcm16) < MIN_SECONDS * SAMPLE_RATE * 2:
            return None, 0.0
        e = self.embed(pcm16)
        with self.lock:
            best, best_sim = None, -1.0
            for s in self.speakers:
                sim = float(np.dot(e, s["centroid"]))
                if sim > best_sim:
                    best, best_sim = s, sim
            if best is not None and best_sim >= SAME_VOICE:
                c = best["centroid"] * best["n"] + e
                best["centroid"] = c / (np.linalg.norm(c) or 1.0)
                best["n"] = min(best["n"] + 1, 20)      # keep adapting to recent speech
                return best["id"], best_sim
            new = {"id": len(self.speakers), "centroid": e, "n": 1}
            self.speakers.append(new)
            return new["id"], best_sim

    def vote_wearer(self, speaker_id, yes):
        if speaker_id is None:
            return
        with self.lock:
            self.wearer_votes[speaker_id] = self.wearer_votes.get(speaker_id, 0) + (1 if yes else -1)
            best = max(self.wearer_votes.items(), key=lambda kv: kv[1])
            self.wearer_id = best[0] if best[1] > 0 else None


class AudioBuffer:
    """Rolling copy of the band's decoded audio, indexed by seconds since the stream began
    (the same clock as Deepgram's word timestamps)."""

    def __init__(self, seconds=180):
        self.max_bytes = seconds * SAMPLE_RATE * 2
        self.buf = bytearray()
        self.offset = 0               # bytes dropped from the front so far
        self.lock = threading.Lock()

    def add(self, pcm16):
        with self.lock:
            self.buf += pcm16
            extra = len(self.buf) - self.max_bytes
            if extra > 0:
                del self.buf[:extra]
                self.offset += extra

    def slice(self, start_s, end_s):
        with self.lock:
            a = int(start_s * SAMPLE_RATE) * 2 - self.offset
            b = int(end_s * SAMPLE_RATE) * 2 - self.offset
            a, b = max(0, a), min(len(self.buf), b)
            return bytes(self.buf[a:b]) if b > a else b""
