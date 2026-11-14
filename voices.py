"""Voiceprints: tells speakers apart by the sound of their voice, consistently for a whole session.

Each finished sentence from Deepgram gets its slice of the band's audio run through
SpeechBrain's ECAPA speaker model (CPU). The resulting voiceprint is compared with the
voices heard so far: close enough to one of them -> same person, otherwise a new speaker.
Sentences spoken with the wrist raised and clearly loud teach Cast which voice is the wearer.
"""
import json
import threading

import numpy as np

SAME_VOICE = 0.42        # cosine similarity above this: same person (measured: same voice 0.8-0.9 on long speech)
NEW_SPEAKER_SECONDS = 2.0  # need at least this much audio to start a new speaker
MERGE_AT = 0.6           # two clusters this similar are the same person
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
        self.aliases = {}         # merged speaker id -> surviving id
        self.names = {}           # speaker id -> name ("Hardik"), learned from conversation
        self.next_id = 0

    def embed(self, pcm16):
        import torch
        wav = torch.from_numpy(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
        with torch.no_grad():
            e = self.model.encode_batch(wav).squeeze().numpy()
        return e / (np.linalg.norm(e) or 1.0)

    def identify(self, pcm16):
        """Returns (speaker id, similarity) or (None, 0) if the slice is too short.
        Short audio gives unreliable voiceprints, so it is only ever matched to a known voice;
        a new speaker needs NEW_SPEAKER_SECONDS of audio that clearly matches nobody."""
        seconds = len(pcm16) / (SAMPLE_RATE * 2)
        if seconds < MIN_SECONDS:
            return None, 0.0
        e = self.embed(pcm16)
        with self.lock:
            best, best_sim = None, -1.0
            for s in self.speakers:
                sim = float(np.dot(e, s["centroid"]))
                if sim > best_sim:
                    best, best_sim = s, sim
            confident = seconds >= NEW_SPEAKER_SECONDS
            if best is not None and (best_sim >= SAME_VOICE or not confident):
                if confident and best_sim >= SAME_VOICE:      # only learn from good evidence
                    c = best["centroid"] * best["n"] + e
                    best["centroid"] = c / (np.linalg.norm(c) or 1.0)
                    best["n"] = min(best["n"] + 1, 20)
                    self._merge_close(best)
                return best["id"], best_sim
            self.next_id += 1
            new = {"id": self.next_id - 1, "centroid": e, "n": 1}
            self.speakers.append(new)
            return new["id"], best_sim

    def _merge_close(self, keep):
        """Two clusters that turn out to be one voice: fold the newer into the older."""
        for other in list(self.speakers):
            if other is keep:
                continue
            if float(np.dot(keep["centroid"], other["centroid"])) >= MERGE_AT:
                a, b = (keep, other) if keep["id"] < other["id"] else (other, keep)
                c = a["centroid"] * a["n"] + b["centroid"] * b["n"]
                a["centroid"] = c / (np.linalg.norm(c) or 1.0)
                a["n"] = min(a["n"] + b["n"], 20)
                self.speakers.remove(b)
                self.aliases[b["id"]] = a["id"]
                if b["id"] in self.names and a["id"] not in self.names:
                    self.names[a["id"]] = self.names.pop(b["id"])
                if self.wearer_id == b["id"]:
                    self.wearer_id = a["id"]
                return

    def resolve(self, speaker_id):
        while speaker_id in self.aliases:
            speaker_id = self.aliases[speaker_id]
        return speaker_id

    # ---- remembered between runs, so a restart does not forget who is who

    def save(self, path):
        with self.lock:
            if not self.speakers:
                return
            np.savez(path, ids=[s["id"] for s in self.speakers], n=[s["n"] for s in self.speakers],
                     centroids=np.stack([s["centroid"] for s in self.speakers]),
                     meta=json.dumps({"wearer_id": self.wearer_id, "next_id": self.next_id,
                                      "votes": {str(k): v for k, v in self.wearer_votes.items()},
                                      "names": {str(k): v for k, v in self.names.items()}}))

    def load(self, path):
        try:
            d = np.load(path)
            meta = json.loads(str(d["meta"]))
        except (OSError, ValueError, KeyError):
            return False
        with self.lock:
            self.speakers = [{"id": int(i), "centroid": c, "n": int(n)} for i, c, n in zip(d["ids"], d["centroids"], d["n"])]
            self.wearer_id, self.next_id = meta["wearer_id"], meta["next_id"]
            self.wearer_votes = {int(k): v for k, v in meta["votes"].items()}
            self.names = {int(k): v for k, v in meta["names"].items()}
        return True

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
