"""Voiceprints: tells speakers apart by the sound of their voice, consistently for a whole session.

Each finished sentence from Deepgram gets its slice of the band's audio run through
SpeechBrain's ECAPA speaker model (CPU). The resulting voiceprint is compared with the
voices heard so far: close enough to one of them -> same person, otherwise a new speaker.
Sentences spoken with the wrist raised and clearly loud teach Cast which voice is the wearer.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np

from ..paths import DATA

SAME_VOICE = 0.42        # cosine similarity above this: same person (measured: same voice 0.8-0.9 on long speech)
NEW_SPEAKER_SECONDS = 1.5  # need at least this much audio to start a new speaker
NEW_PERSON_BELOW = 0.55    # 5 s of speech less similar than this to every known voice: someone new
UNKNOWN = -1               # heard, but too short to say who
MERGE_AT = 0.6           # two clusters this similar are the same person
MIN_SECONDS = 0.7        # shorter slices give unreliable voiceprints
SAMPLE_RATE = 16000


class Voices:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir=str(DATA / "models" / "ecapa"), run_opts={"device": "cpu"})
        self.lock = threading.Lock()
        self.speakers = []        # [{"id": int, "centroid": np.ndarray, "n": int}]
        self.wearer_id = None
        self.wearer_votes = {}
        self.aliases = {}         # merged speaker id -> surviving id
        self.enrolled = {}        # speaker id -> voiceprint from a 40 s enrolment recording
        self.mu = None            # mean of the enrolled voiceprints (subtracted before comparing)
        self.names = {}           # speaker id -> name ("Hardik"), learned from conversation
        self.next_id = 0

    def embed(self, pcm16):
        import torch
        wav = torch.from_numpy(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0).unsqueeze(0)
        with torch.no_grad():
            e = self.model.encode_batch(wav).squeeze().numpy()
        return e / (np.linalg.norm(e) or 1.0)

    # ---- enrolled voices (measured on this band: right 95-100% of the time when it commits)

    EXTRA = ("nemo_en_titanet_large", "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common")
    EXTRA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/%s.onnx"

    def _load_extra(self):
        """Two more voiceprint models (NVIDIA TitaNet-Large, ERes2NetV2). On this band's recordings
        the three together, with a clear-lead rule, were right 96-98% of the time they committed;
        ECAPA alone was 92-95% and committed less often."""
        import sherpa_onnx
        import urllib.request
        folder = DATA / "models" / "spk"
        folder.mkdir(parents=True, exist_ok=True)
        self.extra = []
        for name in self.EXTRA:
            f = folder / (name + ".onnx")
            if not f.exists():
                print("Downloading voice model %s..." % name)
                urllib.request.urlretrieve(self.EXTRA_URL % name, f)
            ex = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(f), num_threads=2))
            self.extra.append(ex)

    def embed_all(self, pcm16):
        """One voiceprint per model: [ECAPA, TitaNet, ERes2NetV2]."""
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        out = [self.embed(pcm16)]
        for ex in self.extra:
            st = ex.create_stream()
            st.accept_waveform(SAMPLE_RATE, x)
            st.input_finished()
            out.append(np.array(ex.compute(st)))
        return out

    def enroll_folder(self, folder):
        """Each <label>_<time>.pcm is ~40 s of one person. 'wearer' is the person wearing the band.
        Every recording becomes its own example of that voice (a voice sounds different to the band
        with the wrist at the mouth, resting, or at the side)."""
        folder = Path(folder)
        files = sorted(folder.glob("*.pcm"))
        labels = sorted({f.stem.rsplit("_", 1)[0] for f in files}, key=lambda l: (l != "wearer", l))
        if "wearer" not in labels or len(labels) < 2:
            return False
        self._load_extra()
        try:
            given = json.loads((folder / "names.json").read_text())
        except (OSError, ValueError):
            given = {}
        n = 3 * SAMPLE_RATE
        raw = {}
        for i, label in enumerate(labels):
            raw[i] = []
            for f in files:
                if f.stem.rsplit("_", 1)[0] != label:
                    continue
                x = np.frombuffer(f.read_bytes(), dtype=np.int16)
                ref = np.sqrt(np.mean(x.astype(float) ** 2))
                embs = [self.embed_all(x[a:a + n].tobytes()) for a in range(0, len(x) - n, n // 2)
                        if np.sqrt(np.mean(x[a:a + n].astype(float) ** 2)) > 0.35 * ref]      # skip silence
                if embs:
                    raw[i].append([np.mean([e[m] for e in embs], axis=0) for m in range(len(embs[0]))])
            if label != "wearer":
                self.names[i] = given.get(label) or "Person %d" % (i + 1)
        n_models = 1 + len(self.extra)
        self.mu = [np.mean([ex[m] for exs in raw.values() for ex in exs], axis=0) for m in range(n_models)]
        self.enrolled = {i: [[self._unit(ex[m] - self.mu[m]) for m in range(n_models)] for ex in exs]
                         for i, exs in raw.items()}
        self.plain = {i: [[self._unit(ex[m]) for m in range(n_models)] for ex in exs] for i, exs in raw.items()}
        self.wearer_id = 0
        self.labels = labels
        self.base_counts = {i: len(exs) for i, exs in self.enrolled.items()}
        return True

    # ---- what was learned while running (corrections, new people, clear long clips)

    def save_learned(self, path):
        import pickle
        if not self.enrolled:
            return
        with self.lock:
            data = {"labels": list(self.labels), "base": dict(self.base_counts),
                    "extra": {i: (ex[self.base_counts.get(i, 0):], self.plain.get(i, [])[self.base_counts.get(i, 0):])
                              for i, ex in self.enrolled.items()},
                    "names": dict(self.names)}
        tmp = str(path) + ".tmp"
        Path(tmp).write_bytes(pickle.dumps(data))
        os.replace(tmp, path)

    def load_learned(self, path):
        import pickle
        try:
            data = pickle.loads(Path(path).read_bytes())
        except Exception:  # noqa: BLE001
            return 0
        if data.get("labels") != list(self.labels) or data.get("base") != dict(self.base_counts):
            return 0                                  # the enrolment recordings changed: start clean
        n = 0
        with self.lock:
            for i, (centred, plain) in data["extra"].items():
                self.enrolled.setdefault(i, []).extend(centred)
                self.plain.setdefault(i, []).extend(plain)
                n += len(centred)
            for i, name in data["names"].items():
                if i in self.enrolled:
                    self.names[i] = name
        return n

    def teach(self, pcm16, vid=None, name=None):
        """The user said who a stretch of speech was. Keep it as an example of that voice
        (vid), or as a new person (name). Returns the voice id, or None if there is nothing
        to learn from (no enrolment, or under a second of audio)."""
        if not self.enrolled or len(pcm16) / (SAMPLE_RATE * 2) < 1.0:
            return vid
        embs = self.embed_all(pcm16)
        centred = [self._unit(v - mu) for v, mu in zip(embs, self.mu)]
        plain = [self._unit(v) for v in embs]
        with self.lock:
            if vid is None:
                vid = max(self.enrolled) + 1
                self.enrolled[vid], self.plain[vid] = [], []
                self.names[vid] = name or "Person %d" % (vid + 1)
            self.enrolled[vid].append(centred)
            self.plain.setdefault(vid, []).append(plain)
        return vid

    def discover(self, pcm16):
        """A long stretch (5 s+) from a voice that matches nobody enrolled becomes a new person.
        Measured on this band with 5 s clips: a known voice scores 0.61+ against itself (lowest
        10%), a never-heard voice 0.56 or less against anyone (highest 10%)."""
        if not self.enrolled or len(pcm16) / (SAMPLE_RATE * 2) < 5.0:
            return None
        embs = self.embed_all(pcm16)
        raw = [self._unit(v) for v in embs]
        best = max(float(np.mean([max(float(raw[m] @ ex[m]) for ex in exs) for m in range(len(raw))]))
                   for exs in self.plain.values())
        if best >= NEW_PERSON_BELOW:
            return None
        with self.lock:
            nid = max(self.enrolled) + 1
            self.plain[nid] = [raw]
            self.enrolled[nid] = [[self._unit(v - mu) for v, mu in zip(embs, self.mu)]]
            self.names[nid] = "Person %d" % (nid + 1)
        return nid

    @staticmethod
    def _unit(e):
        return e / (np.linalg.norm(e) or 1.0)

    def identify_enrolled(self, pcm16):
        seconds = len(pcm16) / (SAMPLE_RATE * 2)
        if seconds < 1.0:
            return None, 0.0
        e = [self._unit(v - mu) for v, mu in zip(self.embed_all(pcm16), self.mu)]
        # per person: for each model, the best match among that person's examples; then average the models
        with self.lock:                                # a correction on the screen can add a person meanwhile
            people = [(i, list(exs)) for i, exs in self.enrolled.items()]
        scores = sorted(((float(np.mean([max(float(e[m] @ ex[m]) for ex in exs) for m in range(len(e))])), i)
                         for i, exs in people), reverse=True)
        margin = scores[0][0] - scores[1][0]
        need = 0.20 if seconds >= 1.5 else 0.26       # commit only on a clear lead
        if margin >= 0.30 and seconds >= 2.5 and len(self.enrolled[scores[0][1]]) < 14:
            self.enrolled[scores[0][1]].append(e)      # a clear, long clip: one more example of this voice
            self.plain.setdefault(scores[0][1], []).append([self._unit(v) for v in self.embed_all(pcm16)])
        return (scores[0][1] if margin >= need else UNKNOWN), margin

    def identify(self, pcm16):
        """Returns (speaker id, similarity) or (None, 0) if the slice is too short.
        Short audio gives unreliable voiceprints, so it is only ever matched to a known voice;
        a new speaker needs NEW_SPEAKER_SECONDS of audio that clearly matches nobody."""
        if self.enrolled:
            return self.identify_enrolled(pcm16)
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
            if best is not None and not confident and best_sim < SAME_VOICE:
                # too short to start a new speaker, and it does not sound like anyone known:
                # say "unknown" rather than pin it on the closest voice
                return UNKNOWN, best_sim
            if best is not None and best_sim >= SAME_VOICE:
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
                self.wearer_votes[a["id"]] = self.wearer_votes.get(a["id"], 0) + self.wearer_votes.pop(b["id"], 0)
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
            tmp = str(path) + ".tmp.npz"
            np.savez(tmp, ids=[s["id"] for s in self.speakers], n=[s["n"] for s in self.speakers],
                     centroids=np.stack([s["centroid"] for s in self.speakers]),
                     meta=json.dumps({"wearer_id": self.wearer_id, "next_id": self.next_id,
                                      "votes": {str(k): v for k, v in self.wearer_votes.items()},
                                      "names": {str(k): v for k, v in self.names.items()}}))
            os.replace(tmp, path)

    def load(self, path):
        try:
            d = np.load(path)
            meta = json.loads(str(d["meta"]))
        except Exception:  # noqa: BLE001 - a damaged file just means starting fresh
            return False
        with self.lock:
            self.speakers = [{"id": int(i), "centroid": c, "n": int(n)} for i, c, n in zip(d["ids"], d["centroids"], d["n"])]
            self.wearer_id, self.next_id = meta["wearer_id"], meta["next_id"]
            self.wearer_votes = {int(k): v for k, v in meta["votes"].items()}
            self.names = {int(k): v for k, v in meta["names"].items()}
        return True

    def vote_wearer(self, speaker_id, yes):
        if speaker_id is None or speaker_id < 0 or self.enrolled:   # enrolled: the wearer is known
            return
        with self.lock:
            self.wearer_votes[speaker_id] = self.wearer_votes.get(speaker_id, 0) + (1 if yes else -1)
            best = max(self.wearer_votes.items(), key=lambda kv: kv[1])
            current = self.wearer_votes.get(self.wearer_id, 0) if self.wearer_id is not None else None
            if current is None:
                self.wearer_id = best[0] if best[1] > 0 else None
            elif best[0] != self.wearer_id and best[1] > current and best[1] > 0:
                self.wearer_id = best[0]              # someone else clearly has more evidence
            elif current <= -2:
                self.wearer_id = None                 # one stray "no" never clears the wearer


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
