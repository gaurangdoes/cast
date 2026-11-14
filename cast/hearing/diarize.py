"""Nemotron 3 Diarization on CPU via onnxruntime (no torch, no NeMo).

Uses onnx-community/Nemotron-3-Diarization-ONNX (fp32 graph). The ONNX graph takes
128-bin log-mel features; the feature extractor below is a numpy port of
transformers' NemotronAsrStreamingFeatureExtractor (preemphasis 0.97, n_fft 512,
win 400 symmetric hann, hop 160, slaney mel, log(x + 2**-24), no normalization).

Offline (window) mode: one ONNX call per window of up to ~27 s (340 encoder frames),
with an empty speaker cache. Speaker ids are in order of first arrival *within the
window*, so they are not stable across separate calls.

Needs: onnxruntime numpy huggingface_hub (and transformers only for mel_filter_bank;
replaced here by a local implementation so transformers is NOT required).
"""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import onnxruntime as ort

from ..paths import DATA

SR = 16000
N_FFT, WIN, HOP, N_MELS, PREEMPH = 512, 400, 160, 128, 0.97
REPO = "onnx-community/Nemotron-3-Diarization-ONNX"
MODEL_FILE = os.environ.get("NEMOTRON_ONNX_FILE", "onnx/model.onnx")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.environ.get("NEMOTRON_DIR", str(DATA / "models" / "nemotron"))

_session = None
_mel = None


def _hz_to_mel_slaney(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz_slaney(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def _mel_filters():
    # Same as transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")
    fft_freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    mel_pts = np.linspace(_hz_to_mel_slaney(0.0), _hz_to_mel_slaney(SR / 2), N_MELS + 2)
    filt_freqs = _mel_to_hz_slaney(mel_pts)
    fdiff = np.diff(filt_freqs)
    slopes = filt_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = np.maximum(0, np.minimum(down, up))  # (n_freq, n_mels)
    enorm = 2.0 / (filt_freqs[2 : N_MELS + 2] - filt_freqs[:N_MELS])
    return (fb * enorm[None, :]).astype(np.float32)


def _features(x: np.ndarray) -> np.ndarray:
    """x: float32 mono 16 kHz in [-1, 1]. Returns (frames, 128) log-mel (center=True)."""
    global _mel
    if _mel is None:
        _mel = _mel_filters()
    x = np.concatenate([x[:1], x[1:] - PREEMPH * x[:-1]]).astype(np.float32)
    n_valid = len(x) // HOP
    x = np.pad(x, (N_FFT // 2, N_FFT // 2))
    n_frames = 1 + (len(x) - N_FFT) // HOP
    win = np.hanning(WIN).astype(np.float32)  # symmetric == torch periodic=False
    lpad = (N_FFT - WIN) // 2
    window = np.zeros(N_FFT, np.float32)
    window[lpad : lpad + WIN] = win
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2
    mel = np.log(spec.astype(np.float32) @ _mel + 2.0**-24)
    return mel[:n_valid].astype(np.float32)


def _get_session():
    global _session
    if _session is None:
        path = os.path.join(LOCAL_DIR, MODEL_FILE)
        if not os.path.exists(path):
            from huggingface_hub import snapshot_download

            snapshot_download(REPO, local_dir=LOCAL_DIR, allow_patterns=[MODEL_FILE, MODEL_FILE + "_data", "*.json"])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _session


def speaker_probs(x: np.ndarray) -> np.ndarray:
    """Returns (frames@10ms, 8) speaker activity probabilities for float32 audio."""
    feats = _features(x)[None]
    sess = _get_session()
    out = sess.run(
        ["logits"],
        {
            "input_features": feats,
            "cached_embeds": np.zeros((1, 0, 512), np.float32),
            "attention_mask": np.ones((1, -(-feats.shape[1] // 8)), np.int64),  # encoder-frame mask (cache + chunk)
        },
    )[0][0]
    logits = out[: feats.shape[1]]
    return 1.0 / (1.0 + np.exp(-logits))


def diarize(pcm_int16: bytes, sample_rate: int = 16000, threshold: float = 0.5, min_dur: float = 0.2):
    """Returns [(speaker:int, start:float, end:float), ...] sorted by start time."""
    if sample_rate != SR:
        raise ValueError("resample to 16 kHz first")
    x = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
    probs = speaker_probs(x)
    active = (probs > threshold).astype(np.int8)
    pad = np.zeros((1, active.shape[1]), np.int8)
    d = np.diff(np.concatenate([pad, active, pad]), axis=0)
    segs = []
    for spk in range(active.shape[1]):
        starts = np.nonzero(d[:, spk] == 1)[0]
        ends = np.nonzero(d[:, spk] == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) * 0.01 >= min_dur:
                segs.append((int(spk), round(float(s) * 0.01, 2), round(float(e) * 0.01, 2)))
    segs.sort(key=lambda t: (t[1], t[0]))
    return segs


# ---------------------------------------------------------------------------------------------
# Streaming mode: audio in small chunks, speaker cache carried across chunks.
#
# This is a numpy port of the streaming forward of transformers'
# Nemotron3DiarizationForAudioFrameClassification and its Nemotron3DiarizationSpeakerCache
# (NeMo's Sortformer "arrival-order speaker cache" + FIFO). The ONNX graph is stateless: it
# takes the mel features of one chunk plus the encoder-frame embeddings to attend to
# (`cached_embeds` = speaker cache followed by the FIFO queue), and returns logits for
# (cached + chunk) frames plus the chunk's own embeddings; all the cache bookkeeping is here.
# Because every chunk attends to the cache, output column k is the same person for the whole
# session (speakers are numbered in order of first arrival).
# ---------------------------------------------------------------------------------------------
SUB = 8                      # mel frames (10 ms) per encoder frame (80 ms)
FRAME_S = HOP / SR           # 0.01 s
# name -> (chunk_len, right_context) in encoder frames; latency = (chunk_len + right_context) * 80 ms
STREAMING_MODES = {
    "1.04s": (9, 4), "low_latency": (9, 4),
    "0.64s": (6, 2), "very_low_latency": (6, 2),
    "0.32s": (3, 1), "ultra_low_latency": (3, 1),
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class _SpeakerCache:
    """Arrival-order speaker cache + FIFO queue of recent encoder frames (batch size 1).

    Defaults are `streaming_config` of the checkpoint's config.json: the FIFO holds the most
    recent 264 encoder frames (21 s); when it overflows, its oldest 222 frames move to the
    speaker cache, which is compressed back to 264 frames by keeping, per speaker, the frames
    where that speaker is most clearly the only one talking (plus a silence frame each).
    """

    def __init__(self, fifo_len=264, update_period=222, cache_len=264, n_spk=8, sil_per_spk=1,
                 score_threshold=0.25, latest_boost=0.05, strong_rate=0.75, weak_rate=1.5, min_pos_rate=0.5):
        self.fifo_len, self.update_period, self.cache_len = fifo_len, update_period, cache_len
        self.n_spk, self.n_sil = n_spk, sil_per_spk
        self.thr, self.latest_boost = score_threshold, latest_boost
        budget = cache_len // n_spk - sil_per_spk
        self.min_pos = math.floor(budget * min_pos_rate)
        self.n_strong = math.floor(budget * strong_rate)
        self.n_weak = math.floor(budget * weak_rate)
        self.embeds = np.zeros((0, 512), np.float32)      # speaker cache
        self.probs = np.zeros((0, n_spk), np.float32)     # speaker probabilities stored with it
        self.fifo = np.zeros((0, 512), np.float32)
        self.compressed = False

    def get(self) -> np.ndarray:
        return np.concatenate([self.embeds, self.fifo], axis=0)

    def update(self, step_embeds: np.ndarray, step_logits: np.ndarray, silence: np.ndarray, n_chunk: int):
        """step_embeds: (cached + chunk + lookahead, 512) encoder input of the step; step_logits:
        (that * 8, n_spk). Pushes the n_chunk chunk frames to the FIFO (the look-ahead frames are
        fed again next step) and moves the oldest FIFO frames to the speaker cache on overflow."""
        n_cache, n_fifo = len(self.embeds), len(self.fifo)
        n_in = len(step_embeds)
        probs = _sigmoid(step_logits[: n_in * SUB].astype(np.float32)).reshape(n_in, SUB, self.n_spk).mean(axis=1)
        start = n_cache + n_fifo
        fifo = np.concatenate([self.fifo, step_embeds[start : start + n_chunk]], axis=0)
        n_pop = 0
        if len(fifo) > self.fifo_len:
            n_pop = min(max(self.update_period, len(fifo) - self.fifo_len), len(fifo))
        if n_pop:
            fifo_probs = probs[n_cache : start + n_chunk]
            # an uncompressed cache is still in time order, so this step re-estimated its
            # probabilities; a compressed one keeps the probabilities stored with its frames
            stored = self.probs if self.compressed else probs[:n_cache]
            embeds = np.concatenate([self.embeds, fifo[:n_pop]], axis=0)
            cprobs = np.concatenate([stored, fifo_probs[:n_pop]], axis=0)
            fifo = fifo[n_pop:]
            if len(embeds) > self.cache_len:
                embeds, cprobs = self._compress(embeds, cprobs, silence)
                self.compressed = True
            self.embeds, self.probs = embeds, cprobs.astype(np.float32)
        self.fifo = fifo

    def _scores(self, probs):
        with np.errstate(divide="ignore"):
            lp = np.log(np.maximum(probs, self.thr))
            lc = np.log(np.maximum(1.0 - probs, self.thr))
        scores = lp - lc + lc.sum(axis=1, keepdims=True) - math.log(0.5)
        speech = probs > 0.5
        scores[~speech] = -np.inf
        positive = scores > 0
        enough = positive.sum(axis=0, keepdims=True) >= self.min_pos
        scores[~positive & speech & enough] = -np.inf
        return scores

    @staticmethod
    def _boost(scores, k, boost):
        top = np.argsort(-scores, axis=0, kind="stable")[:k]            # best k frames of each speaker
        np.add.at(scores, (top, np.arange(scores.shape[1])[None, :]), boost)
        return scores

    def _compress(self, embeds, probs, silence):
        n, n_spk = probs.shape
        scores = self._scores(probs.astype(np.float32))
        scores[self.cache_len :] += self.latest_boost                   # the frames just popped from the FIFO
        scores = self._boost(scores, self.n_strong, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.n_weak, -math.log(0.5))
        scores = np.concatenate([scores, np.full((self.n_sil, n_spk), np.inf, scores.dtype)], axis=0)
        embeds = np.concatenate([embeds, silence.reshape(1, -1).astype(embeds.dtype)], axis=0)
        probs = np.concatenate([probs, np.zeros((1, n_spk), probs.dtype)], axis=0)
        n_scored = n + self.n_sil
        sentinel = n_scored * n_spk
        flat = scores.T.reshape(-1)                                     # speaker-major
        top = np.argsort(-flat, kind="stable")[: self.cache_len]
        top = np.where(np.isneginf(flat[top]), sentinel, top)
        top.sort()
        idx = np.where(top == sentinel, n, np.minimum(top % n_scored, n))  # n = the silence frame
        return embeds[idx], probs[idx]


class StreamingDiarizer:
    """Nemotron 3 streaming diarization with a speaker cache that persists for the session.

        d = StreamingDiarizer()                  # latency "1.04s" (default), "0.64s" or "0.32s"
        d.feed(pcm_bytes)                        # 16 kHz mono int16, any amount, as audio arrives
        d.activity(t0, t1)   -> [(speaker, fraction_active), ...]  most active first
        d.segments(t0, t1)   -> [(speaker, start, end), ...]       may overlap between speakers

    Times are seconds of stream audio since construction / reset(). Speaker ids (0..7, in order
    of first arrival) keep meaning the same person until reset(). The model emits a chunk once
    its look-ahead has arrived, so the newest `latency` seconds are not diarized yet:
    `processed_time` says how far it got; pass `provisional=True` to activity()/segments() to
    also get a one-off estimate of that tail (computed without touching the cache).

    With background=True (default) feed() only queues the audio and a daemon thread runs the
    model (about 40-90 ms per 0.72 s step on this CPU), so feed() is safe to call from an audio
    callback; wait() blocks until everything fed has been processed. With background=False the
    model runs inside feed().
    """

    def __init__(self, latency: str = "1.04s", chunk_len: int | None = None, right_context: int | None = None,
                 threshold: float = 0.5, min_dur: float = 0.2, history_s: float = 300.0, background: bool = True):
        if chunk_len is None or right_context is None:
            if latency not in STREAMING_MODES:
                raise ValueError("latency must be one of %s" % sorted(STREAMING_MODES))
            chunk_len, right_context = STREAMING_MODES[latency]
        self.chunk_len, self.right_context = int(chunk_len), int(right_context)
        self.latency_s = (self.chunk_len + self.right_context) * SUB * FRAME_S
        self.threshold, self.min_dur = threshold, min_dur
        self.background = background
        self._hist_len = max(int(history_s / FRAME_S), 100)
        self._lock = threading.Lock()            # history + pending audio
        self._cond = threading.Condition(self._lock)
        self._proc_lock = threading.Lock()       # model state (features, cache); taken before _lock
        self._thread = None
        self._closed = False
        self._gen = 0
        self._window = np.zeros(N_FFT, np.float32)
        lpad = (N_FFT - WIN) // 2
        self._window[lpad : lpad + WIN] = np.hanning(WIN).astype(np.float32)
        self._melfb = _mel_filters()
        self.compute_s = 0.0                     # seconds spent in feature extraction + model
        self._init_state()

    def _init_state(self):
        self._pending = []                       # raw bytes not yet processed
        self._busy = False
        self._odd = b""
        self._samples_fed = 0
        self._last_raw = None                    # previous raw sample, for preemphasis
        self._wave = np.zeros(N_FFT // 2, np.float32)   # preemphasized audio not yet framed (center=True padding)
        self._mel = np.zeros((0, N_MELS), np.float32)   # mel frames not yet committed
        self._cache = _SpeakerCache()
        self._hist = np.zeros((self._hist_len, 8), np.float32)
        self._n_frames = 0                       # 10 ms frames committed since the start of the stream

    # ---- feeding

    def feed(self, pcm_int16: bytes) -> None:
        """Any amount of 16 kHz mono int16 audio, in stream order."""
        if not pcm_int16:
            return
        with self._cond:
            self._pending.append(bytes(pcm_int16))
            self._samples_fed += len(pcm_int16) // 2      # (an odd trailing byte is carried over)
            if self.background:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="StreamingDiarizer", daemon=True)
                    self._thread.start()
                self._cond.notify_all()
                return
        self._drain()

    def _take(self):
        """Under _cond: pops the queued audio."""
        data = b"".join(self._pending)
        self._pending = []
        self._busy = True
        return data, self._gen

    def _drain(self):
        with self._cond:
            data, gen = self._take()
        try:
            with self._proc_lock:
                if gen == self._gen:
                    self._process(data)
        finally:
            with self._cond:
                self._busy = bool(self._pending)
                self._cond.notify_all()

    def _run(self):
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
            try:
                self._drain()
            except Exception as e:  # noqa: BLE001 - keep the worker alive, but say so
                print("[StreamingDiarizer] step failed:", repr(e))

    def wait(self, timeout: float | None = None) -> bool:
        """Blocks until all audio fed so far has been processed. Returns False on timeout."""
        with self._cond:
            return self._cond.wait_for(lambda: not self._pending and not self._busy, timeout)

    def close(self) -> None:
        """Stops the background thread. The object can still be queried, and fed (the model
        then runs inside feed())."""
        self.wait(5.0)
        with self._cond:
            self._closed = True
            self.background = False
            thread, self._thread = self._thread, None
            self._cond.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(5.0)

    def reset(self) -> None:
        """Forgets everything: speakers, history, and the clock restarts at 0."""
        with self._proc_lock:
            with self._cond:
                self._gen += 1
                self._init_state()
                self._cond.notify_all()

    # ---- model

    def _process(self, data: bytes):
        """Under _proc_lock: features for the new audio, then as many model steps as are ready."""
        t0 = time.perf_counter()
        data = self._odd + data
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        else:
            self._odd = b""
        samples = np.frombuffer(data, dtype=np.int16)
        need = (self.chunk_len + self.right_context) * SUB
        step = self.chunk_len * SUB
        for i in range(0, len(samples), 30 * SR):         # at most 30 s at a time
            x = samples[i : i + 30 * SR].astype(np.float32) / 32768.0
            pre = np.empty_like(x)
            pre[0] = x[0] if self._last_raw is None else x[0] - PREEMPH * self._last_raw
            pre[1:] = x[1:] - PREEMPH * x[:-1]
            self._last_raw = float(x[-1])
            self._wave = np.concatenate([self._wave, pre])
            # frame i is centred on sample i * HOP, exactly as one centre=True pass over the whole stream
            if len(self._wave) >= N_FFT:
                n = 1 + (len(self._wave) - N_FFT) // HOP
                idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
                spec = np.abs(np.fft.rfft(self._wave[idx] * self._window, axis=1)) ** 2
                mel = np.log(spec.astype(np.float32) @ self._melfb + 2.0**-24).astype(np.float32)
                self._mel = np.concatenate([self._mel, mel], axis=0)
                self._wave = self._wave[n * HOP :]
            while len(self._mel) >= need:
                self._step(self._mel[:need])
                self._mel = self._mel[step:]
        self.compute_s += time.perf_counter() - t0

    def _infer(self, mel: np.ndarray):
        cached = self._cache.get()
        n_enc = -(-len(mel) // SUB)
        logits, embeds, silence = _get_session().run(
            ["logits", "chunk_embeds", "silence_embeds"],
            {
                "input_features": mel[None],
                "cached_embeds": cached[None],
                "attention_mask": np.ones((1, len(cached) + n_enc), np.int64),
            },
        )
        return cached, logits[0], embeds[0], silence

    def _step(self, mel: np.ndarray):
        """One streaming step: mel = chunk + look-ahead frames. Scores the chunk, updates the cache."""
        cached, logits, embeds, silence = self._infer(mel)
        n_cached = len(cached)
        self._cache.update(np.concatenate([cached, embeds], axis=0), logits, silence, self.chunk_len)
        probs = _sigmoid(logits[n_cached * SUB : (n_cached + self.chunk_len) * SUB]).astype(np.float32)
        with self._lock:
            pos = (self._n_frames + np.arange(len(probs))) % self._hist_len
            self._hist[pos] = probs
            self._n_frames += len(probs)

    # ---- reading

    @property
    def time(self) -> float:
        """Seconds of audio fed so far."""
        return self._samples_fed / SR

    @property
    def processed_time(self) -> float:
        """Seconds of the stream that have final speaker probabilities."""
        return self._n_frames * FRAME_S

    def _provisional(self):
        """Under _proc_lock: estimate for the frames after processed_time, from the audio already
        here (no look-ahead yet, cache untouched)."""
        mel = self._mel[: len(self._mel) // SUB * SUB]
        if not len(mel):
            return np.zeros((0, 8), np.float32)
        cached, logits, _, _ = self._infer(mel)
        return _sigmoid(logits[len(cached) * SUB : len(cached) * SUB + len(mel)]).astype(np.float32)

    def probs(self, t_start: float, t_end: float, provisional: bool = False):
        """(first_frame_time, (frames, 8) probabilities) for the part of [t_start, t_end) that is
        known: 10 ms frames, columns = session-stable speaker ids."""
        f0, f1 = max(0, int(round(t_start / FRAME_S))), int(round(t_end / FRAME_S))
        if provisional and f1 > self._n_frames:
            with self._proc_lock:                # no step can commit while the tail is estimated
                tail = self._provisional()
                first, out = self._known(f0, f1)
                n = self._n_frames
            piece = tail[max(f0, n) - n : max(f1 - n, 0)]
            if len(piece):
                if not len(out):
                    first = max(f0, n)
                out = np.concatenate([out, piece], axis=0)
        else:
            first, out = self._known(f0, f1)
        return first * FRAME_S, out

    def _known(self, f0: int, f1: int):
        with self._lock:
            n = self._n_frames
            a, b = max(f0, n - self._hist_len), min(f1, n)
            if b <= a:
                return min(max(f0, n), max(f1, 0)), np.zeros((0, 8), np.float32)
            return a, self._hist[np.arange(a, b) % self._hist_len]

    def activity(self, t_start: float, t_end: float, provisional: bool = False) -> list[tuple[int, float]]:
        """[(speaker, fraction of the stretch that speaker is active), ...], most active first.
        The fraction is over the frames of the stretch that are known (see processed_time);
        fractions can add up to more than 1 when people talk over each other."""
        _, p = self.probs(t_start, t_end, provisional)
        if not len(p):
            return []
        frac = (p > self.threshold).mean(axis=0)
        order = np.argsort(-frac, kind="stable")
        return [(int(s), float(frac[s])) for s in order if frac[s] > 0]

    def segments(self, t_start: float, t_end: float, provisional: bool = False) -> list[tuple[int, float, float]]:
        """[(speaker, start, end), ...] within the stretch, sorted by start; pieces of different
        speakers may overlap. Pieces shorter than min_dur are dropped."""
        t0, p = self.probs(t_start, t_end, provisional)
        if not len(p):
            return []
        active = (p > self.threshold).astype(np.int8)
        pad = np.zeros((1, active.shape[1]), np.int8)
        d = np.diff(np.concatenate([pad, active, pad]), axis=0)
        segs = []
        for spk in range(active.shape[1]):
            for s, e in zip(np.nonzero(d[:, spk] == 1)[0], np.nonzero(d[:, spk] == -1)[0]):
                if (e - s) * FRAME_S >= self.min_dur:
                    segs.append((int(spk), round(t0 + float(s) * FRAME_S, 2), round(t0 + float(e) * FRAME_S, 2)))
        segs.sort(key=lambda t: (t[1], t[0]))
        return segs
