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

import os
import numpy as np
import onnxruntime as ort

SR = 16000
N_FFT, WIN, HOP, N_MELS, PREEMPH = 512, 400, 160, 128, 0.97
REPO = "onnx-community/Nemotron-3-Diarization-ONNX"
MODEL_FILE = os.environ.get("NEMOTRON_ONNX_FILE", "onnx/model.onnx")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.environ.get("NEMOTRON_DIR", os.path.join(HERE, "data", "models", "nemotron"))

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
