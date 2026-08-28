# AudioMuse-AI-SAE - https://github.com/NeptuneHub/AudioMuse-AI-SAE
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI-SAE/blob/main/LICENSE>
#
# Method from "Steering dense music retrieval with open-vocabulary concept discovery"
# by Julien Guinot, Alain Riou, Elio Quinton and Gyorgy Fazekas
# <https://arxiv.org/abs/2608.08757>, used under CC BY 4.0
# <https://creativecommons.org/licenses/by/4.0/>

"""ONNX wrappers around the DCLAP audio and text towers.

Mirrors tasks/clap_analyzer.py from the AudioMuse-AI core so that the embeddings
the SAE is trained on are bit compatible with the ones the analyser writes into
the clap_embedding table: same 48 kHz resampling, same int16 round trip, same
10 second window with a 5 second hop and the same mel front end. Using a
different front end here would train the dictionary on a distribution the
production pipeline never produces.

Main Features:
* DclapAudioEncoder.encode_file returns the per segment 512 dim matrix as well
  as the mean pooled track vector, so both granularities come from one decode.
* The int16 quantisation round trip is reproduced exactly; dropping it shifts
  the embeddings enough to matter for a dictionary.
* DclapTextEncoder embeds free form concepts with the RoBERTa tokenizer used by
  the CLAP text tower, for concept_attribution.py and steer.py.
* Sessions are single threaded by default so a process pool can fan out over
  tracks without oversubscribing the CPU.
"""

import numpy as np

SAMPLE_RATE = 48000
SEGMENT_SAMPLES = 480000
HOP_SAMPLES = 240000
EMBEDDING_DIM = 512

MEL_N_FFT = 2048
MEL_HOP_LENGTH = 480
MEL_N_MELS = 128
MEL_FMIN = 0
MEL_FMAX = 14000
MEL_TRANSPOSE = False

TEXT_MAX_LENGTH = 77


_MEL_BASIS = {}


def mel_basis(sr=SAMPLE_RATE):
    import librosa

    key = (sr, MEL_N_FFT, MEL_N_MELS, MEL_FMIN, MEL_FMAX)
    basis = _MEL_BASIS.get(key)
    if basis is None:
        basis = librosa.filters.mel(
            sr=sr, n_fft=MEL_N_FFT, n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX
        ).astype(np.float32)
        _MEL_BASIS[key] = basis
    return basis


def compute_mel_spectrogram(audio, sr=SAMPLE_RATE):
    import librosa

    spectrum = (
        np.abs(
            librosa.stft(
                audio,
                n_fft=MEL_N_FFT,
                hop_length=MEL_HOP_LENGTH,
                win_length=MEL_N_FFT,
                window='hann',
                center=True,
                pad_mode='reflect',
            )
        )
        ** 2.0
    )
    mel = mel_basis(sr) @ spectrum
    mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    if MEL_TRANSPOSE:
        mel = mel.T
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def quantize_roundtrip(audio):
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767.0).astype(np.int16)
    return (audio / 32767.0).astype(np.float32)


def split_segments(audio):
    total = len(audio)
    if total <= SEGMENT_SAMPLES:
        return [np.pad(audio, (0, SEGMENT_SAMPLES - total), mode='constant')]
    segments = []
    for start in range(0, total - SEGMENT_SAMPLES + 1, HOP_SAMPLES):
        segments.append(audio[start : start + SEGMENT_SAMPLES])
    if len(segments) * HOP_SAMPLES < total:
        segments.append(audio[-SEGMENT_SAMPLES:])
    return segments


def load_audio(path, sr=SAMPLE_RATE):
    import librosa

    audio, _sr = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def _make_session(model_path, providers, intra_threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if providers is None:
        providers = ['CPUExecutionProvider']
    return ort.InferenceSession(model_path, sess_options=options, providers=providers)


class DclapAudioEncoder:
    def __init__(self, model_path, providers=None, intra_threads=1):
        self.session = _make_session(model_path, providers, intra_threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def encode_segments(self, segments):
        embeddings = []
        for segment in segments:
            mel = compute_mel_spectrogram(segment)
            output = self.session.run([self.output_name], {self.input_name: mel})[0]
            embeddings.append(np.asarray(output, dtype=np.float32).reshape(-1))
        if not embeddings:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack(embeddings).astype(np.float32)

    def encode_file(self, path):
        audio = load_audio(path)
        if audio is None or audio.size == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), None, 0.0
        duration = len(audio) / SAMPLE_RATE
        audio = quantize_roundtrip(audio)
        segments = self.encode_segments(split_segments(audio))
        if segments.shape[0] == 0:
            return segments, None, duration
        track = segments.mean(axis=0)
        track = track / (np.linalg.norm(track) + 1e-9)
        return segments, track.astype(np.float32), duration


class DclapTextEncoder:
    def __init__(self, model_path, tokenizer_name='roberta-base', providers=None, intra_threads=0):
        from transformers import AutoTokenizer

        self.session = _make_session(model_path, providers, intra_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.output_name = self.session.get_outputs()[0].name

    def encode(self, texts):
        encoded = self.tokenizer(
            list(texts),
            max_length=TEXT_MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='np',
        )
        inputs = {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64),
        }
        output = self.session.run([self.output_name], inputs)[0].astype(np.float32)
        return output / np.linalg.norm(output, axis=1, keepdims=True).clip(1e-9)
