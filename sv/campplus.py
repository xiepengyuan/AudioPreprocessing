# -*- coding: utf-8 -*-
import glob

import onnxruntime as ort
from scipy.spatial.distance import cosine
import os
import sys
import shutil
import torch
import torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sv.processor import FBank


def cosine_similarity(embedding1, embedding2):
    embedding1 = embedding1.reshape(-1)
    embedding2 = embedding2.reshape(-1)
    return 1 - cosine(embedding1, embedding2)


def load_wav(wav_file, obj_fs=16000):
    """
        wav shape: ()
    """
    wav, fs = torchaudio.load(wav_file)
    if fs != obj_fs:
        # print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
        wav, fs = torchaudio.sox_effects.apply_effects_tensor(
            wav, fs, effects=[['rate', str(obj_fs)]]
        )
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    return wav


class Campplus:
    def __init__(self, model_path, sample_rate=16000, frame_length_seconds=3, frame_shift_seconds=1):
        self.sample_rate = sample_rate  # 约定数据16k
        self.frame_length_seconds = frame_length_seconds
        self.frame_shift_seconds = frame_shift_seconds
        ort_session_options = ort.SessionOptions()
        ort_session_options.intra_op_num_threads = 1
        ort_session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, ort_session_options)
        self.feature_extractor = FBank(80, sample_rate=sample_rate, mean_nor=True)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.embedding_dict = {}

    def load_embedding_wav(self, speaker, wav_file):
        wav = load_wav(wav_file, obj_fs=self.sample_rate)
        print(wav.shape)
        embedding = self.infer(wav)
        if speaker not in self.embedding_dict:
            self.embedding_dict[speaker] = []
        self.embedding_dict[speaker].append(embedding)

    def infer(self, wav):
        feat = self.feature_extractor(wav).unsqueeze(0).numpy()
        embedding = self.session.run([self.output_name], {self.input_name: feat})[0]
        return embedding

    def recognize(self, embedding):
        result = {}
        for speaker, target_embeddings in self.embedding_dict.items():
            max_similarity = 0
            for target_embedding in target_embeddings:
                similarity = cosine_similarity(embedding, target_embedding)
                max_similarity = max(max_similarity, similarity)
            result[speaker] = max_similarity
        return result

    def verify(self, embedding, target_speaker):
        max_similarity = 0
        for target_embedding in self.embedding_dict[target_speaker]:
            similarity = cosine_similarity(embedding, target_embedding)
            max_similarity = max(max_similarity, similarity)
        return max_similarity

    def recognize_wav(self, wav, target_speaker):
        pad_len = self.frame_length_seconds * self.sample_rate - wav.shape[-1] % (self.frame_length_seconds * self.sample_rate)
        padded_wav = torch.nn.functional.pad(wav, (0, pad_len))
        results = []
        for i in range(0, padded_wav.shape[-1], self.frame_length_seconds * self.sample_rate):
            embedding = self.infer(padded_wav[:, i: i+self.frame_length_seconds * self.sample_rate])
            score = self.verify(embedding, target_speaker)
            start_time = i / self.sample_rate
            end_time = (i + self.frame_length_seconds * self.sample_rate) / self.sample_rate
            results.append(
                {
                    "score": score,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
            if score > 0.4:
                print(score, start_time, end_time)
            if i > 10000000:
                break
        return results

    def verify_long_wave(self, wav, target_speaker, min_score=0.4, min_num_verified=10):
        pad_len = self.frame_length_seconds * self.sample_rate - wav.shape[-1] % (
                    self.frame_length_seconds * self.sample_rate)
        padded_wav = torch.nn.functional.pad(wav, (0, pad_len))
        num_verified = 0
        for i in range(0, padded_wav.shape[-1], self.frame_length_seconds * self.sample_rate):
            embedding = self.infer(padded_wav[:, i: i + self.frame_length_seconds * self.sample_rate])
            score = self.verify(embedding, target_speaker)
            if score > min_score:
                num_verified += 1
            if num_verified >= min_num_verified:
                return True
        return False


def test():
    campplus = Campplus("/data1/xiepengyuan/models/sv/campplus/campplus_2s.onnx", frame_length_seconds=2)
    campplus.load_embedding_wav("12240737", "./clip_wav/12240737_clip_001.wav")
    wave_path = "../samples/12240737_001.wav"
    wav = load_wav(wave_path)
    results = campplus.recognize_wav(wav, "12240737")
    tmp_results = []
    for result in results:
        if result["score"] <= 0.4:
            continue
        if not tmp_results:
            tmp_results.append(result)
            continue
        if tmp_results[-1]["end_time"] == result["start_time"]:
            tmp_results.append(result)
        else:
            print(tmp_results[0]["start_time"], tmp_results[-1]["end_time"])
            tmp_results = []
    if tmp_results:
        print(tmp_results[0]["start_time"], tmp_results[-1]["end_time"])


def run_single_verify_long_wave():
    campplus = Campplus("/data1/xiepengyuan/models/sv/campplus/campplus_2s.onnx", frame_length_seconds=2)
    campplus.load_embedding_wav("12240737", "./clip_wav/12240737_2023-12-12-10-08-54_2023-12-12-15-07-19_001_8k.wav")
    wave_path = "/data1/xiepengyuan/data/cc/audio_separation/12240737/2023-12-11-10-07-07_2023-12-11-14-22-40_004800-005100_0.wav"
    wav = load_wav(wave_path)
    is_verified = campplus.verify_long_wave(wav, "12240737")
    print(is_verified)


def run_dir_verify_long_wave(src_dir, dst_dir):
    campplus = Campplus("/data1/xiepengyuan/models/sv/campplus/campplus_2s.onnx", frame_length_seconds=2)
    campplus.load_embedding_wav("12240737", "./clip_wav/12240737_2023-12-12-10-08-54_2023-12-12-15-07-19_001_8k.wav")
    wav_paths = glob.glob(os.path.join(src_dir, "*.wav"))
    wav_paths = sorted(wav_paths)
    os.makedirs(dst_dir, exist_ok=True)
    for wav_path in wav_paths:
        wav = load_wav(wav_path)
        is_verified = campplus.verify_long_wave(wav, "12240737")
        if is_verified:
            print(is_verified, wav_path)
            shutil.copy(wav_path, os.path.join(dst_dir, os.path.basename(wav_path)))


if __name__ == '__main__':
    # test()
    run_dir_verify_long_wave(
        "/data1/xiepengyuan/data/cc/audio_separation/12240737",
        "/data1/xiepengyuan/data/cc/audio_separation_select/12240737"
    )
