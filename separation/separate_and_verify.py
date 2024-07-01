# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy
import soundfile as sf
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sv.campplus import Campplus
import torch
import torchaudio


separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer2_separation_temporal_8k')

frame_length_seconds = 2
target_speaker = "12240737"
# campplus = Campplus("/data1/xiepengyuan/models/sv/campplus/campplus_2s.onnx", frame_length_seconds=frame_length_seconds)
# campplus.load_embedding_wav("12240737", "/data1/xiepengyuan/audio//sv/clip_wav/12240737_clip_001.wav")


def separate(wav_path, dst_dir):
    result = separation(wav_path)
    name = os.path.splitext(os.path.basename(wav_path))[0]
    for i, signal in enumerate(result['output_pcm_list']):
        save_path = os.path.join(dst_dir, f'{name}_{i}.wav')
        sf.write(save_path, numpy.frombuffer(signal, dtype=numpy.int16), 8000)


def separate_and_verify(wav_path, dst_dir, target_speaker, min_score=0.45, min_num_verified=15):
    result = separation(wav_path)
    os.makedirs(dst_dir, exist_ok=True)
    for i, signal in enumerate(result['output_pcm_list']):
        wav = numpy.frombuffer(signal, dtype=numpy.int16).astype(numpy.float32) / 2 ** 15
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        import pdb
        pdb.set_trace()
        wav = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)(wav)

        num_verified = 0
        pad_len = campplus.frame_length_seconds * campplus.sample_rate - wav.shape[-1] % (
                  campplus.frame_length_seconds * campplus.sample_rate)
        padded_wav = torch.nn.functional.pad(wav, (0, pad_len))
        for i in range(0, padded_wav.shape[-1], campplus.frame_length_seconds * campplus.sample_rate):
            embedding = campplus.infer(padded_wav[:, i: i + campplus.frame_length_seconds * campplus.sample_rate])
            score = campplus.verify(embedding, target_speaker)
            print(score)
            if score >= min_score:
                num_verified += 1
        if num_verified >= min_num_verified:
            save_path = os.path.join(dst_dir, os.path.basename(wav_path))
            sf.write(save_path, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
            break


def run_dir_separate(src_dir, dst_dir):
    separated_filename = set()
    separated_wav_paths = glob.glob(os.path.join(dst_dir, "*.wav"))
    for separated_wav_path in separated_wav_paths:
        name = os.path.splitext(os.path.basename(separated_wav_path))[0][:-2]
        filename = f"{name}.wav"
        separated_filename.add(filename)
    wave_paths = glob.glob(os.path.join(src_dir, "*.wav"))
    for wave_path in tqdm(wave_paths):
        if os.path.basename(wave_path) in separated_filename:
            print(f"skip: {wave_path}")
            continue
        try:
            separate(wave_path, dst_dir)
        except Exception as e:
            print(f"error: {wave_path}, {e}")


if __name__ == '__main__':
    run_dir_separate("/data1/xiepengyuan/data/cc/audio_clip/12240737", "/data1/xiepengyuan/data/cc/audio_separation/12240737")
