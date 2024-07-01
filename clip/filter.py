import torchaudio
import shutil
import os
import glob


def filter_short(wave_path, dst_dir, min_ms=2000):
    wave, sample_rate = torchaudio.load(wave_path)
    if wave.shape[-1] < min_ms * sample_rate / 1000:
        print(f"Short wave: {wave_path}")
    else:
        shutil.copy(wave_path, dst_dir)


def run_dir_filter_short():
    wave_dir = "/data1/xiepengyuan/audio/tts/OC8145/clip_v2"
    dst_dir = "/data1/xiepengyuan/audio/tts/OC8145/clip_v2_filter_2s"
    os.makedirs(dst_dir, exist_ok=True)
    for wave_path in glob.glob(os.path.join(wave_dir, "*.wav")):
        filter_short(wave_path, dst_dir)


if __name__ == '__main__':
    run_dir_filter_short()
