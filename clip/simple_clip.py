import os
import torchaudio


def slice_audio(audio_path, seconds, dst_dir):
    waveform, sample_rate = torchaudio.load(audio_path)
    length = waveform.shape[1]
    slice_length = int(seconds * sample_rate)
    os.makedirs(dst_dir, exist_ok=True)
    for i in range(0, length, slice_length):
        start = int(i / sample_rate)
        end = int(min((i + slice_length) / sample_rate, length / sample_rate))
        filename = os.path.basename(audio_path)
        filename = os.path.splitext(filename)[0]
        filename = f"{filename}_{str(start).zfill(6)}-{str(end).zfill(6)}.wav"
        slice_path = os.path.join(dst_dir, filename)
        torchaudio.save(slice_path, waveform[:, i:i + slice_length], sample_rate)


if __name__ == '__main__':
    slice_audio(
        "/data1/xiepengyuan/data/cc/audio/12240737/2023-12-07-10-06-31_2023-12-07-10-50-24.wav",
        60*5,
        "/data1/xiepengyuan/data/cc/audio_clip/12240737"
    )
