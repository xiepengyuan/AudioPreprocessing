from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio
import os
import glob

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)


def clip(wave_path, dst_dir):
    wave, sr = torchaudio.load(wave_path)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    resampled_wave = resampler(wave).numpy()[0]
    segments_result = inference_pipeline(audio_in=resampled_wave, audio_fs=16000)
    if "text" not in segments_result:
        print("no segments found in {}".format(wave_path))
        return
    segments = segments_result["text"]
    print(segments)
    name = os.path.splitext(os.path.basename(wave_path))[0]
    for segment in segments:
        clip_path = os.path.join(dst_dir, f"{name}_{str(segment[0]).zfill(6)}_{str(segment[1]).zfill(6)}.wav")
        torchaudio.save(clip_path, wave[:, int(segment[0]*sr/1000):int(segment[1]*sr/1000)], sample_rate=sr)


def run_single():
    wave_path = "../samples/OC8145_00436.wav"
    dst_dir = "../outputs/OC8145"
    os.makedirs(dst_dir, exist_ok=True)
    clip(wave_path, dst_dir)


def run_dir():
    wave_dir = "/data1/xiepengyuan/audio/tts/OC8145/raw_left"
    dst_dir = "/data1/xiepengyuan/audio/tts/OC8145/clip_v2"
    os.makedirs(dst_dir, exist_ok=True)
    for wave_path in glob.glob(os.path.join(wave_dir, "*.wav")):
        clip(wave_path, dst_dir)


if __name__ == '__main__':
    run_dir()
