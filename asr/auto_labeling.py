# -*- coding: utf-8 -*-

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import glob


inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)


def vits_labeling(wave_path, anno_path, speaker):
    save_path = f"./dataset/{speaker}/{os.path.basename(wave_path)}"
    rec_result = inference_pipeline(audio_in=wave_path)
    if 'text' not in rec_result:
        print(f"Text is not recognized，ignoring...，{wave_path}\n")
        return
    anno_text = save_path + "|" + speaker + "|ZH|" + rec_result['text'] + "\n"
    with open(anno_path, 'a', encoding='utf-8') as f:
        f.write(anno_text)
    print(anno_text)


def run_single_vits_labeling():
    wave_path = "/data1/xiepengyuan/audio/tts/OC8145/clip_v2_filter_2s/OC8145_00002_017230_028850.wav"
    anno_path = "../outputs/OC8145/OC8145.list"
    speaker_name = "OC8145"
    vits_labeling(wave_path, anno_path, speaker_name)


def run_dir_vits_labeling():
    speaker = "OC8145"
    wave_dir = "/data1/xiepengyuan/audio/tts/OC8145/clip_v2_filter_2s"
    anno_path = "/data1/xiepengyuan/workspace/audio/Bert-VITS2/filelists/OC8145/OC8145.list"
    complete_list = []
    os.makedirs(os.path.dirname(anno_path), exist_ok=True)
    if os.path.exists(anno_path):
        with open(anno_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                pt, _, _ = line.strip().split('|')
                complete_list.append(os.path.basename(pt))
    for wave_path in glob.glob(os.path.join(wave_dir, "*.wav")):
        if os.path.basename(wave_path) in complete_list:
            print(f'{wave_path} is already done, skip!')
        vits_labeling(wave_path, anno_path, speaker)


if __name__ == '__main__':
    run_dir_vits_labeling()
