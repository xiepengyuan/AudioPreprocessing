# -*- coding: utf-8 -*-

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import torchaudio
import glob
import re


inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)


def asr_and_clip(wave_path, anno_path, speaker, dst_dir):
    shift_start_ms = -50
    shift_end_ms = 50
    wave, sr = torchaudio.load(wave_path)
    max_ms = int(wave.shape[-1] / sr * 1000)
    resampler = torchaudio.transforms.Resample(sr, 16000)
    resampled_wave = resampler(wave).numpy()[0]
    rec_result = inference_pipeline(audio_in=resampled_wave)
    if 'text' not in rec_result:
        print(f"Text is not recognized，ignoring...，{wave_path}\n")
        return
    print(rec_result)
    # 根据分句结果来切分音频
    sentences = rec_result["sentences"]
    tmp_sentences = []
    merged_sentences = []
    for sentence_index, sentence in enumerate(sentences):
        # 判断句号是否合理
        if sentence["text"].endswith("。") or sentence["text"].endswith("？") or sentence["text"].endswith("！"):
            if sentence_index != len(sentences) - 1:
                # 当前句的最后一个字符，和下一句的第一个字符间隔有多长
                cur_ts = sentence["ts_list"][-1]
                next_ts = sentences[sentence_index+1]["ts_list"][0]
                # 间隔<100ms，去掉句号，并合并
                if next_ts[0] - cur_ts[1] <= 200:
                    print("去掉末尾句号，并合并", cur_ts, next_ts, sentence["text"], sentences[sentence_index+1]["text"])
                    sentences[sentence_index+1]["text"] = sentence["text"][:-1] + sentences[sentence_index+1]["text"]
                    sentences[sentence_index+1]["ts_list"] = sentence["ts_list"] + sentences[sentence_index+1]["ts_list"]
                    continue

        # 判断句中是否要添加逗号
        new_text = ""
        for i, ts in enumerate(sentence["ts_list"]):
            if i == 0:
                new_text += sentence["text"][0]
                continue
            # 是否要划分句子
            if sentence["ts_list"][i][0] - sentence["ts_list"][i - 1][1] >= 1500:
                print("切句", sentence["ts_list"][i - 1], sentence["ts_list"][i], sentence["text"][i - 1:i + 1],
                      sentence["text"])
                if sentence_index != len(sentences) - 1:
                    sentences[sentence_index + 1]["text"] = sentence["text"][i:-1] + sentences[sentence_index + 1]["text"]
                    sentences[sentence_index + 1]["ts_list"] = sentence["ts_list"][i:] + sentences[sentence_index + 1]["ts_list"]
                    sentence["text"] = sentence["text"][:i] + "。"
                    sentence["ts_list"] = sentence["ts_list"][:i]
                    sentence["end"] = sentence["ts_list"][-1][1]
                    sentences[sentence_index]["text"] = sentence["text"]
                    sentences[sentence_index]["ts_list"] = sentence["ts_list"]
                    sentences[sentence_index]["end"] = sentence["end"]
                    break
            # 间隔>=250ms，加上逗号
            elif sentence["ts_list"][i][0] - sentence["ts_list"][i - 1][1] >= 250:
                print("加逗号", sentence["ts_list"][i - 1], sentence["ts_list"][i], sentence["text"][i-1:i+1], sentence["text"])
                new_text += "，"
            new_text += sentence["text"][i]
        # 补充标点符号
        new_text += sentence["text"][-1]
        sentence["text"] = new_text

        # 判断上一句是否要去掉末尾逗号
        if tmp_sentences:
            # 上一句的最后一个字符，和当前句的第一个字符间隔有多长
            last_sentence = tmp_sentences[-1]
            last_ts = last_sentence["ts_list"][-1]
            cur_ts = sentence["ts_list"][0]
            # 间隔<100ms，去掉末尾逗号
            if cur_ts[0] - last_ts[1] < 100:
                # 去掉末尾逗号
                print("去掉末尾逗号", last_ts, cur_ts, last_sentence["text"], sentence["text"])
                if tmp_sentences[-1]["text"].endswith("，"):
                    tmp_sentences[-1]["text"] = tmp_sentences[-1]["text"][:-1]

        # 判断与下一句的间隔是否太长
        if sentence_index != len(sentences) - 1:
            # 当前句的最后一个字符，和下一句的第一个字符间隔有多长
            cur_ts = sentence["ts_list"][-1]
            next_ts = sentences[sentence_index + 1]["ts_list"][0]
            # 间隔>=1500ms，改为一句
            if next_ts[0] - cur_ts[1] >= 1500:
                print("作为一句，改为句号", cur_ts, next_ts, sentence["text"], sentences[sentence_index + 1]["text"])
                sentences[sentence_index]["text"] = sentences[sentence_index]["text"][:-1] + "。"

        tmp_sentences.append(sentence)

        # 输出句子
        if sentence["text"][-1] in ["。", "？", "！"] or sentence_index == len(sentences) - 1:
            if sentence_index == len(sentences) - 1:
                new_sentences = []
                for s in tmp_sentences:
                    if "非常满意" in s["text"]:
                        break
                    new_sentences.append(s)
                tmp_sentences = new_sentences
            print(tmp_sentences)
            text = "".join([s["text"] for s in tmp_sentences])
            if text == "":
                continue
            if text[-1] == "，":
                text = text[:-1] + "。"
            start = tmp_sentences[0]["ts_list"][0][0]
            end = tmp_sentences[-1]["end"]
            merged_sentences.append(
                {
                    "text": text,
                    "start": start + shift_start_ms,
                    "end": end + shift_end_ms
                }
            )
            tmp_sentences = []

    name = os.path.splitext(os.path.basename(wave_path))[0]
    for sentence in merged_sentences:
        text = sentence["text"]
        start = max(0, sentence["start"])
        end = min(max_ms, sentence["end"])
        filename = f"{name}_{str(start).zfill(6)}_{str(end).zfill(6)}.wav"
        clip_path = os.path.join(dst_dir, filename)
        torchaudio.save(clip_path, wave[:, int(start * sr / 1000):int(end * sr / 1000)], sample_rate=sr)
        save_path = f"./dataset/{speaker}/{filename}"
        anno_text = save_path + "|" + speaker + "|ZH|" + text + "\n"
        with open(anno_path, 'a', encoding='utf-8') as f:
            f.write(anno_text)
        print(anno_text)


def run_single_asr_and_clip():
    speaker_name = "12240737"
    wave_path = f"/data1/xiepengyuan/data/cc/audio_separation_select_denoised/12240737/2023-12-07-10-06-31_2023-12-07-10-50-24_000000-000300_0.wav"
    anno_path = f"../outputs/{speaker_name}/12240737_001_d.list"
    dsr_dir = f"../outputs/{speaker_name}/clip_12240737_001_d"
    os.makedirs(dsr_dir, exist_ok=True)
    asr_and_clip(wave_path, anno_path, speaker_name, dsr_dir)


def run_dir_asr_and_clip():
    speaker = "12240737"
    wave_dir = "/data1/xiepengyuan/data/cc/audio_separation_select_denoised/12240737"
    anno_path = "/data1/xiepengyuan/workspace/audio/Bert-VITS2/filelists/12240737/12240737.list"
    dsr_dir = "/data1/xiepengyuan/audio/tts/12240737/clip"
    os.makedirs(dsr_dir, exist_ok=True)
    complete_set = set()
    os.makedirs(os.path.dirname(anno_path), exist_ok=True)
    if os.path.exists(anno_path):
        with open(anno_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                pt, _, _, _ = line.strip().split('|')
                base_wave_filename = "_".join(os.path.splitext(os.path.basename(pt))[0].split("_")[:-2]) + ".wav"
                complete_set.add(base_wave_filename)
    print(complete_set)
    for wave_path in glob.glob(os.path.join(wave_dir, "*.wav")):
        if os.path.basename(wave_path) in complete_set:
            print(f'{wave_path} is already done, skip!')
            continue
        print(wave_path)
        asr_and_clip(wave_path, anno_path, speaker, dsr_dir)


if __name__ == '__main__':
    run_dir_asr_and_clip()
    # run_single_asr_and_clip()
