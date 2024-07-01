# -*- coding: utf-8 -*-

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
)

rec_result = inference_pipeline(audio_in='/data1/xiepengyuan/audio/tts/OC8145/clip_v2_filter_2s/OC8145_00469_043440_076090.wav')
print(rec_result)

sentences = rec_result["sentences"]

tmp_sentences = []
for sentence in sentences:
    tmp_sentences.append(sentence)
    if sentence["text"].endswith("."):
        text = "".join([s["text"] for s in tmp_sentences])
        start = tmp_sentences[0]["start"]
        end = tmp_sentences[-1]["end"]
        tmp_sentences = []