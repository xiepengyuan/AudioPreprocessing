import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input可以是url也可以是本地文件路径
input = "/data1/xiepengyuan/data/cc/tmp/2023-12-11-01-40-29_2023-12-11-03-30-15_000000-000300_0.wav"
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer2_separation_temporal_8k')
result = separation(input)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
