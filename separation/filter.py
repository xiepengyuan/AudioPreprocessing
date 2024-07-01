import glob
import os


def filter_not_separated(src_dir):
    wave_paths = glob.glob(os.path.join(src_dir, "*.wav"))
    name_dict = {}
    for wave_path in wave_paths:
        name = "_".join(os.path.splitext(os.path.basename(wave_path))[0].split("_")[:-1])
        if name not in name_dict:
            name_dict[name] = []
        name_dict[name].append(wave_path)
    for name, wave_paths in name_dict.items():
        if len(wave_paths) > 1:
            print(f"{name}: {wave_paths}")
            for wave_path in wave_paths:
                os.remove(wave_path)


if __name__ == '__main__':
    filter_not_separated("/data1/xiepengyuan/data/cc/audio_separation_select/12240737")
