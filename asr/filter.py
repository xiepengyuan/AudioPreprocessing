# -*- coding: utf-8 -*-

import os
import re
import random


def filter_non_chinese(text):
    if text == "":
        return ""
    english_pattern = re.compile(r'[A-Za-z]', re.S)
    res = re.findall(english_pattern, text)
    if len(res):
        print(f'Skip non-chinese text : {text}')
        return ""
    return text


def filter_short_audio(path, min_ms):
    if path == "":
        return True
    name = os.path.splitext(os.path.basename(path))[0]
    path_split = name.split("_")
    duration_ms = int(path_split[-1]) - int(path_split[-2])
    if duration_ms < min_ms:
        return True
    return False


def filter_short_sentence(text):
    if text == "":
        return ""
    res = re.findall(r'[\u4e00-\u9fff]', text)
    if len(res) < 5:
        print(f'Skip too short text : {text}')
        return ""
    return text


def filter_special_word(text):
    if text == "":
        return ""
    num_press_pattern = re.compile(r'请按[一二三四五]')
    res = re.findall(num_press_pattern, text)
    if len(res):
        print(f'Skip text with "{res[0]}" : {text}')
        return ""
    return text


def run_filter():
    anno_path = "/data1/xiepengyuan/workspace/audio/Bert-VITS2/filelists/12240737/12240737.list"
    cleaned_anno_path = "/data1/xiepengyuan/workspace/audio/Bert-VITS2/filelists/12240737/12240737_cleaned.list"
    annos = []
    cleaned_new_annos = []
    texts = {}
    is_filter_short_audio = True

    if os.path.exists(anno_path):
        with open(anno_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                annos.append(line.strip())
    else:
        print(f'{anno_path} cannot be found, please confirm that the path is correct')
        return

    random.shuffle(annos)
    for anno in annos:
        path, name, lang, text = anno.split("|")

        if is_filter_short_audio:
            if filter_short_audio(path, 1000):
                continue

        text = filter_non_chinese(text)
        text = filter_short_sentence(text)
        text = filter_special_word(text)
        if text == "":
            continue

        # 判断重复次数
        if text not in texts:
            texts[text] = 0
        if texts[text] >= 3:
            print(f'Skip text repeat 3 times : {text}')
            continue
        texts[text] += 1

        new_anno = f'{path}|{name}|{lang}|{text}'
        cleaned_new_annos.append(new_anno)

    with open(cleaned_anno_path, 'w', encoding='utf-8') as f:
        for new_anno in cleaned_new_annos:
            f.write(new_anno + '\n')
    print(f"total: {len(cleaned_new_annos)}")


if __name__ == '__main__':
    run_filter()
