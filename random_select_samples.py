# -*- coding: utf-8 -*-

import random


def run():
    items = []
    with open("/data1/xiepengyuan/workspace/audio/Bert-VITS2/filelists/naixi_toy/naixi.list") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items.append(line)
        random_items = random.sample(items, k=360)

    with open("/data1/xiepengyuan/workspace/audio/Bert-VITS2/filelists/naixi_toy/naixi_random.list", "w") as f:
        for item in random_items:
            f.write(item + "\n")


if __name__ == '__main__':
    run()
