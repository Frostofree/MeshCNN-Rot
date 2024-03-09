#!/usr/bin/env python3

import pathlib as pl
import random
import multiprocessing

test_or_train = lambda: random.random() < 0.8
is_rotation = lambda p : len(p) > 4 and all(x.isdigit() for x in p[-4:])

def move(p: pl.Path):
        stems = p.stem.split('_')
        new_path = None
        classname = None
        if is_rotation(stems):
            classname = "_".join(stems[:-3])
            new_path = (pl.Path('./datasets/M40_new_small') / classname / ('train' if test_or_train() else 'test'))
        else:
            classname = p.stem
            new_path = (pl.Path('./datasets/M40_new_small') / classname / ('test'))
        new_path.mkdir(parents=True, exist_ok=True)
        new_file_path = new_path / p.name
        p.rename(new_file_path)
        return p

def main():
    with multiprocessing.Pool(8) as p:
        for res in p.imap_unordered(move, pl.Path('./datasets/M40_r').rglob('*.obj')):
            print(res)

if __name__ == '__main__':
    main()