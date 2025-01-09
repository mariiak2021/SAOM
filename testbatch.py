import os
from pathlib import Path
import argparse
from test import test

def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default='/home/mkhan/embclip-rearrangement/Personalize-SAM/data/input/')
    parser.add_argument('--outdir', type=str, default='/home/mkhan/embclip-rearrangement/Personalize-SAM/data/realoutput/')
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print("Args:", args, "\n"), 
    outdir = args.outdir
    indir = args.indir
    files = os.listdir(args.indir)
    files2 = os.listdir(args.outdir)
    for file in files:
        if all(file[:-4] not in file2 for file2 in files2):
            print (file)
            test(file, indir, outdir)
    


if __name__ == '__main__':
    main()