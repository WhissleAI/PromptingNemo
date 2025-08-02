import sys
import glob
import argparse
import json
import librosa


parser = argparse.ArgumentParser(description='Prepare manifest (*.json) file for training or testing NeMo models')
parser.add_argument('-a', '--input', required=True, type=str,
    help="Folder with audio wav files in 16 bit 16 kHz PCM")
parser.add_argument('-o', '--output', default=None, type=str,
    help="output manifest file given, otherwise standout")

args = parser.parse_args()


def main():

    # file pointer for manifest
    
    AudioPaths = glob.glob(args.input+"/*")
    fout = open(args.output, "w") if args.output else sys.stdout

   
    audioFiles = [ x.rstrip() for x in AudioPaths]

    for idx, filename in enumerate(audioFiles):
        
        duration = librosa.get_duration(filename=filename)
        fout.write('{"audio_filepath": "%s", "duration": %f, "text": "%s"}\n' % (filename,duration,"NA"))


if __name__ == "__main__":
    main()

