import argparse
import os
import random
import json
import soundfile
import numpy as np
import librosa
from collections import defaultdict
from tqdm import tqdm



def load_data(data_path):
    wav2speaker = dict()
    wav2transcript = dict()
    speaker2wav = defaultdict(list)
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            speaker, wav_path, transcript = line.split('\t')
            wav2speaker[wav_path] = speaker
            wav2transcript[wav_path] = transcript
            speaker2wav[speaker].append(wav_path)
    return wav2speaker, speaker2wav, wav2transcript


def main(args):
    wav2speaker, speaker2wav, wav2transcript = load_data(args.data_path)
    path_list = wav2speaker.keys()
    speaker_list = set(list(speaker2wav.keys()))
    res = []
    base_path = args.data_dir + '-%dmix' % args.num_speakers
    for idx, wav_path in tqdm(enumerate(path_list), total=len(path_list), desc='Building %d Mix Data' % args.num_speakers):
        res_id = os.path.join(base_path, base_path + '-%d' % idx)
        res_path = res_id + '.wav'
        audio, _ = soundfile.read(wav_path)
        used_speakers = {wav2speaker[wav_path]}
        delays = [0.0]
        wav_paths = [wav_path]
        transcripts = [wav2transcript[wav_path]]
        for _ in range(args.num_speakers - 1):
            while True:
                speaker = random.choice(list(speaker_list - used_speakers))
                wav_path = random.choice(speaker2wav[speaker])
                additional_audio, _ = soundfile.read(wav_path)
                delay = random.uniform(0.5, audio.shape[0] / args.sampling_rate)
                delay_frame = int(delay * args.sampling_rate)
                additional_audio = np.append(np.zeros(delay_frame), additional_audio)
                if is_overlap(audio, additional_audio):
                    used_speakers.add(speaker)
                    delays.append(delay)
                    wav_paths.append(wav_path)
                    transcripts.append(wav2transcript[wav_path])
                    audio = combine_audio(audio, additional_audio)
                    break
        sorted_idx = np.argsort(delays).tolist()
        wav_paths = np.array(wav_paths)[sorted_idx].tolist()
        transcripts = np.array(transcripts)[sorted_idx].tolist()
        used_speakers = np.array(list(used_speakers))[sorted_idx].tolist()
        delays = np.array(delays)[sorted_idx].tolist()
        res.append({'id': res_id,
                    'mixed_wav': res_path,
                    'texts': transcripts,
                    'wavs': wav_paths,
                    'speakers': used_speakers,
                    'delays': delays})

    json.dump(res, args.json_path)



def combine_audio(audio, additional_audio):
    target_length = max(len(audio), len(additional_audio))
    audio = librosa.util.fix_length(audio, size=target_length)
    additional_audio = librosa.util.fix_length(additional_audio, size=target_length)
    audio = audio + additional_audio
    return audio


def is_overlap(audio, additional_audio):
    target_length = max(len(audio), len(additional_audio))
    audio = librosa.util.fix_length(audio, size=target_length)
    additional_audio = librosa.util.fix_length(additional_audio, size=target_length)
    nonzero_indices = np.nonzero(audio)
    return additional_audio[nonzero_indices].sum() != 0.0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=3407)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--num_speakers', type=int, required=True)
    parser.add_argument('--sampling_rate', type=int, default=160000)
    parser.add_argument('--json_path', type=str, required=True)
    _args = parser.parse_args()
    random.seed(_args.random_seed)
    main(_args)
