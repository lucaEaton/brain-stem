import librosa
import numpy as np
import stempeg
from torch.utils.data import Dataset
import os

class musdb18(Dataset):
    def __init__(self, audio_dir):
        self.dir = audio_dir
        self.song_list = [f for f in os.listdir(audio_dir) if f.endswith('.mp4')]
        self.song_list.sort()

    def __len__(self):
        return len(self.song_list)

    # Basically loads both the mixture n the stems, extract features, trims both at the same interval to remove extra
    # noise, we np.mean the stems due to it being stereo as it has left n right channels we want it to be mono then I
    # changed the crop function to also include the audio stems because we want it to be the same crop not different
    # sections. Then return the new data. ~ FOR DEREK since git is fucked rn
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.song_list[index])
        y, sr_native = librosa.load(path)
        y_trim, index_interval = librosa.effects.trim(y, top_db=20)
        mixture_stem = np.abs(librosa.stft(y_trim))

        # loading/reading file 2x (time complexity increase) ~ need to fix
        stems, rate = stempeg.read_stems(path)
        vocal_stem = np.mean(stems[4], axis=1)

        start_idx, end_idx = index_interval
        vocal_stem_trim = vocal_stem[start_idx:end_idx]
        vocal_stem = np.abs(librosa.stft(vocal_stem_trim))

        if mixture_stem.shape[1] > 173:
            mixture_stem,vocal_stem = self._crop(mixture_stem, vocal_stem,  173)

        return mixture_stem, vocal_stem, self.song_list[index]
        # we want to return a tensor of dimensions like 1025 * 173 * 1

    def _crop(self, mixture, stem, dur):
        total_mixture_frames = mixture.shape[1]

        start = np.random.randint(0, total_mixture_frames - dur)
        return mixture[:, start:start + dur], stem[:, start:start + dur]

# test main function
# if __name__ == '__main__':
#     musdb18 = musdb18('../dataset/train')
#     dataloader = DataLoader(musdb18,
#                             batch_size=8,
#                             shuffle=True)
#     data = next(iter(dataloader))
#
#     mixture_stft, vocals_stft,  song_name = data
#     print(mixture_stft.shape)
#     print(vocals_stft.shape)
#     print(mixture_stft)
#     print(vocals_stft)
#     print(song_name)


# kinda have the model?
# dataloader
# stft/inverse stft in C++
# training loop
# testing


