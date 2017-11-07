import moviepy.editor as mp
import librosa
import cv2
from src.vectorise_image import image2vector
from src.mfcc_inversion import MfccInversion
from os import listdir
from os.path import isfile, join
import numpy as np


def load_generic_from_saved_audio_video(directory, file, sample_rate, Mfcc_Inversion, i2v):

    clip = mp.VideoFileClip(directory + file)

    fps = int(clip.fps + 0.1)
    if fps not in [24, 25, 30]:
        return None

    audio, _ = librosa.load(directory + file.split(".")[0] + ".wav", sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    len = audio.shape[0]
    len = len - (len % sample_rate)

    sample_size = int(sample_rate / 20)

    # to get frame
    # clip.get_frame(0)
    # to get image instance from numpy array
    # fps = 30
    num_frames = int(len / sample_rate) * 20
    print("num frames")
    print(num_frames)
    # for i in range(num_frames - 1):
    i = 0
    j = 0
    skipEvery = 3 if fps==30 else 5 if fps==25 else 6 if fps==24 else None

    res = []

    imgVecs = np.zeros((int(num_frames/120)*30, 512))
    mfcc_feats = np.zeros((int(num_frames/120)*30, 32))

    while j < (int(num_frames/120)*30):
        # print("here")
        # print(i)
        if i % 10 == 0:
            print(i)
        if i % skipEvery == 0:
            i+=1
            continue
        # print("step1")
        img = cv2.resize(clip.get_frame(i), (30, 30))
        # print(img.shape)
        # print(img)
        img = img/255.0

        img = img.reshape((1, 30, 30, 3))
        # print(img.shape)
        # print("step1.5")
        imgVec = i2v.convert(img)[0,0,0,:]
        # print("step2")
        fragment = audio[j * sample_size: (j + 1) * sample_size]
        fragment = fragment.reshape(fragment.shape[0])
        # print(fragment.shape)
        mfcc_feat = Mfcc_Inversion.sound2mfcc(fragment, sample_rate)[:,0]
        # print(mfcc_feat.shape)
        # print("step3")
        # # mfcc_feat = mfcc.reshape
        imgVecs[j, :] = imgVec
        mfcc_feats[j, :] = mfcc_feat
        i+=1
        j+=1
        # print("here2")
        # yield a set of data for each frame and corresponding audio's feature
        # yield mfcc_feat, imgVec
        # res.append([mfcc_feat, imgVec])

    return imgVecs, mfcc_feats


def main():
    directory = "../data/save/"
    filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
    files = []
    for filename in filenames:
        if filename.split(".")[1] == "mp4":
            files.append(filename)

    mfccInversion = MfccInversion()
    img2vec = image2vector([30, 30, 3])

    # data = np.zeros((len(files), 20, 32, 512))
    # data = np.zeros((len(files), 1200, 32, 512))
    imgVecs = np.zeros((len(files), 20*30, 512))
    mfcc_feats = np.zeros((len(files), 20*30, 32))

    sample_rate = 44100
    i = 0
    for file in files:
        _imgVecs, _mfcc_feats = load_generic_from_saved_audio_video(directory, file, sample_rate, mfccInversion, img2vec)
        imgVecs[i,:,:] = _imgVecs
        mfcc_feats[i,:,:] = _mfcc_feats
        i += 1
    np.save("training_imgVecs", imgVecs)
    np.save("training_mfcc_feat", mfcc_feats)

if __name__ == "__main__":
   main()