import joblib
import speech_recognition as sr
import numpy as nup
import librosa
import soundfile
import os
import glob
import matplotlib.pyplot as mplt
from IPython.display import Audio
import sys
import librosa.display

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


path = 'microphone-results.wav'
model = joblib.load('finalized_model.sav')

def createWaveplot(data, sr):
    mplt.figure(figsize =(10,3))
    mplt.title('Waveplot for given input audio', size=15)
    librosa.display.waveshow(data, sr=sr)
    mplt.show()

def createSpectrogram(data, sr):
    X= librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    mplt.figure(figsize=(12,3))
    mplt.title('Spectrogram for given input audio', size = 15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    mplt.colorbar()
    mplt.show()

def noise(data):
    noiseAmp = 0.035*nup.random.uniform()*nup.amax(data)
    data = data + noiseAmp*nup.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shiftRange = int(nup.random.uniform(low=-5, high=5)*1000)
    return nup.roll(data, shiftRange)

def pitch(data, samplingRate, pitchFactor=0.7):
    return librosa.effects.pitch_shift(data, samplingRate, pitchFactor)


def extractFeature(filename, mfcc, chroma, mel):
    with soundfile.SoundFile(filename)  as soundFile:
        X = soundFile.read()
        sampleRate = soundFile.samplerate
        if chroma:
            stft=nup.abs(librosa.stft(X))
        result=nup.array([][:])
        if mfcc:
            mfccs = nup.mean(librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T, axis=0)
            result=nup.hstack((result, mfccs))
        if chroma:
            chroma=nup.mean(librosa.feature.chroma_stft(S=stft, sr= sampleRate).T, axis =0)
            result=nup.hstack((result, chroma))
        if mel:
            mel=nup.mean(librosa.feature.melspectrogram(X, sr=sampleRate).T, axis=0)
            result=nup.hstack((result, mel))
    return result

def loadData(file):
    x=[]
    for file in glob.glob("*.wav"):
        fileName = os.path.basename(file)
        feature=extractFeature(fileName, mfcc=True, chroma=True, mel=True)
        x.append(feature)
    return x



recognizer =  sr.Recognizer()
with sr.Microphone() as source:
            print('Clearing Background noice....')
            recognizer.adjust_for_ambient_noise(source, duration=5)
            print('Waiting for your message')
            recorded_audio =recognizer.listen(source, phrase_time_limit=10)
            # print(type(recorded_audio))
            with open("microphone-results.wav", "wb") as f:
                        f.write(recorded_audio.get_wav_data())
            print("Done")




xTest = loadData(recorded_audio)
# print(xTrain, xTest)
yPred = model.predict(xTest)
data, samplingRate = librosa.load(path)
createWaveplot(data, samplingRate)
createSpectrogram(data, samplingRate)
Audio(path)
x = noise(data)
mplt.figure(figsize=(14,4))
mplt.title('noise graph for given input audio', size = 15)
librosa.display.waveshow(y=x, sr=samplingRate)
mplt.show()
Audio(x, rate=samplingRate)


print(yPred[0])
