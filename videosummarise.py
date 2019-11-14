from __future__ import division
import cv2
import numpy as np
import os
import mahotas
import librosa
from math import *
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import *
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
GENRE_LIST = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
def create_summary(filename, regions):
    subclips = []
    input_video = VideoFileClip(filename)
    last_end = 0
    for (start, end) in regions:
        if start>end:
            start,end=end,start
        subclip = input_video.subclip(start, end)
        subclips.append(subclip)
        last_end = end
    return concatenate_videoclips(subclips)
def get_summary(filename,regions):
    summary = create_summary(filename, regions)
    base, ext = os.path.splitext(filename)
    output = "{0}_1.mp4".format(base)
    summary.to_videofile( output, codec="libx264", temp_audiofile="temp.m4a", remove_temp=True, audio_codec="aac")
    return True
def getknapSack(W, wt, val, n):
    print(n,W)
    K = [[0 for w in range(W + 1)] 
            for i in range(n + 1)]
    for i in range(n + 1): 
        for w in range(W + 1): 
            if i == 0 or w == 0: 
                K[i][w] = 0
            elif wt[i - 1] <= w: 
                K[i][w] = max(val[i - 1]  
                  + K[i - 1][w - wt[i - 1]], 
                               K[i - 1][w]) 
            else: 
                K[i][w] = K[i - 1][w]
    res = K[n][W] 
    segments=[]
    w = W
    for i in range(n, 1, -1): 
        if res <= 0: 
            break
        print(i-1,w)
        if res == K[i - 1][w]: 
            continue
        else: 
            segments.append(i - 1) 
            res = res - val[i - 1] 
            w = abs(w - wt[i - 1])
    return segments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
def extract_image_features(frame , vector_size=32):
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    return global_feature
def extract_audio_features(y,sr):
    timeseries_length = 2
    features = np.zeros((1, timeseries_length , 33), dtype=np.float64)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
    features[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
    features[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
    features[0, :, 14:26] = chroma.T[0:timeseries_length, :]
    features[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]
    return features
def load_model(model_path, weights_path):
    with open(model_path, 'r') as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return trained_model
def get_genre_audio(model, y, sr):
    prediction = model.predict(extract_audio_features(y,sr))
    predict_genre = GENRE_LIST[np.argmax(prediction)]
    return predict_genre
def get_genre_image(model,res):
    futr=extract_image_features(res)
    y=futr
    y=np.resize(y,(2,33))
    futr=np.reshape(y,(1,2,33))
    prediction = model.predict(futr)
    predict_genre = GENRE_LIST[np.argmax(prediction)]
    return predict_genre
cap = cv2.VideoCapture(r"E:\\python3\\posit.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
file_name = r"positaudio.mp3"
audio_time_series, sample_rate = librosa.load(file_name)
length_series = len(audio_time_series)
subdiv=(length_series/length)
#print(subdiv)
audio,imag=[],[]
model=load_model("model.json","model_weights.h5")
for segment in range(0,length_series,int(subdiv)):
    y=audio_time_series[segment:segment+int(subdiv)]
    audio.append(get_genre_audio(model,y,sample_rate))
success = 1
while success:
    success, image = cap.read()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #imag.append(get_genre_image(model,gray))
time=[]
for elm in range(len(audio)):
    time.append(subdiv)
W=5000
prob=[]
for i in range(len(audio)):
    prob.append(int(audio.count(audio[i])*100/len(audio)))
segments=getknapSack(int(W),prob,time,len(audio))
clips=sorted(segments)
result=[]
last_cons,last=clips[0],clips[0]
for i in range(1,len(clips)):
    while(clips[i]==last_cons+1):
        last_cons=clips[i]
    result.append((last,last_cons))
get_summary(r"E:\\python3\\posit.mp4",result)
