from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import sys, getopt
from random import shuffle
import random
from os import listdir
from os.path import isfile, join
import time

from pytube import YouTube
import librosa
import subprocess
import moviepy.editor as mp
from PIL import Image

import cv2
from python_speech_features import mfcc
from src.vectorise_image import image2vector
from src.mfcc_inversion import MfccInversion


download = False

# Parameters
learning_rate = 0.001
training_iters = 1000
batch_size = 50
display_step = 10
num_data = 0
# sample_rate = 16000
sample_rate = 44100
directory = "../data"
# Network Parameters
# n_input = 13 # MNIST data input (img shape: 28*28)
n_input = 512 # MNIST data input (img shape: 28*28)
n_steps = 12 # timesteps
n_hidden = 200 # hidden layer num of features
# n_classes = 16*9 # MNIST total classes (0-9 digits)
n_classes = 32*1 # MNIST total classes (0-9 digits)

# tf Graph input
dropout_1 = tf.placeholder(tf.float32)
dropout_2 = tf.placeholder(tf.float32)
dropout_3 = tf.placeholder(tf.float32)

weights = {}
biases = {}


def load_generic_audio_video(directory, sample_rate, video_list, video_index, len_video_list):

    # create or load a list of youtube videos (URL)
    # this function gets called every time the model runs out the given training data

    current_video = video_list[int(video_index % len_video_list)]
    download_youtube(directory, video_name=current_video)
    # clip = mp.VideoFileClip(directory + "/tmp.mp4")
    clip = mp.VideoFileClip(directory + "/tmp.mp4")
    clip.audio.write_audiofile(directory + "/tmp.wav")
    fps = int(clip.fps + 0.1)
    if fps not in [24,25,30]:
        return None
    if fps == 30:
        fps = 24
    audio, _ = librosa.load(directory + "/tmp.wav", sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    len = audio.shape[0]
    len = len - (len % sample_rate)

    sample_size = int(sample_rate / fps)

    # to get frame
    # clip.get_frame(0)
    # to get image instance from numpy array
    fps = 30
    res = []
    hsv_average = np.zeros((144,3))
    num_frames = int(len / 16000) * fps
    for i in range(num_frames-1):
        if fps == 30 and i % 5 == 4:
            continue
        img = cv2.blur(clip.get_frame(i),(100,100))
        img = cv2.resize(img,(16,9))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(144,3)
        Hue_vec = hsv[:,0]
        hsv_average[:, 1:] = hsv_average[:,0:2]
        if sum(Hue_vec) == 0 and i>100:
            continue
        # print(Hue_vec)
        fragment = audio[i * sample_size: (i + 1) * sample_size]
        mfcc_feat = mfcc(fragment, 16000)
        # yield a set of data for each frame and corresponding audio's feature
        yield mfcc_feat, Hue_vec
    #     res.append([mfcc_feat, Hue_vec])
    # return res

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
    while i < (num_frames/60):
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
        imgVec = i2v.convert(img)
        # print("step2")
        fragment = audio[j * sample_size: (j + 1) * sample_size]
        fragment = fragment.reshape(fragment.shape[0])
        # print(fragment.shape)
        mfcc_feat = Mfcc_Inversion.sound2mfcc(fragment, sample_rate)
        # print(mfcc_feat.shape)
        # print("step3")
        # # mfcc_feat = mfcc.reshape
        i+=1
        j+=1
        # print("here2")
        # yield a set of data for each frame and corresponding audio's feature
        # yield mfcc_feat, imgVec
        res.append([mfcc_feat, imgVec])
    return res


def download_youtube(directory, video_name=None):
    subprocess.call(["rm", directory+"/tmp.wav", directory+"/tmp.mp4"])

    # video_id = "h6yJEHHT5eA"
    try:
        youtube = YouTube(video_name)
        youtube.set_filename('tmp')
    except:
        print("there is no video")

    try:
        video = youtube.get('mp4', '360p')
    except:
        print("there is no video for this setting")

    video.download(directory)


def RNN(x):
    weight = tf.Variable(tf.random_normal([n_hidden, n_classes]), name="weight")
    bias = tf.Variable(tf.random_normal([n_classes]), name="bias")

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weight) + bias

def RNN2(x):
    # weight_1 = tf.Variable(tf.random_normal([n_hidden, 240]), name="weight_1")
    # bias_1 = tf.Variable(tf.random_normal([240]), name="bias_1")
    # weight_2 = tf.Variable(tf.random_normal([240, n_classes]), name="weight_2")
    # bias_2 = tf.Variable(tf.random_normal([n_classes]), name="bias_2")
    weight_1 = tf.get_variable("weight_1", [n_hidden, 240], dtype=tf.float32, initializer=tf.random_normal_initializer())
    bias_1 = tf.get_variable("bias_1", [240], dtype=tf.float32, initializer=tf.random_normal_initializer())
    weight_2 = tf.get_variable("weight_2", [240, n_classes], dtype=tf.float32, initializer=tf.random_normal_initializer())
    bias_2 = tf.get_variable("bias_2", [n_classes], dtype=tf.float32, initializer=tf.random_normal_initializer())

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    fc_output = tf.nn.dropout(tf.sigmoid(tf.matmul(outputs[-1], weight_1) + bias_1), dropout_1)

    return tf.nn.dropout(tf.matmul(fc_output, weight_2) + bias_2, dropout_1)

def NN(x):
    W_1 = tf.get_variable("W_1", [n_input, 360], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_1 = tf.get_variable("b_1", [360], dtype=tf.float32, initializer=tf.random_normal_initializer())


    W_2 = tf.get_variable("W_2", [360, 144], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_2 = tf.get_variable("b_2", [144], dtype=tf.float32, initializer=tf.random_normal_initializer())

    # ReLU
    # output_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W_1) + b_1), dropout_1)
    # Sigmoid
    output_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(x, W_1) + b_1), dropout_1)

    # Relu
    # output_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(output_1, W_2) + b_2), dropout_2)
    # Sigmoid
    output_2 = tf.nn.dropout(tf.matmul(output_1, W_2) + b_2, dropout_2)
    # without dropout for this layer
    # output_2 = tf.nn.relu(tf.matmul(output_1, W_2) + b_2)
    return output_2

def NN2(x):
    W_1 = tf.get_variable("W_1", [n_input, 1024], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_1 = tf.get_variable("b_1", [1024], dtype=tf.float32, initializer=tf.random_normal_initializer())


    W_2 = tf.get_variable("W_2", [1024, 512], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_2 = tf.get_variable("b_2", [512], dtype=tf.float32, initializer=tf.random_normal_initializer())

    W_3 = tf.get_variable("W_3", [512, 32], dtype=tf.float32, initializer=tf.random_normal_initializer())
    b_3 = tf.get_variable("b_3", [32], dtype=tf.float32, initializer=tf.random_normal_initializer())

    # W_4 = tf.get_variable("W_4", [1024, 32], dtype=tf.float32, initializer=tf.random_normal_initializer())
    # b_4 = tf.get_variable("b_4", [32], dtype=tf.float32, initializer=tf.random_normal_initializer())
    # ReLU
    output_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, W_1) + b_1), dropout_1)
    # Sigmoid
    # output_1 = tf.nn.dropout(tf.sigmoid(tf.matmul(x, W_1) + b_1), dropout_1)

    # Relu
    # output_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(output_1, W_2) + b_2), dropout_2)
    # Sigmoid
    output_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(output_1, W_2) + b_2), dropout_2)
    # without dropout for this layer
    # output_2 = tf.nn.relu(tf.matmul(output_1, W_2) + b_2)

    output_3 = tf.nn.dropout(tf.matmul(output_2, W_3) + b_3, dropout_2)
    return output_3


def main(argv):

    mfccInversion = MfccInversion()
    img2vec = image2vector([30,30,3])

    if download:
        file_name = "../data/video_list.txt"
        video_list_file = open(file_name, "r")
        video_list = video_list_file.readlines()
        print(len(video_list))
    else:
        directory = "../data/save/"
        filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
        files = []
        for filename in filenames:
            if filename.split(".")[1] == "mp4":
                files.append(filename)
        print(len(files))
    print("here")
    # exit()

    """define the model"""
    isRNN = False
    if isRNN:
        x = tf.placeholder("float", [None, n_steps, n_input])
        prediction = RNN2(x)
    else:
        x = tf.placeholder("float", [None, n_input])
        prediction = NN2(x)

    y = tf.placeholder("float", [None, n_classes])

    # Define loss and optimizer
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    loss = tf.losses.mean_squared_error(y, prediction)

    """l2 regularization"""
    lambda_l2_reg = 0.00005
    l2 = lambda_l2_reg * sum(
        tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables()
        if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
    )
    loss += l2

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()



    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, "../tmp/restore/model.ckpt")
        save_path = saver.save(sess, "../tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
        # exit()
        step = 0
        stop = False
        # Keep training until reach max iterations
        start_time = time.time()
        while step < training_iters:

            try:
                print("step" + str(step))
                if download:
                    iterator = load_generic_audio_video(directory, sample_rate, video_list, step, len(video_list))
                else:
                    if step % len(files) == 0:
                        shuffle(files)
                    file = files[step % len(files)]
                    print(file)
                    iterator = load_generic_from_saved_audio_video(directory, file, sample_rate, mfccInversion, img2vec)
                    # print(iterator)

                frame = 0
                if isRNN:
                    x_ = np.zeros((1, n_steps, n_input))
                else:
                    x_ = np.zeros((1, n_input))
                y_ = np.zeros((1, n_classes))
                # print("here")
                for mfcc_feat, imgVec in iterator:
                    # print(mfcc_feat.shape)
                    # print(imgVec.shape)
                    if isRNN:
                        x_[0, :n_steps-3, :] = x_[0, 3:,:]
                        x_[0, n_steps-3:,:] = mfcc_feat
                    else:
                        x_[0, :] = imgVec[0,0,0,:]
                    y_[0, :] = mfcc_feat[:,0]
                    # print("mfcc feat")
                    # print(mfcc_feat)
                    # print("img vec")
                    # print(imgVec)

                    # Run optimization op (backprop)
                    # print(y_)
                    sess.run(optimizer, feed_dict={x: x_, y: y_, dropout_1: 0.5, dropout_2: 0.5})
                    # print(y_)
                    # input('Press enter to continue: ')
                    if frame % display_step == 0:
                        # Calculate batch loss
                        # print("LOSS")
                        pred = sess.run(prediction, feed_dict={x: x_, y: y_, dropout_1: 1, dropout_2: 1})
                        print("Step " + str(step) + ", frame " + str(frame))
                        # print(pred.astype(int))
                        # print(y_)

                        cost = sess.run(loss, feed_dict={x: x_, y: y_, dropout_1: 1, dropout_2: 1})
                        print("Loss = " + str(cost))
                        # print(pred.astype(int))
                        # print("Step " + str(step) + ", frame " + str(frame) + ", Minibatch Loss= " + \
                        #       "{:.6f}".format(pred))
                    frame += 1
                step += 1
                save_path = saver.save(sess, "../tmp/model.ckpt")
            except:
                step += 1
                save_path = saver.save(sess, "../tmp/model.ckpt")
                continue
            print("Model saved in file: %s" % save_path)

        print("Optimization Finished!")
        end_time = time.time()
        print(end_time - start_time)
        print("sec")
        print("Testing Accuracy:", \
            sess.run(loss, feed_dict={x: x_ , y: y_, dropout_1: 1.0, dropout_2: 1.0}))


if __name__ == "__main__":
   main(sys.argv[1:])