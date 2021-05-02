#!/usr/bin/python3
# -*- coding: utf-8 -*-

### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt
from speechFunctions import *
import keras
from keras.models import Model

### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response



# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

################################################################################
################################## INDEX #######################################
################################################################################

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

################################################################################
################################## RULES #######################################
################################################################################

# Rules of the game
@app.route('/rules')
def rules():
    return render_template('rules.html')

################################################################################
############################### VIDEO INTERVIEW ################################
################################################################################
# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('C:/Users/lenovo/Desktop/PFA/Multimodal-Emotion-Recognition/04-WebApp/static/js/db/histo.txt', sep=",")

# Video interview template

@app.route('/video', methods=['POST'])
def video() :
    # Display a warning message
    flash('You will have 45 seconds to discuss the topic mentioned above. Due to restrictions, we are not able to redirect you once the video is over. Please move your URL to /video_dash instead of /video_1 once over. You will be able to see your results then.')
    return render_template('video.html')



################################################################################
############################### AUDIO INTERVIEW ################################
################################################################################

# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():

    # Flash message
    flash("After pressing the button above, you will have 15sec to answer the question.")
    
    return render_template('audio.html', display_button=False)

# Audio Recording
@app.route('/audio_recording', methods=("POST", "GET"))
def audio_recording():

    # Instanciate new SpeechEmotionRecognition object
    
    

    # Voice Recording
    rec_duration = 16 # in sec
    rec_sub_dir = os.path.join('C:/Users/lenovo/Desktop/PFA/Multimodal-Emotion-Recognition/04-WebApp/tmp','voice_recording.wav')
    voice_recording(rec_sub_dir, duration=rec_duration)
    
    # Send Flash message
    flash("The recording is over! You now have the opportunity to do an analysis of your emotions. If you wish, you can also choose to record yourself again.")

    return render_template('audio.html', display_button=True)


# Audio Emotion Analysis
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():

    # Sub dir to speech emotion recognition model
    #model_sub_dir = os.path.join('Models', 'audio.hdf5')
    loaded_model = keras.models.load_model('models/Emotion_Voice_Detection_Model.h5')
    emotion = make_predictions(file='C:/Users/lenovo/Desktop/PFA/Multimodal-Emotion-Recognition/04-WebApp/tmp/voice_recording.wav', loaded_model = loaded_model)
    #return render_template('audio_dash.html', emo=emotion, emo_other=0, prob=0, prob_other=0)
    
    # Send Flash message
    flash("you are "+emotion)
    return render_template('audio.html', display_button=True)



################################################################################
############################### TEXT INTERVIEW #################################
################################################################################


@app.route('/text', methods=['POST'])
def text() :
    return render_template('text.html')


if __name__ == '__main__':
    app.run(debug=True)
