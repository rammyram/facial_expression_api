# -*- coding: utf-8 -*-
"""
@author: Rammy Ram 
"""
import os 
import sys
import cv2
import numpy as np
import pandas  as pd

import matplotlib.pyplot as plt 
#% matplotlib inline
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


from keras.models import model_from_json
import numpy as np 
from flask import Flask, render_template, Response , request

facec = cv2.CascadeClassifier('opencv/haarcascade_frontalface_alt2.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
emotions_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class Facial_exp_model(object):
    '''
     Class to give us the predictions :
    '''
    
    def __init__(self, model_json , model_weights):
        # load json model :
        with open(model_json , 'r') as json_file:
            self.loaded_model = model_from_json( json_file.read() )
  
        # Load wts in the new model :
        self.loaded_model.load_weights( model_weights )
        self.loaded_model._make_predict_function()
  
    def predict_emotion(self, img):
      self.prediction = self.loaded_model.predict(img)	  
      sorted_args = np.argsort(self.prediction)
      prediction_1 = sorted_args[0][-1]
      prediction_2 = sorted_args[0][-2]
      
      print('{} - {}'.format( emotions_list[ prediction_1 ],emotions_list[ prediction_2 ] ))
      return '{} - {}'.format( emotions_list[ prediction_1 ],emotions_list[ prediction_2 ] )
  
    
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = roi.astype('float32')/255
            roi = np.asarray(roi)
            pred = loaded_model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


def detect_image(addr,result_path):    
    font = cv2.FONT_HERSHEY_SIMPLEX
    im = cv2.imread(addr)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray,scaleFactor=1.2)
    #print(faces)
    for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2,5)
            face_crop = im[y:y+h,x:x+w]
            #cv2_imshow( face_crop)
            
            face_crop = cv2.resize(face_crop,(48,48))            
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop = face_crop.astype('float32')/255
            face_crop = np.asarray(face_crop)
            face_crop = face_crop.reshape(1, face_crop.shape[0],face_crop.shape[1] , 1 )
            result = loaded_model.predict_emotion(face_crop)
            cv2.putText(im,result,(x,y), font, 1, (200,0,0), 3, cv2.LINE_AA)
            
            
    #im = cv2.resize( im , (500,500) )
            
    #cv2.imshow(result_path , im)  
    #cv2.waitKey(0)
    if not cv2.imwrite(result_path,im):
        raise Exception("Could not write image")
    cv2.destroyAllWindows()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Root path :
root =  os.getcwd()
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__ , template_folder= root + r'\Templates')
app.secret_key = 'BhargavVadlamudi'
app.static_url_path= r'\static'
loaded_model = Facial_exp_model( 'checkpoint/new/model_new.json' , 'checkpoint/new/model_new_wts.h5'  )
#loaded_model = Facial_exp_model( 'checkpoint/model.json' , 'checkpoint/model_wts.h5'  )

@app.route('/')
def index():
    return render_template( 'index.html' )

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/cam')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
					
@app.route('/img_upload', methods = ['GET' , 'POST'])
def img_upload():    
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:            
                return ('No file part')
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename            
            if file.filename == '':
                return ('No selected file')            
            if file and allowed_file(file.filename):
                UPLOAD_FOLDER = 'TestData'
                img_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(img_path)
                result_path = r"static\{}".format(file.filename) 
                print(result_path)
                detect_image(img_path,result_path)
                return '<img src="'+ result_path +'" alt="Smiley face" height="700" width="700">'
        return render_template('upload.html')
    
    except Exception as e:
        if hasattr(e, 'message'):
            return(e.message)
        else:
            return('{}'.format(e))
            
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)