from django.shortcuts import render,redirect
from django.http import *
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from time import sleep
import csv
from .dataop import dataR,fdata,getdata,score
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
epoch=10
epoch2=2
"""DEEP LEARNING MODEL """


classifier=Sequential()

# step 1 : convolution

classifier.add(Convolution2D(64,3,3,input_shape=(256,256,3),activation='relu'))


#step 2 : max pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 : Flattern

classifier.add(Flatten())

global turl
turl=0

#step 4 : Full Connection ANN

classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#image data generator

#from keras.preprocessing.image import ImageDatagenerator
flagClassifier=1


eyeclassifier=Sequential()

#step 1 : convolution

eyeclassifier.add(Convolution2D(64,3,3,input_shape=(256,256,3),activation='relu'))


#step 2 : max pooling

eyeclassifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 : MaxPooling 

eyeclassifier.add(Flatten())


#step 4 : Full Connection ANN

eyeclassifier.add(Dense(128,activation='relu'))
eyeclassifier.add(Dense(1,activation='sigmoid'))

eyeclassifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#image data generator

# from .form import ImageForm
# from .models import Image

# Create your views here.

"""DEEP LEARNING MODEL EXECUTED SUCCESSFULLY """





def traindataset():
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    #TRAIN_STEPS_PER_EPOCH = np.ceil((image_count*0.8/BATCH_SIZE)-1)
    # to ensure that there are enough images for training bahch
    #VAL_STEPS_PER_EPOCH = np.ceil((image_count*0.2/BATCH_SIZE)-1)
    
    train_set = train_datagen.flow_from_directory(
            './media/data/4/train',
            target_size=(256, 256),
            batch_size=32,
            class_mode='binary')

    test_set = test_datagen.flow_from_directory(
            './media/data/4/test',
            target_size=(256, 256),
            batch_size=32,
            class_mode='binary')

    classifier.fit(
            train_set,
            steps_per_epoch=15,
            epochs=epoch,
            validation_data=test_set,
            validation_steps=3)

    print(train_set.class_indices)

        
    train_datagen2 = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen2 = ImageDataGenerator(rescale=1./255)

    #TRAIN_STEPS_PER_EPOCH = np.ceil((image_count*0.8/BATCH_SIZE)-1)
    # to ensure that there are enough images for training bahch
    #VAL_STEPS_PER_EPOCH = np.ceil((image_count*0.2/BATCH_SIZE)-1)

    train_set2 = train_datagen2.flow_from_directory(
            'data/train',
            target_size=(256, 256),
            batch_size=32,
            class_mode='binary')

    test_set2 = test_datagen2.flow_from_directory(
            './data/test',
            target_size=(256, 256),
            batch_size=32,
            class_mode='binary')

    eyeclassifier.fit(
            train_set2,
            steps_per_epoch=200,
            epochs=epoch2,
            validation_data=test_set2,
            validation_steps=100)

    print(print(train_set2.class_indices))
    return None

def WebJDN(request):
    global turl,train1,train2
    if(turl==1):
        turlc='green'
        train1='Done '
        train2=' '
    else:
        turlc='red'
        train1='Not Yet!'
        train2='please train befor analyze'
    return render(request,'home.html',{
    'train1':train1,
    'train2':train2,
    'trainc':turlc,
    })

def about(request):
    return render(request,'about.html')

def train():
    if flagClassifier:
        traindataset()
        print("Done......")
    return None

def check(request):
    return render(request,'error.html')

def waiting(request):
    print("________________Waiting______________")
    global turl
    time=epoch*25
    print(turl)
    if(turl==1):
        return redirect('tcheck')
    elif(turl==0):
        turl=1
        return render(request,'wait.html')

def AddPatient(request):
    print("________________AddPatient______________")
    return render(request,'PatientDetails.html')

def done(request):
    print("________________Done______________")
    train()
    return render(request,'done.html')

def Predict_eye_img(ad):
    
    sample=image.load_img(ad,target_size=(256,256))
    #sample.show()
    sample=image.img_to_array(sample)
    sample=np.expand_dims(sample,axis=0)

    r=eyeclassifier.predict(sample)
    if r[0][0]==1:
        print("Not a eye Image")
        return 0


    result=classifier.predict(sample)

    if result[0][0]==1:
        predection="Jaundice"
    else:
        predection="Normal"

    print(predection)
    return predection

def error(request):
    return render(request,'error.html')

def Add(request):
    # if(flagClassifier):
    #     return redirect('error')
    if request.method=="POST":
        data=request.POST
        fs=request.FILES
    

    # print(data,"\n\n")
    # print(data.values,"\n\n")
    # print(data.keys,"\n\n")
    # data=data.values()
    # yield from data
    # print(data)
    id=fdata(fs)
    dataR(data)
    # l=fs.get_all_values()
    # print(l)
    print("ID_____________",id)
    eyepath='./media/img/'+id+'.png'
    eye_result=Predict_eye_img(eyepath)
    points=score(data)
    print(points)
    if(eye_result==0):
        return render(request,'imgerror.html')
    tt=points['total']

    if(points['type']=='infant'):
        tt=points['max']

    if(points['ye']==0 and eye_result=="Jandice"):
        tt=tt+20
    if(tt>40):
        sresult="True"
    else: 
        sresult="False"

    if(eye_result=="Jaundice"):
        color='red'
        yresult='True'
    else:
        color='green'
        yresult='False'

    if((tt>47 and eye_result=="Jaundice") or (tt>19 and eye_result=="Jaundice" and points['max']==0) ):
        fresult="Jaundice"
    else:
        fresult="Normal"
    if(tt>50 and eye_result=="Normal"):
        fresult="Sorry... Some thing Went Wrong! Please Re-test with valid information and good Image"


    global turl
    if(turl==1):
        turlc='green'
    else:
        turlc='red'

    if(fresult=="Normal"):
        color='green'
    else:
        color='red'


    print(data.get('dob',False))
    

    
    return render (request,'report.html',{
        'name':data.get('name',False),
        'type':data.get('type',False),
        'father_name':data.get('fname',False),
        'mother_name':data.get('mname',False),
        'age':data.get('age',False),
        'gender':data.get('gender',False),
        'dob':data.get('dob',False),
        'address':data.get('address',False),
        'pincode':data.get('pincode',False),

        'fever':data.get('fever',False),
        'cough':data.get('cough',False),
        'headheach':data.get('headache',False),
        'loss_of_hepatities':data.get('loss_of_hepatities',False),
        'diarrhea':data.get('diarrhea',False),
        'abdominal_pain':data.get('abdominal_pain',False),

        'hitchinng_skin':data.get('hitchingskin',False),
        'hitching_eye':data.get('hitchingeye',False),

        'yellowing_of_eye':eye_result,
        'yellowing_of_skin':data.get('yellowingskin',False),
        'yellowing_of_nails':data.get('yellowingnails',False),
        'yellowing_of_palm':data.get('yellowingpalm',False),


        'yellowing_of_eye_r':yresult,
        'score':tt,
        'fresult':fresult,
        'sresult':sresult,

        'patient_id':id,
        'eye_img_id':id,
        'report_id':1,
        'color':color,
        'imgpath':eyepath,
        'trainc':turlc,
        'train':turl,
        })

