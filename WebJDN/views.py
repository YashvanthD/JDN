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
from .dataop import dataR,fdata,getdata
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset


"""DEEP LEARNING MODEL """


classifier=Sequential()

# step 1 : convolution

classifier.add(Convolution2D(64,3,3,input_shape=(256,256,3),activation='relu'))


#step 2 : max pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 : Flattern

classifier.add(Flatten())


#step 4 : Full Connection ANN

classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#image data generator

#from keras.preprocessing.image import ImageDatagenerator
flagClassifier=1


# from .form import ImageForm
# from .models import Image

# Create your views here.

"""

classifier=Sequential()

# step 1 : convolution

classifier.add(Convolution2D(64,3,3,input_shape=(256,256,3),activation='relu'))


#step 2 : max pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 : MaxPooling 

classifier.add(Flatten())


#step 4 : Full Connection ANN

classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#image data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
#from keras.preprocessing.image import ImageDatagenerator

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
        steps_per_epoch=12,
        epochs=10,
        validation_data=test_set,
        validation_steps=10)

train_set.class_indices




# from .form import ImageForm
# from .models import Image

# Create your views here.

"""


""" DEEP LEARNING MODEL EXECUTED SUCCESSFULLY """



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
            epochs=10,
            validation_data=test_set,
            validation_steps=3)

    train_set.class_indices
    return None

def WebJDN(request):
    return render(request,'home.html')

def train():
    if flagClassifier:
        traindataset()
        print("Done......")
    return None

def waiting(request):
    print("________________Waiting______________")
    
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
    



    if(eye_result=="Jaundice"):
        color='red'
    else:
        color='green'
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
        'hitchinng_skin':data.get('hitchingskin',False),
        'hitching_eye':data.get('hitchingeye',False),
        'yellowing_of_eye':eye_result,
        'yellowing_of_skin':data.get('yellowingofskin',False),
        'patient_id':id,
        'eye_img_id':id,
        'report_id':1,
        'color':color,
        'imgpath':eyepath,
        })

