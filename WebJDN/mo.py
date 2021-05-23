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
