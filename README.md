#How to enable smart advertisement
Open agegender_demo.py
Initialize an empty list in global section (ageList)
For running the web-cam for 6 seconds, do following- 

    t_end = time.time() + 6
    while time.time() < t_end:
        while time.time() < t_end:
            #Face Detection
            ret, frame = cap.read() #BGR
    
            #frame = cv2.imread("images/dress3.jpg")
    
            img=frame
            img = img[...,::-1]  #BGR 2 RGB
            inputs = img.copy() / 255.0
            
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_camera = cv2.resize(inputs, (416,416))
            img_camera = np.expand_dims(img_camera, axis=0)
            out2 = model_face.predict(img_camera)[0]
            results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
    
            #Age and Gender Detection
            
            show_results(img_cv,results, img.shape[1], img.shape[0], model_age, model_gender, model_emotion)
            print(ageList)
    
    a= sum(ageList)
    n= int(len(ageList)) 
    c= a/n
    if 15<c<40:
        startfile(r'C:\Data science\Tensorflow\YoloKerasFaceDetection\ADHM.mkv')


# Yolo Keras Face Detection

Implement Face detection, and Age and Gender Classification using Keras.

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/demo.jpg" width="50%" height="50%">
(image from wider face dataset)

# Overview

## Functions

Face Detection

Age and Gender Classification

## Requirements

Keras2 (Tensorflow backend)

OpenCV

Python 2.7

Darknet (for Training)

# Test

## Download Pretrained-Model

`python download_model.py`

## Predict from Camera Image

Here is a run using pretrained model .

`python agegender_demo.py`

# Train

## Install

### Keras

`pip install keras`

### Darknet

Download Darknet and put in the same folder.

https://github.com/pjreddie/darknet

## Face Detection (FDDB)

### Create dataset

Download fddb dataset (FDDB-folds and originalPics folder) and put in the dataset/fddb folder.

http://vis-www.cs.umass.edu/fddb/

Create dataset/fddb/FDDB-folds/annotations_darknet folder for darknet.

`python annotation_fddb_darknet.py`

Preview converted annotations.

`python annotation_view.py fddb`

[![FDDB dataset overview](https://img.youtube.com/vi/KGeY_PFhRYA/0.jpg)](https://www.youtube.com/watch?v=KGeY_PFhRYA&feature=youtu.be)

### Train using Darknet

Here is a training using YoloV2.

`cd darknet`

`./darknet detector train data/face-one-class.data cfg/yolov2-tiny-train-one-class.cfg`

### Test using Darknet

Here is a test.

`./darknet detector demo data/face-one-class.data cfg/yolov2-tiny-train-one-class.cfg backup-face/yolov2-tiny-train-one-class_32600.weights -c 0`

### Training Result

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/yolov2-tiny-train-one-class_32600.jpg" width="50%" height="50%">

<http://www.abars.biz/keras/yolov2-tiny-one-class.cfg>

<http://www.abars.biz/keras/yolov2-tiny-train-one-class_32600.weights>

### Convert to Keras Model

Download YAD2K

https://github.com/allanzelener/YAD2K

This is a convert script.

`python3 yad2k.py yolov2-tiny-train-one-class.cfg yolov2-tiny-train-one-class_32600.weights yolov2_tiny-face.h5`

This is a converted model.

<https://github.com/abars/YoloKerasFaceDetection/releases/download/1.10/yolov2_tiny-face.h5>

## Age and Gender classification

### Create Dataset

#### Use AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset

Download AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification dataset and put in the dataset/adience folder.

https://www.openu.ac.il/home/hassner/Adience/data.html#agegender

Create dataset/agegender_adience/annotations for keras.

`python annotation_agegender_adience_keras.py`

#### Use IMDB-WIKI dataset

Download IMDB-WIKI dataset (Download faces only 7gb) and put in the dataset/imdb_crop folder.

https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Create dataset/agegender_imdb/annotations for keras.

`python annotation_imdb_keras.py`

#### Use UTKFace dataset

Download UTKFace dataset and put in the dataset/imdb_crop folder.

https://susanqq.github.io/UTKFace/

Create dataset/agegender_utk/annotations for keras.

`python annotation_utkface_keras.py`

#### Use AppaReal dataset

Download AppaReal dataset and put in the dataset/appa-real-release folder.

http://chalearnlap.cvc.uab.es/dataset/26/description/

Create dataset/agegender_appareal/annotations for keras.

`python annotation_appareal_keras.py`

### Train using Keras

Install keras-squeezenet

https://github.com/rcmalli/keras-squeezenet

Run classifier task using keras.

`python agegender_train.py age101 squeezenet imdb`

`python agegender_train.py gender squeezenet imdb`

### Test using Keras

Test classifier task using keras.

`python agegender_predict.py age101 squeezenet imdb`

`python agegender_predict.py gender squeezenet imdb`

### Training result

Age101 (IMDB) (EPOCHS=100)

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/agegender_age101_squeezenet_imdb.png" width="50%" height="50%">

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/benchmark_age101_squeezenet_imdb.png" width="50%" height="50%">

<https://github.com/abars/YoloKerasFaceDetection/releases/download/1.10/agegender_age101_squeezenet_imdb.hdf5>

Gender (IMDB) (EPOCHS=25)

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/agegender_gender_squeezenet.png" width="50%" height="50%">

<img src="https://github.com/abars/YoloKerasFaceDetection/blob/master/pretrain/benchmark_gender_squeezenet_imdb.png" width="50%" height="50%">

<https://github.com/abars/YoloKerasFaceDetection/releases/download/1.10/agegender_gender_squeezenet_imdb.hdf5>

# Related Work

<https://github.com/dannyblueliu/YOLO-Face-detection>

<https://github.com/oarriaga/face_classification>

<https://github.com/yu4u/age-gender-estimation>

<https://github.com/abars/YoloKerasFaceDetection>

