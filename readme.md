# Handwritten Chinese Character Recognition

hccr using tensorflow.The model likes VGG net.

DataBase: HWDB1.1. u can get it from http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html

## dependencies

* Python
* Tensorflow
* OpenCV
* ...

## how to use

* download the database
* run unpack_data.py, get the pics
* run make_tf_data.py, get the tf data for training
* run train, get the model

## heroku application

i make it a heroku application. demo website:  https://shielded-atoll-68504.herokuapp.com/

![website](https://raw.githubusercontent.com/zealerww/HCCR/master/demo_pic/demo.png)

you can get more information about heroku from my another project: https://github.com/zealerww/digits_recognition





