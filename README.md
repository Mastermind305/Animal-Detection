# Animal Detection Using Deep Learning


## Overview

Welcome to the Animal Detection project. This is the deep learning boject detection model that was trained to detect various stray and wild animals that we might encounter. I have fine-tuned MobileNet object detection model provided by "TensorFlow Object Detection API" to detect different animals.

Video from various sources can be inferenced using this Object Detection model. The model expects images and videos in the format **[ 1, height, weidth, channel ]**. This code based is designed for video processing, and if you want to use it to inference image data you need to make changes yourself as of your need.

## Table of Contents

- **[data](data):**
    <p>This folder contains, test data for the model.</p>

-  **[mobilenet](mobilenet):**

    <p>This folder contains all the model weights and fingerprints for it to perform inference on the data.</p>

- **detection.py:**

    <p>contains code for loading model, preprocessing data, running inference and displaying the result.</p>

- **main.py:**

    <p>contains code for loading video, calls various functions in detection to perform the detection.</p>

- **requirements.txt:**

    <p>this file contains all reqirements that this projects needs to run successfully.</p>

## Installation

To run this project in your computer clone this repository perform following tasks: 

1. Start a python virtual environment using command 
<br> For Linux <br>
<code> python3 -m venv venv </code>
<br> For Windows <br>
<code> python -m venv venv </code>
<br>
You can also use anaconda too create the virtual environmment.<br>

2. Navigate to the repository and start the environment. After initializing, run following command<br>
<code> pip install -r requirements.txt </code>

3. Once the installition of requirements is cou complete you can run it in your computer. Initially i have provided test video. To test it you can just run the following command:<br>
<code> python main.py </code>
<br>
This should get you going

Note: If you want to run inference on your own data then you need to change file path in **main.py** line **8**.

## Features

It is able to detect and classify various wild domestic and wild animals with high level of accuracy. It has following classes of animals:

1. boar
2. buffalo
3. cow/bull
4. dog
5. elephant
6. leopard
7. monkey
8. snake
9. tiger


