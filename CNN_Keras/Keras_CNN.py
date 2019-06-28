#Set the matplotlib so figures can be saved
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from net import Net
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random 
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 100
INIT_LR = 0.001
BATCH_SIZE = 32

print("[INFO] Loading data...")
data = []
temp= []
labels = []
label = []

with open("label.txt", "r") as text_file:
    temps = text_file.readline()
    temps = temps.rstrip('\n')
labels = temps.split(",")

print(labels)

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(30)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    #Preprocess images
    image = cv2.imread(imagePath)
    image = cv2.resize(image,(28,28))
    image = img_to_array(image)
    data.append(image)

    #Get the class label
    name = imagePath.split(os.path.sep)[-2]
    temp = labels.index(name)
    label.append(temp)

data = np.array(data, dtype = "float") / 255.0
label = np.array(label)
trainX, testX, trainY, testY = train_test_split(data, label, test_size=0.25, random_state=30)

classes = len(labels)
trainY = to_categorical(trainY, num_classes = classes)
testY = to_categorical(testY, num_classes = classes)

aug = ImageDataGenerator(rotation_range = 30, width_shift_range=0.1,height_shift_range=0.1,shear_range = 0.2, zoom_range=0.2, horizontal_flip=True,fill_mode="nearest")

print("[INFO] Compiling model...")
model = Net.build(width=28,height=28, depth=3, classes = classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

print("[INFO] Training model...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = BATCH_SIZE),validation_data =(testX,testY), steps_per_epoch = len(trainX) // BATCH_SIZE, epochs = EPOCHS, verbose =1)

model.save(args["model"])
print("[INFO] Model Saved...")
