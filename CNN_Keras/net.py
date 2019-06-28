from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class Net:
    @staticmethod
    def build(width,height,depth,classes):
        #Init model
        model = Sequential()
        inputShape = (height,width,depth)

        #Updateinput shape if we use channels_first
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)

        #1st set
        model.add(Conv2D(20,(5,5),padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #2nd set
        model.add(Conv2D(20,(5,5),padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #3rd set
        model.add(Conv2D(20,(5,5),padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #4th set
        model.add(Conv2D(20,(5,5),padding="same",input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #FC set
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        #Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

