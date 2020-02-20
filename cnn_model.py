import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Activation
from keras.layers import Conv2D, MaxPooling2D, Multiply, BatchNormalization
from keras.optimizers import RMSprop
import os
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.activations import linear
from resnet50 import ResNet50_2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
vgg16 = VGG16(include_top=False, input_shape=(224, 224, 3))
resnet50 = ResNet50(include_top=False, weights='imagenet',
                    input_shape=(224, 224, 3))
vgg16_2 = VGG16(include_top=False, weights='imagenet',
                input_shape=(224, 224, 3))

#Squeeze and  Excitation

def se_block(input, channels, r=8):
    #Squeeze
    x = GlobalAveragePooling2D()(input)
    #Excitation
    x = Dense(channels//r, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)
    return Multiply()([input, x])

# cnnのモデルを定義する

def def_model(in_shape, nb_classes):
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation="relu",
                     input_shape=in_shape))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation="relu"))
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation="relu",
                     name="relu_conv2")
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=nb_classes, activation="softmax"))

    return model

# vgg16を利用するモデル


def build_transfer_model(vgg16, nb_classes):
    model = Sequential(vgg16.layers)
    for layer in model.layers[:14]:
        layer.trainable = False
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=nb_classes, activation='softmax'))
    return model

#Resnet50を使用するモデル

def build_transfer_model2(resnet50, nb_classes):
    x = resnet50.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=resnet50.input, outputs=predictions)
    return model

def build_transfer_model3(resnet50, nb_classes):
    x = resnet50.layers[0].input
    for i, layer in enumerate(resnet50.layers):
#        print(layer.name)
        if i == 0: continue
        x = layer(x)
        if ("res" in layer.name) and ("branch2" in layer.name):
            print("add " + layer.name) 
#            x = se_block(x, layer.filters)
#        elif (layer.name == "conv1"):
            x = se_block(x, layer.filters)
#        else:
#            x = layer(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation="softmax")(x)
    model = Model(resnet50.input, x)
    return model

def build_transfer_model4(vgg16, nb_classes):
    x = vgg16.layers[0].input
    for i, layer in enumerate(vgg16.layers):
        if i == 0: continue
        if "conv" in layer.name:
            x = layer(x)
            x = se_block(x, layer.filters)
        else:
            x = layer(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation="softmax")(x)
    model = Model(vgg16.input, x)
    return model

def build_transfer_model5(resnet50, nb_classes):
    x = resnet50
    for i, layer in enumerate(x.layers):
        print(layer.name)
        if i == 0: continue
        if ("res" in layer.name):
             se_block(layer, layer.filters)
#            x = se_block(x, layer.filters)
        elif (layer.name == "conv1"):
             se_block(layer, layer.filters)
    x = x.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation="softmax")(x)
    model = Model(resnet50.input, x)
    return model

# コンパイル済みのcnnのモデルを返す


def get_model(in_shape, nb_classes):
    model = def_model(in_shape, nb_classes)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=RMSprop(),
        metrics=["accuracy"])
    return model


def get_model2(in_shape, nb_classes):
    model = build_transfer_model(vgg16, nb_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
    return model


def get_model3(in_shape, nb_classes):
    model = build_transfer_model2(resnet50, nb_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
    return model

def get_model4(in_shape, nb_classes):
    model = build_transfer_model4(vgg16, nb_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
    return model

def get_model5(in_shape, nb_classes):
    model = ResNet50_2(in_shape, nb_classes)
     model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
    return model




ResNet50_2().summary()
#build_transfer_model3(resnet50, 10).summary()
#vgg16.summary()
#build_transfer_model4(vgg16, 10).summary()