import numpy as np
import keras
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
import tensorflow as tf
import sys
import glob
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
import cnn_model
import json
import pickle
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

# Labels requiring Augmentation
aug_label1 = ["a", "b"]
aug_label2 = ["c", "d"]

# Augmentation settings
datagen1 = ImageDataGenerator(
    rotation_range=90,  # Rotate randomly
    width_shift_range=0.1,  # Shift randomly horizontally
    height_shift_range=0.1,  # Randomly shift vertically
    horizontal_flip=True,  # Randomly flip horizontally
    vertical_flip=True  # Randomly flip vertically
)

datagen2 = ImageDataGenerator(
    width_shift_range=0.1,  # Shift randomly horizontally
    height_shift_range=0.1,  # Randomly shift vertically
    horizontal_flip=True,  # Randomly flip horizontally
    vertical_flip=True  # Randomly flip vertically
)

# 水増しが必要なラベルと設定をまとめる

aug_labels = [aug_label1, aug_label2]
datagens = [datagen1, datagen2]
aug_class = len(aug_labels)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('path', './data', """Image file path""")
tf.app.flags.DEFINE_integer('aug_num', 0, """Augumentation quantity""")
tf.app.flags.DEFINE_integer('photo_size', 224, """Pixel number""")
path = FLAGS.path
aug_num = FLAGS.aug_num
photo_size = FLAGS.photo_size

print("max_photo =") # 
val = input()
max_photo = int(val)
print("rate =") #
rate = float(input())


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def cal_weight(class_name_list, train):  # 重み付け関数
    amounts_of_class_dict = []
    label = 0
    class_weights = {}
    need = max_photo * (fold_num - 1) / fold_num
    for class_name in class_name_list:
        amounts_of_class_dict.append(
            np.count_nonzero(train == label))
        class_weights[label] = round(
            need / amounts_of_class_dict[label], 2)  # weight ＝(maxiam number/data quantity)
        class_weights[label] = max(1.0, class_weights[label] * rate)
        label += 1
    return class_weights


def glob_files(path, label):  # Making dataset
    files = glob.glob(path + "/*.jpg")
    random.shuffle(files)
    num = 0
    for f in files:
        if num >= max_photo:
            break
        num += 1
        # Reading image files
        img = cv2.imread(f)
        img = cv2.resize(img, (im_rows, im_cols))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        x.append(img)
        y.append(label)

    print(num)


def images_gen(x_list, y_list, aug_label, datagen, aug_num):  # Augumantation function
    batch_size = 32
    count = np.count_nonzero(y_list == aug_label)
    need = max_photo * (fold_num - 1) / fold_num
    aug_num_tmp = aug_num
    x_list_need = np.array([])
    y_list_need = np.array([])
    if ((count * aug_num) > need):
        aug_num_tmp = round(need / count)
    aug_num_tmp -= 1
    if (batch_size < count):
        aug_num_tmp = aug_num_tmp * math.ceil(count / batch_size)
    for x, y in zip(x_list, y_list):
        p = 0
        if (y == aug_label):
            p += 1
            x_list_need = np.append(x_list_need, x)
            y_list_need = np.append(y_list_need, y)
            x_list_need = x_list_need.reshape((-1,) + in_shape)
            if (p == count):
                break
    i = 0
    for batch, label in datagen.flow(x_list_need, y=y_list_need, batch_size=batch_size):
        if (i > (aug_num_tmp - 1)):
            break
        i += 1
        batch = batch.astype(np.uint8)
        label = label.astype(np.uint8)
        batch = batch.reshape((-1,) + in_shape)
        x_list = np.append(x_list, batch)
        y_list = np.append(y_list, label)
        x_list = x_list.reshape((-1,) + in_shape)
    print("finish augment " + labels[aug_label])
    return x_list, y_list


def print_cmx(y_true, y_pred, cv_count): # Making confusion matrix function
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    df_cmx.to_csv("./matrix_" + str(cv_count) + ".csv")


def make_label(path):
    files = os.listdir(path)
    label = [f for f in files if os.path.isdir(os.path.join(path, f))]
    label.sort()
    return label


labels = make_label(path)

# Specify input and output
im_rows = photo_size
im_cols = photo_size
im_color = 3
in_shape = (im_rows, im_cols, im_color)
nb_classes = len(labels)

outfile = path + "/photos_add_" + val + "_" + str(photo_size) + ".npz"


x = []  # Image data
y = []  # Label data
j = 0
for label in labels:
    glob_files(path + "/" + label, j)
    j += 1

# Save to file
np.savez(outfile, x=x, y=y)
print("保存しました：" + outfile, len(x))

# Reading image data
photos = np.load(outfile)
x = photos["x"]
y = photos["y"]

# Convert read data to 3D array
x = x.reshape(-1, im_rows, im_cols, im_color)
y = y.astype("int32")


# Number for CV
# Fix random seed for reproducibility
fold_num = 5
epochs = 25
seed = 7
batch_size = 32

# Define X-fold cross validation
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
cvscores = []

for p in range(aug_class):
    for i in range(len(labels)):
        for j in range(len(aug_labels[p])):
            if (labels[i] == aug_labels[p][j]):
                aug_labels[p][j] = i


count = 1

# Cross validation
for train, test in kfold.split(x, y):

    x_train = x[train]
    y_train = y[train]
    x_test = x[test]

    if (aug_num > 0):  # Data Augumentation
        for i in range(aug_class):
            for aug_label in aug_labels[i]:
                x_train, y_train = images_gen(
                    x_train, y_train, aug_label, datagens[i], aug_num)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Get cnn_models

    model = cnn_model.get_model3(in_shape, nb_classes)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=40, verbose=1, min_delta=0, mode="auto")

    filepath = os.path.join("models", "model_" + str(photo_size) + "_aug_" + str(
        aug_num) + "_max_" + val + "_loss_{val_loss:.4f}_acc_{val_acc:.2f}.hdf5")

    modelCheckpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0,
                                      save_best_only=True, save_weights_only=True, mode='auto', period=1)

    # Training
    hist = model.fit(x_train, keras.utils.np_utils.to_categorical(y_train, nb_classes),
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(
        x_test, keras.utils.np_utils.to_categorical(y[test], nb_classes)),
        class_weight=cal_weight(labels, y_train),
        callbacks=[early_stopping, modelCheckpoint]
    )

    predict_classes = model.predict(x_test, batch_size=batch_size)
    predict_classes = np.argmax(predict_classes, axis=1)
    true_classes = np.argmax(
        keras.utils.np_utils.to_categorical(y[test], nb_classes), 1)

    # Evaluate
    scores = model.evaluate(x_test, keras.utils.np_utils.to_categorical(
        y[test], nb_classes), verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    print_cmx(true_classes, predict_classes, count)

    count += 1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
sys.exit()
