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

# Augmentationが必要なラベル
aug_label1 = ["04_hikkakikizu_tate", "05_hikkakikizu_yoko", "06_jikizu_senjou", "08_mittyakukizu", "11_oshikomi_doushu", "18_oshikomihagare_koi",
              "24_senjouteishiato_kuroshimi", "25_senjouteishiato_line", "29_senjouteishiato_tatesuji", "30_senjouteishiato_yokosuzi", "36_tomozure_tate", "44_yousetsu_sonota"]
aug_label2 = ["07_kebaoshikomi", "09_mushi_fuyuu", "12_oshikomi_ishu",
              "14_oshikomi_sonota_senjou", "27_senjouteishiato_shiwa", "34_syoukizu", "41_yogore"]
aug_label3 = ["21_rollteishiato_midare"]
aug_label4 = ["35_tomozure_naname"]

# 水増しの設定
datagen1 = ImageDataGenerator(
    # 1234
    width_shift_range=0.1,  # 水平方向にランダムでシフト
    height_shift_range=0.1,  # 垂直方向にランダムでシフト
    horizontal_flip=True,  # 垂直方向にランダムで反転
    vertical_flip=True  # 水平方向にランダムで反転
)

datagen2 = ImageDataGenerator(
    # 12345
    rotation_range=90,  # ランダムに回転
    width_shift_range=0.1,  # 水平方向にランダムでシフト
    height_shift_range=0.1,  # 垂直方向にランダムでシフト
    horizontal_flip=True,  # 垂直方向にランダムで反転
    vertical_flip=True  # 水平方向にランダムで反転
)

datagen3 = ImageDataGenerator(
    # 123
    height_shift_range=0.1,  # 垂直方向にランダムでシフト
    horizontal_flip=True,  # 垂直方向にランダムで反転
    vertical_flip=True  # 水平方向にランダムで反転
)

datagen4 = ImageDataGenerator(
    # 34
    width_shift_range=0.1,  # 水平方向にランダムでシフト
    height_shift_range=0.1,  # 垂直方向にランダムでシフト
)

# 水増しが必要なラベルと設定をまとめる

aug_labels = [aug_label1, aug_label2, aug_label3, aug_label4]
datagens = [datagen1, datagen2, datagen3, datagen4]
aug_class = len(aug_labels)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('path', './data_reclass_1', """画像ファイルパス""")
tf.app.flags.DEFINE_integer('aug_num', 0, """水増し数""")
tf.app.flags.DEFINE_integer('photo_size', 224, """画素数""")
path = FLAGS.path
aug_num = FLAGS.aug_num
photo_size = FLAGS.photo_size

print("max_photo =")
val = input()
max_photo = int(val)
print("rate =")
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
            need / amounts_of_class_dict[label], 2)  # 重み＝（データ数/最大値）の逆数
        class_weights[label] = max(1.0, class_weights[label] * rate)
        label += 1
    return class_weights


def glob_files(path, label):  # データセット作成
    files = glob.glob(path + "/*.bmp")
    random.shuffle(files)
    # 各ファイルを処理
    num = 0
    for f in files:
        if num >= max_photo:
            break
        num += 1
        # 画像ファイルを読む
        img = cv2.imread(f)
        img = cv2.resize(img, (im_rows, im_cols))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img)
        x.append(img)
        y.append(label)

    print(num)


def images_gen(x_list, y_list, aug_label, datagen, aug_num):  # 水増し用関数
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
        batch = batch.astype(np.uint8)  # データ型を揃える！！
        label = label.astype(np.uint8)
        batch = batch.reshape((-1,) + in_shape)
        x_list = np.append(x_list, batch)
        y_list = np.append(y_list, label)
        x_list = x_list.reshape((-1,) + in_shape)
    print("finish augment " + labels[aug_label])
    return x_list, y_list


def print_cmx(y_true, y_pred, cv_count):
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

# 入力と出力を指定
im_rows = photo_size
im_cols = photo_size
im_color = 3
in_shape = (im_rows, im_cols, im_color)
nb_classes = len(labels)

outfile = path + "/photos_add_" + val + "_" + str(photo_size) + ".npz"


x = []  # 画像データ
y = []  # ラベルデータ
j = 0
for label in labels:
    glob_files(path + "/" + label, j)
    j += 1

# ファイルへ保存
np.savez(outfile, x=x, y=y)  # xとyがnumpyのリストとして与えられる
print("保存しました：" + outfile, len(x))

# 写真データを読み込み
photos = np.load(outfile)
x = photos["x"]
y = photos["y"]

# 読み込んだデータを三次元配列に変換
x = x.reshape(-1, im_rows, im_cols, im_color)
y = y.astype("int32")


# number for CV
# fix random seed for reproducibility
fold_num = 5
epochs = 25
seed = 7
batch_size = 32

# define X-fold cross validation
kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
cvscores = []

for p in range(aug_class):
    for i in range(len(labels)):
        for j in range(len(aug_labels[p])):
            if (labels[i] == aug_labels[p][j]):
                aug_labels[p][j] = i


# cnnモデルを取得

count = 1
for train, test in kfold.split(x, y):

    x_train = x[train]
    y_train = y[train]
    x_test = x[test]

    if (aug_num > 0):  # データ水増し
        for i in range(aug_class):
            for aug_label in aug_labels[i]:
                x_train, y_train = images_gen(
                    x_train, y_train, aug_label, datagens[i], aug_num)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = cnn_model.get_model3(in_shape, nb_classes)

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=40, verbose=1, min_delta=0, mode="auto")

    filepath = os.path.join("models", "model_" + str(photo_size) + "_aug_" + str(
        aug_num) + "_max_" + val + "_loss_{val_loss:.4f}_acc_{val_acc:.2f}.hdf5")

    modelCheckpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0,
                                      save_best_only=True, save_weights_only=True, mode='auto', period=1)

    # 学習を実行
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
