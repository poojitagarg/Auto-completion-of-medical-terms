# upload image :endpoint
# save image
# function to predict image

# show result
import os
from flask import Flask
from flask import request
from flask import render_template
import numpy as np
import argparse
import imutils
import cv2
from imutils.contours import sort_contours
import matplotlib.image as mpimg
from keras.models import load_model
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from word_util import Prediction_n
# from char_util import ocr_pred
import imutils

app = Flask(__name__)
output1 = ""
pred_array = [0]
model_hand = None
model_3 = None
model_4 = None
model_5 = None
model_6 = None
model_7 = None
model_8 = None
models = []
UPLOAD_FOLDER = "/Users/mac/Desktop/Minor project/web_app/static"
alphabet = "abcdefghijklmnopqrstuvwxyz 0123456789,;.!?:'\"/\\|_@#%^&*~`+-=<>()[]{}"
char_len = len(alphabet)

# creating an inverse dictionary for decoding char's
inverse_dict = {}
char_dict = {}
inverse_dict[0] = '$'
char_dict['$'] = 0
for i, char in enumerate(alphabet):
    inverse_dict[i+1] = char
    char_dict[char] = i + 1

# one-hot encode single char into list


def one_hot(val, len=char_len+1):
    temp_list = []
    for var in val:
        temp = np.zeros(len, dtype=int)
        temp[int(var)] = 1
        temp_list.append(temp)
    return temp_list

# one hot encode single char into np array given char


def one_hot_char(val, len=char_len+1):
    temp = np.zeros(len, dtype=int)
    temp[char_dict[val]] = 1
    return np.array(temp)

# one hot encode single char into np array given position


def one_hot_value(pos, len=char_len+1):
    temp = np.zeros(len, dtype=int)
    temp[int(pos)] = 1
    return np.array(temp)


def Vectorize(word):
    word = word.lower()
    length = len(word)
    if(length < 3):
        print("Enter more letters")
        return np.zeros((1, 1))
    word_arr = []
    for i in word:
        word_arr.append(char_dict[i])
    word_vect = np.array(one_hot(word_arr))
    return np.reshape(word_vect, (1, word_vect.shape[0], word_vect.shape[1]))


def one_hot_output(vect):
    max = 0
    max_pos = -1
    eof = False
    for i, val in enumerate(np.reshape(vect, (vect.shape[1],))):
        if(val > 0.5):
            if(i == 0):
                eof = True
            return one_hot_value(i), eof
        elif(max > val):
            max = val
            max_pos = i
        if(max_pos == 0):
            eof = True
    return one_hot_value(max_pos), eof


def word_pred(word_vect, models):
    if(word_vect.shape[1] == 1):
        return np.zeros((1, 1))
    eof = False
    if(word_vect.shape[1] < 8):
        len_ = word_vect.shape[1]
        for i in range(len_, 8):
            next_word, eof = one_hot_output(models[i-3].predict(word_vect))
            word_vect = np.append(word_vect, np.reshape(
                next_word, (1, 1, char_len+1)), axis=1)
            if(eof):
                return word_vect
    while((not eof) and (word_vect.shape[1] < 50)):
        next_word, eof = one_hot_output(
            models[5].predict(word_vect[:, -8:, :]))
        word_vect = np.append(word_vect, np.reshape(
            next_word, (1, 1, char_len+1)), axis=1)
    return word_vect


def deencode(one_vect):
    for i, val in enumerate(one_vect):
        if(val == 1):
            return inverse_dict[i]


def decode(vect):
    if(vect.shape[1] == 1):
        return np.zeros((1, 1))
    word = ""
    for i in range(vect.shape[1]):
        word += deencode(np.reshape(vect[:, i, :], (vect.shape[2])))
    return word


def Prediction(word, models):
    # Convert original text to Vector by one hot encoding
    word_vect = Vectorize(word)

    # Predict the Output Vector using Deep Learning Models
    output_vect = word_pred(word_vect, models)

    # Convert the Output Vector to Human Redable Word
    actual_word = decode(output_vect)

    return actual_word


def one_hot_output_n(vect, n_pred_left):
    eof = []
    for i in range(n_pred_left):
        eof.append(False)
    word_val = []
    rem = n_pred_left
    temp = np.reshape(vect, (vect.shape[1],))
    index_list = np.argsort(temp)
    index_list = index_list.tolist()
    index_list.reverse()
    first = temp[index_list[0]]
    second = temp[index_list[1]]
    # print("value of first is",first,"and second is",second)
    for i in index_list:
        if(temp[i] > 0.7):
            if(i == 0):
                eof[0] = True
            word_val.append(one_hot_value(i))
            return word_val, eof
        elif ((rem != 0) and (first - temp[i] <= .3)):
            # print("this happened and value is", first-temp[i])
            word_val.append(one_hot_value(i))
            if(i == 0):
                eof[n_pred_left-rem] = True
            rem = rem-1
        else:
            break
    return word_val, eof


def get_possib(word_vect, models, n_pred):
    t_list = []
    if(word_vect.shape[1] == 1):
        return np.zeros((1, 1))
    first = True
    rem = 2
    if(word_vect.shape[1] < 8):
        len_ = word_vect.shape[1]
        for i in range(len_, 8):
            next_word_list, eof_list = one_hot_output_n(
                models[i-3].predict(word_vect), 2)
            if(first):
                t_list.append(np.append(word_vect, np.reshape(
                    next_word_list[0], (1, 1, char_len+1)), axis=1))
                first = False
            if(len(next_word_list) > 1 and rem > 0):
                t_list.append(np.append(word_vect, np.reshape(
                    next_word_list[1], (1, 1, char_len+1)), axis=1))
                # print("happened")
                rem = rem-1
            word_vect = np.append(word_vect, np.reshape(
                next_word_list[0], (1, 1, char_len+1)), axis=1)
            t_list[0] = word_vect
            if(eof_list[0]):
                # print("rem is :",rem)
                return t_list
    while((not eof_list[0]) and (word_vect.shape[1] < 50)):
        next_word_list, eof_list = one_hot_output_n(
            models[5].predict(word_vect[:, -8:, :]), 2)
        if(first):
            t_list.append(np.append(word_vect, np.reshape(
                next_word_list[0], (1, 1, char_len+1)), axis=1))
            first = False
        if(len(next_word_list) > 1 and rem > 0):
            t_list.append(np.append(word_vect, np.reshape(
                next_word_list[1], (1, 1, char_len+1)), axis=1))
            rem = rem-1
            # print("is something wrong here")
        word_vect = np.append(word_vect, np.reshape(
            next_word_list[0], (1, 1, char_len+1)), axis=1)
        t_list[0] = word_vect
    # print("here rem is :",rem)
    return t_list


def word_pred_n(vect, models, n_pred):
    temp_list = get_possib(vect, models, n_pred)
    # temp_list = np.array(temp_list)
    # print(temp_list.shape)
    words = []
    words.append(temp_list[0])
    # print(len(temp_list))
    for i in range(1, len(temp_list)):
        words.append(word_pred(temp_list[i], models))
        # words.append(word_pred(temp_list[2]))
    return words


# PREDICTION FOR MULTIPLE OUTPUTS
def Prediction_n(word, models, n_pred=3):

    if(n_pred < 1):
        # print("Error...Please enter the correct number of predictions")
        return []

    # Convert original text to Vector by one hot encoding
    word_vect = Vectorize(word)

    # Predict the Output Vector using Deep Learning Models

    output_vect = word_pred_n(word_vect, models, n_pred)

    # Convert all the Output Vectors to Human Redable Words
    actual_words = []
    for i in range(len(output_vect)):
        actual_words.append(decode(output_vect[i]))

    return actual_words


def ocr_pred(image_path, model):

    # load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # perform edge detection, find contours in the edge map, and sort the
    # resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # initialize the list of contour bounding boxes and associated
    # characters that we'll be OCR'ing
    chars = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)

            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)

    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    output = ""
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        # if probability of charachter is greater than 80% we add it to original word
        if(pred[i] > .80):
            output = output + labelNames[i]

    return output


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )

            image_file.save(image_location)
            print(image_location)

            output1 = ocr_pred(image_location, model_hand)
            print(output1)
            pred_array = []
            p = Prediction_n(output1, models)
            x = len(p)
            p = [x[:-1] for x in p]
            for i in range(len(p)):
                pred_array.append(p[i])
            if len(pred_array) == 2:
                pred_array.append("")
            if len(pred_array) == 1:
                pred_array.append("")
                pred_array.append("")
            if len(pred_array) == 0:
                pred_array.append("")
                pred_array.append("")
                pred_array.append("")

            print(pred_array)

            return render_template("index.html", prediction1=pred_array[0], prediction2=pred_array[1], prediction3=pred_array[2], image_loc=image_file.filename, x=x)

    return render_template("index.html", prediction1="", prediction2="", prediction3="", image_loc=None, x=0)


if __name__ == "__main__":
    model_hand = load_model('./handwriting.model')
    model_3 = load_model('./model_3word.h5')
    model_4 = load_model('./model_4word.h5')
    model_5 = load_model('./model_5word.h5')
    model_6 = load_model('./model_6word.h5')
    model_7 = load_model('./model_7word.h5')
    model_8 = load_model('./model_8word.h5')
    models.append(model_3)
    models.append(model_4)
    models.append(model_5)
    models.append(model_6)
    models.append(model_7)
    models.append(model_8)

    app.run(port=1200, debug=True)
