import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

alphabet = "abcdefghijklmnopqrstuvwxyz 0123456789,;.!?:'\"/\\|_@#%^&*~`+-=<>()[]{}"
char_len = len(alphabet)
# creating an inverse dictionary for decoding char's
def create_inverse_dict():

  inverse_dict = {}
  inverse_dict[0] = '$'
  for i, char in enumerate(alphabet):
      inverse_dict[i+1] = char
  return inverse_dict

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
          word_vect = np.append(word_vect, np.reshape(next_word, (1, 1,char_len+1)),axis=1)
          if(eof):
              return word_vect
  while((not eof) and (word_vect.shape[1] < 50)):
      next_word, eof = one_hot_output(models[5].predict(word_vect[:, -8:, :]))
      word_vect = np.append(word_vect, np.reshape(next_word, (1, 1,char_len+1)),axis=1)
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
              t_list.append(np.append(word_vect, np.reshape(next_word_list[0], (1, 1,char_len+1)),axis=1))
              first = False
          if(len(next_word_list) > 1 and rem >0):
              t_list.append(np.append(word_vect, np.reshape(next_word_list[1], (1, 1,char_len+1)),axis=1))
              # print("happened")
              rem = rem-1
          word_vect = np.append(word_vect, np.reshape(next_word_list[0], (1, 1,char_len+1)),axis=1)
          t_list[0] = word_vect
          if(eof_list[0]):
              # print("rem is :",rem)
              return t_list
  while((not eof_list[0]) and (word_vect.shape[1] < 50)):
      next_word_list, eof_list = one_hot_output_n(models[5].predict(word_vect[:, -8:, :]),2)
      if(first):
          t_list.append(np.append(word_vect, np.reshape(next_word_list[0], (1, 1,char_len+1)),axis=1))
          first = False
      if(len(next_word_list) > 1 and rem >0):
          t_list.append(np.append(word_vect, np.reshape(next_word_list[1], (1, 1,char_len+1)),axis=1))
          rem = rem-1
          # print("is something wrong here")
      word_vect = np.append(word_vect, np.reshape(next_word_list[0], (1, 1,char_len+1)),axis=1)
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
