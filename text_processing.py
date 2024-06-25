import nltk
import pandas as pd
import numpy as np
import torch

nltk.download('stopwords')

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords

wd2vc = KeyedVectors.load_word2vec_format('./models/crawl-300d-2M.vec')
stop_words = set(stopwords.words('english'))

def remove_StopWords(sentence):
    filtered_sentence = [] 
    for w in sentence: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    
    return filtered_sentence

def txt2vec(lst, max_num_sentence, max_num_words, max_length_sent):
    finalMatrix = np.zeros((20, 250, 300))
    
    if(max_length_sent < len(lst)):
        max_length_sent = len(lst)
        sent = lst
    
    for i in range(min(max_num_sentence, len(lst))):
        try:
            sentence = lst[i].split(" ")
        except:
            continue
        sentence = remove_StopWords(sentence)
        for j in range(min(max_num_words, len(sentence))):
            try:
                word = sentence[j]
                if(word[0] == '<'):
                    if(word.find('>')!=-1):
                        word = word[1:-1]
                    else:
                        word = word[1:]
                else:
                    if(word.find('>')!=-1):
                        word = word[0:-1]
                finalMatrix[i][j] = np.array(wd2vc[word])
            except Exception as e:
                continue
    return finalMatrix

def prcs_txt(txt_csv):
    lst = []
    txt_csv = np.array(txt_csv)[:, 2:4]
    
    max_num_words = 20
    max_num_sentence = 250
    max_length_sent = 0

    for i in range(1, len(txt_csv)):
        if(txt_csv[i][0] == 'Participant'):
            lst.append(txt_csv[i][1])
            
    lst = np.array(lst)

    return txt2vec(lst, max_num_sentence, max_num_words, max_length_sent)

def return_tensor(txt_csv):
    return torch.Tensor(prcs_txt(txt_csv)).to(torch.float32).view(-1, 300)