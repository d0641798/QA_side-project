# -*- coding: UTF-8 -*-
import csv
import tkinter as tk
import os
import numpy as np
from scipy.linalg import norm
from numpy import linalg as LA
import torch
import pdb
from transformers import AutoConfig
from transformers import BertTokenizer, BertModel, BertConfig

import heapq

from tkinter import *
import time

questiontext=[]
answerindex=[]
answertext=[]

with open('QA_new.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:

    questiontext.append(row[0])
    answerindex.append(row[1])


with open('answer_index.csv',encoding='utf-8', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:

    answertext.append(row[0])

tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
model=BertModel.from_pretrained("clue/roberta_chinese_base", output_hidden_states = True)

question_vector=np.load('questions_vector_roberta.npy')
#print(question_vector.shape)
model.eval()

s = []
temp=0


def input_button1():
    question_input=entry.get()
    index_judge=[]
    answer_history=[]
       
    start = time.time()
    for i in range(len(question_vector)):
        temp=vector_similarity(question_input,question_vector[i])
        #print(question_vector[i])
        
        answer_history.append(temp)

    end = time.time()
    print(end - start)
    result=''

  
    
    combine_question_index=list(zip(answer_history,answerindex))

    final=sorted(combine_question_index, key = lambda s: s[0],reverse=True)
    #print(answer_history.index(max_score))
    count=0
    score_5=[]
    for i in final:
        if count==5:
            print(count)
            break  
        if i[1] in index_judge:
            continue
        else:
            score_5.append(i[0])
            index_judge.append(i[1])
            count+=1

    for i in range(len(score_5)):

        for j in range(len(answer_history)): #所有相似度
            if answer_history[j]==score_5[i]:
                
                var1=tk.StringVar()
                if score_5[i]<0.8:  #這裡可以更改信心分數下限
                    answer_output=str(i+1)+"."+"沒有找到匹配的答案"
                    result=result+answer_output
                    break    
                else:

                    answer_output=str(i+1)+"."+answertext[int(answerindex[j])-1]+'\n'
                    result=result+answer_output
                    ans_label1['text']=result
                    break
    


def vector_similarity(s1, s2):

    
 
    string1 = "[CLS]" + s1 + "[SEP]"

    tokenized_string = tokenizer.tokenize(string1)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_string)      
    tokens_tensor = torch.tensor([tokens_ids])
    outputs = model(tokens_tensor) 

    if model.config.output_hidden_states:
        hidden_states = outputs[2]
            
        # last_layer = outputs[-1]
        second_to_last_layer = hidden_states[-2]
        print(hidden_states[0].shape)
        token_vecs = second_to_last_layer[0]

        sentence_embedding = torch.mean(token_vecs, dim=0)
        
        vector=sentence_embedding.detach().numpy()

    #print(cos_similarity)
    #print(np.dot(vector, s2) / (norm(vector) * norm(s2)))
        #sentence_embedding=sentence_embedding.cuda()
        #s2=s2.cuda()
    return np.dot(vector, s2) / (LA.norm(vector) * LA.norm(s2))


window = tk.Tk()
window.title('問答系統_RoBERTa')
window.geometry("1500x1000")

# 標示文字
version = tk.Label(window, text = 'RoBERTa版本')
version.pack()

label = tk.Label(window, text = '輸入問題')
label.pack()

var1=tk.StringVar()


# 輸入欄位
entry = tk.Entry(window,     # 輸入欄位所在視窗
                 width = 200) # 輸入欄位的寬度
entry.pack()

# 按鈕
result=''
button1 = tk.Button(window, text = "確定", command = input_button1)



#ans_label1.config()

button1.pack()


    
var1.set(result)
        
ans_label1 = tk.Label(window,justify = 'left',width=200,wraplength=800)
        
ans_label1.place(x=50,y=100)

window.mainloop()