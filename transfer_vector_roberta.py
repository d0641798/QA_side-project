import csv
import tkinter as tk
import os
import numpy as np
from scipy.linalg import norm
import torch
import pdb
from transformers import BertTokenizerFast,AutoModel
from transformers import BertTokenizer, BertModel, BertConfig



questiontext=[]
question_vector=[]

tokenizer = BertTokenizer.from_pretrained("clue/roberta_chinese_base")
model=BertModel.from_pretrained("clue/roberta_chinese_base", output_hidden_states = True)

with open('QA_new.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:
    print(row)
    questiontext.append(row[0])
    
    #print(row[0])


    string1 = "[CLS]" + row[0] + "[SEP]"

    tokenized_string = tokenizer.tokenize(string1)
    tokens_ids = tokenizer.convert_tokens_to_ids(tokenized_string)
        
    tokens_tensor = torch.tensor([tokens_ids])
    outputs = model(tokens_tensor) 

    if model.config.output_hidden_states:
        hidden_states = outputs[2]
        second_to_last_layer = hidden_states[-2]
        token_vecs = second_to_last_layer[0]

        sentence_embedding = torch.mean(token_vecs, dim=0)
        vector=sentence_embedding.detach().numpy()

        question_vector.append(vector)


corpus_vector = np.array(question_vector)
np.save('questions_vector_roberta',corpus_vector)
print(corpus_vector.shape)