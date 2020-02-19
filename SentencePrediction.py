import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = input("Enter the text for prediction: ") 
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

list = []
indexed_tokens = tokenizer.encode(text)

# Convert indexed tokens in a PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
model.to('cuda')

outputs = model(tokens_tensor)
predictions = outputs[0]
predictions[0][0][0]

# Get the predicted next sub-word
sorted_length, sorted_index = torch.sort(predictions[0 ,-1, :], dim = 0, descending=True)
new_sentence_order = sorted_index.tolist()

predicted_index = new_sentence_order[0]
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
text =  predicted_text
list.append(text)

stopWords =[',', '!', '.', '?', '/'];

#def getCompleteSentence(text,stopwords):
countwords = 0    

while(1): 
    countwords = countwords + 1
    indexed_tokens = tokenizer.encode(list[0])
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    model.to('cuda')
    
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
            
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    list[0] = predicted_text
    if any(predicted_text.endswith(s) for s in stopWords):
        break;
    if(countwords >= 10 ):
        break;

print(list[0])
