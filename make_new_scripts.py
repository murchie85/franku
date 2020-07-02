
"""

## LAYERED LSTM Script generator
  
1. Import dependencies
2. Append all files to name array
    2.1 Append all file contents to string.
3. Load Spacy language model 
4. Extract tokens with manual function.  
5. Generate sequence array, with each element being 26 words, the following elemenet moved along by one word only. 
6. Import Keras tokeniser and `fit_on_texts` on sequence array. 
    6.1 Apply `texts_to_sequences` to the same sequence array.
    6.2 Verify fitted correctly by checking index 
7. Pass sequence to np array 
8. Construct observation matrix 
9. Define X and y values (25 tokens and 26th target token)
10. Define LSTM multi layer model with embeddings, softmax and relu 
11. Create model 
12. Fit model 
13. Save tokeniser and model
14. Generate new text
    14.1 Try existing seed seed
    14.2 Try with brand new seed
15. Repeat with various temperature values

"""
import numpy as np
import pandas as pd


# GET PATH 

from os import listdir
from os.path import isfile, join
mypath = input('Please enter exact path to target txt files \n')
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

data = ''
fullData = ''
fileList = []
for file in onlyfiles:
    if file[-3:] == 'txt':
        print(file)
        with open(str(mypath) + '/' + str(file)) as f:
            data = f.read().replace('\n', ' ')
    fullData = fullData + data


import spacy

# Load language model. 
nlp = spacy.load('en', disable = ['parser', 'tagger', 'ner'])


# EXTRACT TOKENS 
def get_tokens(doc_text):
    # This pattern is a modification of the defaul filter from the
    # Tokenizer() object in keras.preprocessing.text. 
    # It just indicates which patters no skip.
    skip_pattern = '\r\n \n\n \n\n\n!"-#$%&()--.*+,-./:;<=>?@[\\]^_`{|}~\t\n\r '
    
    tokens = [token.text.lower() for token in nlp(doc_text) if token.text not in skip_pattern]
    
    return tokens

# Get tokens.
tokens = get_tokens(fullData)

print('Token sample: ' + str(tokens[0:9]))
# Compute the number of tokens list.
print('Length of processed tokens are : ' + str(len(tokens) ))


len_0 = 25
print('Tokens/feature example is: \n'+ str(tokens[0:len_0]))
print('')
print('Target or label is ' + str(tokens[len_0:len_0 + 1]))

train_len = len_0 + 1

text_sequences = []

for i in range(train_len, len(tokens)):
    # Construct sequence.
    seq = tokens[i - train_len: i]
    # Append.
    text_sequences.append(seq)




# WORD TO VEC

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)


# Get numeric sequences.
sequences = tokenizer.texts_to_sequences(text_sequences)


print('Please verify these match')
print(tokenizer.index_word[int(sequences[0][0])])
print(fullData.split(' ')[0])

vocabulary_size = len(tokenizer.word_counts)
print('vocab size ' + str(vocabulary_size))




# Pass sequences to np array so we can use it in our xy split
sequences = np.array(sequences)


from keras.utils import to_categorical
# select all but last word indices.
X = sequences[:, :-1]
seq_len = X.shape[1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=(vocabulary_size + 1))

print('printing X array ')
print(X)
print('printing y array ')
print(y)


# Define Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

def create_model(vocabulary_size, seq_len):
    
    model = Sequential()

    model.add(Embedding(input_dim=vocabulary_size, 
                        output_dim=seq_len, 
                        input_length=seq_len))

    model.add(LSTM(units=50, return_sequences=True))

    model.add(LSTM(units=50))

    model.add(Dense(units=50, activation='relu'))

    model.add(Dense(units=vocabulary_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    model.summary()
    return model

model = create_model(vocabulary_size=(vocabulary_size + 1), seq_len=seq_len)


chosenEpoch = int(input('Please enter number of epochs, from 0 to 1000 \n'))
print('chosen value is: ' + str(chosenEpoch))
chosenBatch = int(input('Please enter number of batches, from 0 to 256. Default 128 \n'))
print('chosen value is: ' + str(chosenBatch))



print('fitting model please wait....')
model.fit(x=X, y=y, batch_size=chosenBatch, epochs=chosenEpoch, verbose=1)


# Produce loss and accuracy values
loss, accuracy =  model.evaluate(x=X, y=y)
print(f'Loss: {loss}\nAccuracy: {accuracy}')



from pickle import dump
dump(tokenizer, open('model/tokenizer', 'wb'))  
model.save('model/model.h5')


# GENERATE NEW TEXT 
print('generating new text')
from keras.preprocessing.sequence import pad_sequences 

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    # List to store the generated words. 
    output_text = []
    # Set seed_text as input_text. 
    input_text = seed_text
    
    for i in range(num_gen_words):
        # Encode input text. 
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        # Add if the input tesxt does not have length len_0.
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        # Do the prediction. Here we automatically choose the word with highest probability. 
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        # Convert from numeric to word. 
        pred_word = tokenizer.index_word[pred_word_ind]
        # Attach predicted word. 
        input_text += ' ' + pred_word
        # Append new word to the list. 
        output_text.append(pred_word)
        
    return ' '.join(output_text)



sample_text = ' '.join(tokens[100:150])
print('chosen sample pool is:..')
print(sample_text)  
seed_text = sample_text[0:100]
print('Seed text is...')
print(seed_text)

# GENERATE  

generated_text = generate_text(model=model, 
                               tokenizer=tokenizer,
                               seq_len=seq_len, 
                               seed_text=seed_text, 
                               num_gen_words=40)

print(seed_text + ' ' + generated_text + '...')


# SECOND MANUAL EXAMPLE  

seed_text = 'I eat a lot of ass, the whole ass I love the whole ass'

generated_text = generate_text(model=model, 
                               tokenizer=tokenizer,
                               seq_len=seq_len, 
                               seed_text=seed_text, 
                               num_gen_words=40)

print(seed_text + ' ' + generated_text + '...')  



# Trying again with various temp settings 


def generate_text2(model, tokenizer, seq_len, seed_text, num_gen_words, temperature):
    
    output_text = []
    
    input_text = seed_text
    
    for i in range(num_gen_words):
        # Encode input text. 
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
         # Add if the input tesxt does not have length len_0.
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        # Get learned distribution.
        pred_distribution = model.predict(pad_encoded, verbose=0)[0]
        
        # Apply temperature transformation.
        new_pred_distribution = np.power(pred_distribution, (1 / temperature)) 
        new_pred_distribution = new_pred_distribution / new_pred_distribution.sum()
        
        # Sample from modified distribution.
        choices = range(new_pred_distribution.size)
 
        pred_word_ind = np.random.choice(a=choices, p=new_pred_distribution)
        
        # Convert from numeric to word. 
        pred_word = tokenizer.index_word[pred_word_ind]
        # Attach predicted word. 
        input_text += ' ' + pred_word
        # Append new word to the list. 
        output_text.append(pred_word)
    return ' '.join(output_text)
        

seed_text = 'I eat a lot of ass, the whole ass I love the whole ass'
temp = [0.9, 0.5, 0.1]
for tempValue in temp:
    print("Trying temperature at value : " + str(tempValue))
    generated_text = generate_text2(model=model, 
                                tokenizer=tokenizer,
                                seq_len=seq_len, 
                                seed_text=seed_text, 
                                num_gen_words=80, 
                                temperature=tempValue)

    print(str(seed_text) + ' ' + str(generated_text) + ' ...')
    print('')
    print('--------')

