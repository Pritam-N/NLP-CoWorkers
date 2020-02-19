<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 0ade81c21c6b974412a602a001f568d0415ae82d
import nltk
import pandas as pd

nltk.download_shell()
message = [line.rstrip() for line in open('SMSSpamCollection')]

#the message are tab separated as ham/span 
#so split/separate that message to make the data set of label and message
messages = pd.read_csv('SMSSpamCollection', sep='\t' , names=['label','message'])
#to show the dataset structure
messages.head()
#to see the description of the message
messages.describe()
#to see the description of the message groupby label
messages.groupby('label').describe()
#adding a new column length to see the message length by applying len funtion
messages['length'] = messages['message'].apply(len)

#data visualisation for by length of message for more exploring the the dataset
import matplotlib.pyplot as plt
import seaborn as sns
#histogram
messages['length'].plot.hist(bins=50)

# Now how to deal with sentence to fetch the words for further use and trainimg and making bags of words
#so for that from each message/sentence follow these steps
# 1.> Remove the puntuation 	
# 2.> Remove the stop words
# 3.> Return the list of clean words

# for getting the Stopwords of english words we need to download that from download_shell()
# Enter the command d for download and then enter stopwords and the quite
########  Example ########
nopunc.split()
nltk.download_shell()

import string
mess = 'sample message! it has a puntuation for check.'
nopunc = [c for c in mess if c not in string.punctuation]
from nltk.corpus import  stopwords
nopunc = ''.join(nopunc)
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#O/P - clean_mess - ['sample', 'message', 'puntuation', 'check']

# function to return the clean_mess

#BoW model
# 1.>count the times that a word occured in a message (term frequency)
# 2.>Weight the counts, so that frequency tokens get lower weight(inverse document frequency)
# 3.>Normalise the vectors to unit length, to abstract from the origional text length (L2 norm)

# Then a matrix so denote words occured number of times in different messages
#use sparse matrix

def text_process(mess):
  nopunc = [char for char in mess if char not in string.punctuation]
  nopunc = ''.join(nopunc)
  return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
  
#Frequency of the words
from sklearn.feature_extraction.text import CountVectorizer
#Bags of Words
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

#To check the BoW
mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])

#O/P
///
  (0, 4068)	2
  (0, 4629)	1
  (0, 5261)	1
  (0, 6204)	1
  (0, 6222)	1
  (0, 7186)	1
  (0, 9554)	2

///


<<<<<<< HEAD
=======
=======
import nltk
import pandas as pd

nltk.download_shell()
message = [line.rstrip() for line in open('SMSSpamCollection')]

#the message are tab separated as ham/span 
#so split/separate that message to make the data set of label and message
messages = pd.read_csv('SMSSpamCollection', sep='\t' , names=['label','message'])
#to show the dataset structure
messages.head()
#to see the description of the message
messages.describe()
#to see the description of the message groupby label
messages.groupby('label').describe()
#adding a new column length to see the message length by applying len funtion
messages['length'] = messages['message'].apply(len)

#data visualisation for by length of message for more exploring the the dataset
import matplotlib.pyplot as plt
import seaborn as sns
#histogram
messages['length'].plot.hist(bins=50)

# Now how to deal with sentence to fetch the words for further use and trainimg and making bags of words
#so for that from each message/sentence follow these steps
# 1.> Remove the puntuation 	
# 2.> Remove the stop words
# 3.> Return the list of clean words

# for getting the Stopwords of english words we need to download that from download_shell()
# Enter the command d for download and then enter stopwords and the quite
########  Example ########
nopunc.split()
nltk.download_shell()

import string
mess = 'sample message! it has a puntuation for check.'
nopunc = [c for c in mess if c not in string.punctuation]
from nltk.corpus import  stopwords
nopunc = ''.join(nopunc)
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#O/P - clean_mess - ['sample', 'message', 'puntuation', 'check']

# function to return the clean_mess

#BoW model
# 1.>count the times that a word occured in a message (term frequency)
# 2.>Weight the counts, so that frequency tokens get lower weight(inverse document frequency)
# 3.>Normalise the vectors to unit length, to abstract from the origional text length (L2 norm)

# Then a matrix so denote words occured number of times in different messages
#use sparse matrix

def text_process(mess):
  nopunc = [char for char in mess if char not in string.punctuation]
  nopunc = ''.join(nopunc)
  return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
  
#Frequency of the words
from sklearn.feature_extraction.text import CountVectorizer
#Bags of Words
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

#To check the BoW
mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])

#O/P
///
  (0, 4068)	2
  (0, 4629)	1
  (0, 5261)	1
  (0, 6204)	1
  (0, 6222)	1
  (0, 7186)	1
  (0, 9554)	2

///

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(messages_bow)

#Using Nayive bayes classifire
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])

from sklearn.model_selection import train_test_split

msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
                     ('bow', CountVectorizer(analyzer=text_process)),
                     ('tfidf', TfidfTransformer()),
                     ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)

predictions = pipeline.predict(msg_test)

from sklearn.metrics import classification_report
print(classification_report(label_test, predictions))


>>>>>>> 0ade81c21c6b974412a602a001f568d0415ae82d
