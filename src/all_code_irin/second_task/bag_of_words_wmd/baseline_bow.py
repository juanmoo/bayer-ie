import sys
import math
import json
import nltk 
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')

#filename = sys.argv[1]


MINIMUM_FREQUENCY_THRESHOLD = 0 #TODO decide this later

#text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
#The sky is pinkish-blue. You shouldn't eat cardboard irin irin Irin""" #TODO remove this later

text_list = ["I am a cow and I am a girl vector Vector I am a vector matrix too whatever"]
#with open(filename) as f:   
    #text_list=json.load(f)

text= " ".join(text_list)
    
tokenized_text=sent_tokenize(text) #tokenized_text is a list of sentences
tokenized_word=word_tokenize(text) #tokenized_word is a list of words

punctuation_removed = list(filter(lambda word: word.isalnum(), tokenized_word))
lowercase_words = list(map(lambda word: word.lower(), punctuation_removed))

stop_words=set(stopwords.words("english"))
filtered_words=list(filter(lambda word: word not in stop_words, lowercase_words)) #stop words removed


lem = WordNetLemmatizer()
stemmed_words = list(map(lambda word: lem.lemmatize(word,"v"), filtered_words))

lowercase_words = list(map(lambda word: word.lower(), stemmed_words)) 

fdist = FreqDist(lowercase_words)
word_frequency_list = fdist.most_common(len(tokenized_word)) #list of (word, frequency) pairs
word_frequency_threshold = list(filter(lambda t:  t[1] > MINIMUM_FREQUENCY_THRESHOLD, word_frequency_list))

sum_frequency_squares = sum([frequency*frequency for (word, frequency) in word_frequency_threshold])

normalizing_factor = math.sqrt(sum_frequency_squares)
word_frequency_normalized = [(word, frequency/normalizing_factor) for (word, frequency) in word_frequency_threshold]
  

def get_bow_vector(paragraph, relevant_words):
    tokenized_word=word_tokenize(text) #tokenized_word is a list of words
    punctuation_removed = list(filter(lambda word: word.isalnum(), tokenized_word))
    lowercase_words = list(map(lambda word: word.lower(), punctuation_removed))
    filtered_words=list(filter(lambda word: word not in stop_words, lowercase_words)) #stop words removed
    stemmed_words = list(map(lambda word: lem.lemmatize(word,"v"), filtered_words))
    lowercase_words = list(map(lambda word: word.lower(), stemmed_words)) 
    fdist = FreqDist(lowercase_words)
    word_frequency_list = fdist.most_common(len(tokenized_word)) #list of (word, frequency) pairs
    word_frequency_dictionary = {word:frequency for word,frequency in word_frequency_list}
    print("check=",  word_frequency_dictionary )
    word_vector = [0 for word in relevant_words]
    for i, word in enumerate(relevant_words):
        if word in word_frequency_dictionary:
            word_vector[i] = word_frequency_dictionary[word]
    word_vector = np.asarray(word_vector)
    return word_vector/np.linalg.norm(word_vector)

print(get_bow_vector(text_list, ['cow','girl','vector','matrix']))



