import os
import sys
import argparse
import json
import re
from fuzzywuzzy import fuzz
from datetime import datetime
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api

# Parse Command Line Args #
argv = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--excel_path', help='Path to the Excel file')
parser.add_argument('--xml_path', help='Path of folder where XMLs are to be placed')
parser.add_argument('--method', default='BOW')
parser.add_argument('--output_path')
args = parser.parse_args(argv)


MINIMUM_FREQUENCY_THRESHOLD = 5 #TODO decide this later
NUMBER_OF_PARAGRAPHS = 10
MAXIMUM_WORDS_IN_TITLE = 9
MINIMUM_SIZE_OF_PARAGRAPH = 3
COSINE_SIMILARITY_THRESHOLD = 0.90
PRECISION_AT = 1,5,10

# lem = WordNetLemmatizer()
STOP_WORDS=set(stopwords.words("english"))

corpus = api.load('text8')
model = api.load("glove-wiki-gigaword-50")

def get_bow_vector(text, relevant_words):
    tokenized_word=word_tokenize(text) #tokenized_word is a list of words
    punctuation_removed = list(filter(lambda word: word.isalnum(), tokenized_word))
    words = list(map(lambda word: word.lower(), punctuation_removed))
    words=list(filter(lambda word: word not in STOP_WORDS, words)) #stop words removed
    # stemmed_words = list(map(lambda word: lem.lemmatize(word,"v"), filtered_words))
    words = list(map(lambda word: word.lower(), words)) 
    fdist = FreqDist(words)
    word_frequency_list = fdist.most_common(len(tokenized_word)) #list of (word, frequency) pairs
    if len(word_frequency_list) < MINIMUM_SIZE_OF_PARAGRAPH:
        return None
    word_frequency_dictionary = {word:frequency for word,frequency in word_frequency_list}
    word_vector = [0 for word in relevant_words]
    for i, word in enumerate(relevant_words):
        if word in word_frequency_dictionary:
            word_vector[i] = word_frequency_dictionary[word]
    word_vector = np.asarray(word_vector)
    return word_vector/np.linalg.norm(word_vector)

def get_relevant_words(text, threshold=1):
    tokenized_word=word_tokenize(text) #tokenized_word is a list of words
    punctuation_removed = list(filter(lambda word: word.isalnum(), tokenized_word))
    words = list(map(lambda word: word.lower(), punctuation_removed))
    words=list(filter(lambda word: word not in STOP_WORDS, words)) #stop words removed
    # stemmed_words = list(map(lambda word: lem.lemmatize(word,"v"), filtered_words))
    words = list(map(lambda word: word.lower(), words)) 
    fdist = FreqDist(words)
    word_frequency_list = fdist.most_common(len(tokenized_word)) #list of (word, frequency) pairs
    word_frequency_threshold = list(filter(lambda t:  t[1] >= threshold, word_frequency_list))
    relevant_words = [word for (word, frequency) in word_frequency_threshold]
    return relevant_words

def get_review_paragraphs(xml_file):
    
    if not os.path.isfile(xml_file):
        print(xml_file, "not exists")
        return None
    
    try:
        root = ET.parse(xml_file).getroot()
    except:
        print('xml_file', 'parsing failed')
        return None

    # remove tei-rul
    for child in root.iter():
        child.tag = child.tag.replace('{http://www.tei-c.org/ns/1.0}', '')

    paragraphs = []
    for p_elem in root.iter('p'):
        paragraphs.append(''.join(p_elem.itertext()).strip())
        
    return paragraphs

def contains_test(query, whole, ignore_spacing=True, minimum_paragraph_length=10):

    if len(query.split()) < minimum_paragraph_length:
        return False
    
    if ignore_spacing:
        # Remove spaces for consistent matchings
        query = re.sub('\s|\n', '', query)
        whole = re.sub('\s|\n', '', whole)

    threshold = 95
    match_score = fuzz.partial_ratio(query, whole)

    return match_score >= threshold


xls_file = pd.ExcelFile(args.excel_path)
dfs = {sheet_name: xls_file.parse(sheet_name) 
          for sheet_name in xls_file.sheet_names}

df = dfs["Main Worksheet"]

df_output = pd.DataFrame(columns=['Link to label (or filename if using files sent by file transfer)',
                                 'Broad concept', 'Broad Concept Paragraph', 'Text in assessment report 1',
                                 'Predicted paragraphs', 'Precision@1','Precision@5','Precision@10', 
                                 'Recall@1', 'Recall@5', 'Recall@10'])
    
total_precision = {i:0.0 for i in PRECISION_AT}  
total_recall = {i:0.0 for i in PRECISION_AT} 
total_valid_samples = 0
for index, row in df.iterrows():
    #try:
    broad_concept = row['Broad concept']
    broad_concept_paragraph = row['Broad Concept Paragraph']
    assessment = row['Text in assessment report 1']
    #file_name = row['Link to label (or filename if using files sent by file transfer)']
    file_name = row['Link to assessment report 1 (or filename if using files sent by file transfer)']
    
    if type(assessment) != str or type(file_name) != str:
        continue
        
    review_paragraphs = get_review_paragraphs(os.path.join(args.xml_path, file_name[1:].replace('.pdf', '.xml')))
    
    if review_paragraphs is None:
        print("skipping this")
        continue
    
    text= " ".join(review_paragraphs)
    relevant_words = get_relevant_words(text, MINIMUM_FREQUENCY_THRESHOLD)
    
    def computeBOW(broad_concept_paragraph, review_paragraphs):
        vector2 = get_bow_vector(broad_concept_paragraph, relevant_words)
        cosine_distances = []
        for paragraph in review_paragraphs:
            vector1 = get_bow_vector(paragraph, relevant_words)
            if vector1 is None:
                continue
            distance = np.dot(vector1, vector2)
            cosine_distances.append((paragraph, distance))
        cosine_distances.sort(key=lambda t: t[1], reverse=True)
        if len(cosine_distances) > NUMBER_OF_PARAGRAPHS:
            cosine_distances = cosine_distances[: NUMBER_OF_PARAGRAPHS]
        return cosine_distances
    
    def computeWMD(broad_concept_paragraph, review_paragraphs):
        word_set1 = set(get_relevant_words(broad_concept_paragraph))
        wmd_distances = []
        for paragraph in review_paragraphs:
            word_set2 = set(get_relevant_words(paragraph))
            distance = model.wmdistance(word_set1, word_set2)
            wmd_distances.append((paragraph, distance))
        wmd_distances.sort(key=lambda t: t[1])
        if len(wmd_distances) > NUMBER_OF_PARAGRAPHS:
            wmd_distances = wmd_distances[: NUMBER_OF_PARAGRAPHS]
        return wmd_distances
        
    if args.method == 'BOW':
        distances = computeBOW(broad_concept_paragraph, review_paragraphs)
    elif args.method == 'WMD':
        distances = computeWMD(broad_concept_paragraph, review_paragraphs)
    else:
        print('Method', args.method, 'not defined.')
        exit()
    
    all_predicted_paragraphs = [paragraph for paragraph, _ in distances]
    top_paragraphs = "\n\n".join(all_predicted_paragraphs) 
    
    truth_table = np.zeros(len(all_predicted_paragraphs), dtype=bool)
    
    for i,predicted_paragraph in enumerate(all_predicted_paragraphs):
        if contains_test(predicted_paragraph, assessment):
            truth_table[i] = True
    
    precisions = {}
    recalls = {}
    
    for n in PRECISION_AT:
        true_positives = np.sum(truth_table[ 0: min(len(all_predicted_paragraphs),n)])
        precision = true_positives/min(len(all_predicted_paragraphs), n)
        precisions[n] = precision   
        total_precision[n] += precision
        recall = true_positives/len(assessment.split('\n'))
        recalls[n] = recall
        print("Precision@"+str(n)+":"+str(precision), "Recall@"+str(n)+":"+str(recall))
        total_recall[n] += recall

                                 
    df_output = df_output.append({'Link to label (or filename if using files sent by file transfer)':file_name,
                      'Broad concept': broad_concept,
                      'Broad Concept Paragraph': broad_concept_paragraph,
                      'Text in assessment report 1':assessment,
                      'Predicted paragraphs': top_paragraphs,
                      'Precision@1': precisions[1],
                      'Precision@5': precisions[5],
                      'Precision@10': precisions[10],
                      'Recall@1': recalls[1],
                      'Recall@5': recalls[5],
                      'Recall@10': recalls[10]
                      }, ignore_index=True)    
    total_valid_samples = total_valid_samples + 1 
    print("index=", index)
#    except:
#        print("exception handled")
#        pass


df_output = df_output.append({'Link to label (or filename if using files sent by file transfer)':'Average',
                      'Broad concept': '',
                      'Broad Concept Paragraph': '',
                      'Text in assessment report 1':'',
                      'Predicted paragraphs': '',
                      'Precision@1': total_precision[1]/total_valid_samples,
                      'Precision@5': total_precision[5]/total_valid_samples,
                      'Precision@10': total_precision[10]/total_valid_samples,
                      'Recall@1': total_recall[1]/total_valid_samples,
                      'Recall@5': total_recall[5]/total_valid_samples, 
                      'Recall@10': total_recall[10]/total_valid_samples,
                      }, ignore_index=True)
    
    
writer = pd.ExcelWriter(args.output_path, engine='xlsxwriter')
df_output.to_excel(writer,sheet_name='Main Worksheet',index=False)
writer.save()  
