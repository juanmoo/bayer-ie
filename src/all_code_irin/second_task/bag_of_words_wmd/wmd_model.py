import sys
import pandas as pd
import json
import ast
import os
from nltk.corpus import stopwords
import gensim.downloader as api
from datetime import datetime
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer

corpus = api.load('text8')
model = api.load("glove-wiki-gigaword-50")

MINIMUM_FREQUENCY_THRESHOLD = 5 #TODO decide this later
NUMBER_OF_PARAGRAPHS = 10
MAXIMUM_WORDS_IN_TITLE = 9
MINIMUM_SIZE_OF_PARAGRAPH = 3
COSINE_SIMILARITY_THRESHOLD = 0.90
PRECISION_AT = 1,5,10


lem = WordNetLemmatizer()
STOP_WORDS=set(stopwords.words("english"))


def get_bow_vector(text, relevant_words):
    tokenized_word=word_tokenize(text) #tokenized_word is a list of words
    punctuation_removed = list(filter(lambda word: word.isalnum(), tokenized_word))
    lowercase_words = list(map(lambda word: word.lower(), punctuation_removed))
    filtered_words=list(filter(lambda word: word not in STOP_WORDS, lowercase_words)) #stop words removed
    stemmed_words = list(map(lambda word: lem.lemmatize(word,"v"), filtered_words))
    lowercase_words = list(map(lambda word: word.lower(), stemmed_words)) 
    fdist = FreqDist(lowercase_words)
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
    lowercase_words = list(map(lambda word: word.lower(), punctuation_removed))
    filtered_words=list(filter(lambda word: word not in STOP_WORDS, lowercase_words)) #stop words removed
    stemmed_words = list(map(lambda word: lem.lemmatize(word,"v"), filtered_words))
    lowercase_words = list(map(lambda word: word.lower(), stemmed_words)) 
    fdist = FreqDist(lowercase_words)
    word_frequency_list = fdist.most_common(len(tokenized_word)) #list of (word, frequency) pairs
    word_frequency_threshold = list(filter(lambda t:  t[1] >= threshold, word_frequency_list))
    relevant_words = [word for (word, frequency) in word_frequency_threshold]
    return relevant_words


file_name = sys.argv[1]
json_files_path = sys.argv[2]

xls_file = pd.ExcelFile(file_name)
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
    
    broad_concept = row['Broad concept']
    broad_concept_paragraph = row['Broad Concept Paragraph']
    assessment = row['Text in assessment report 1']
    file_name = file_name = row['Link to assessment report 1 (or filename if using files sent by file transfer)']
    
    if type(assessment)==float:
        continue
    
    ####### from baseline_bow.py
    #drug_name =  get_drug_name(file_name)
    try:
        #file_path = (filter(lambda file: file[: len(drug_name)]==drug_name, review_files_xls))[0]
        file_path = file_name[:-4]+'.txt'
    except:
        #print("file for drug "+drug_name+" not found")
        print("filename not found : ", file_name)
        continue
    
    data = None
    if file_path[0]!='/':
        file_path = '/'+file_path
    try:  
        f = open(json_files_path + file_path[: -3]+'json', "r")
        data = f.read()
        print("starting file ")
        print(json_files_path + file_path[: -3]+'json')
    except:
        print("skipping this")
        
    if data==None:
        print("skipping this")
        continue
    
    #data = data.split("\n\n")
    data = ast.literal_eval(data)
    
    word_set1 = set(get_relevant_words(broad_concept_paragraph))
    wmd_distances = []
    for paragraph in data:
        word_set2 = set(get_relevant_words(paragraph))
        distance = model.wmdistance(word_set1, word_set2)
        wmd_distances.append((paragraph, distance))
    wmd_distances.sort(key=lambda t: t[1])
    if len(wmd_distances) > NUMBER_OF_PARAGRAPHS:
        wmd_distances = wmd_distances[: NUMBER_OF_PARAGRAPHS]
    
    all_predicted_paragraphs = [paragraph for paragraph, _ in wmd_distances]
    top_paragraphs = "\n\n".join(all_predicted_paragraphs) 
    
    assessment_list = assessment.split("\n")
    merged_assessment_list = [] #merging titles and body
    
    i=0
    while(True):
        if i >= len(assessment_list):
            break
        if len(word_tokenize(assessment_list[i])) <= MAXIMUM_WORDS_IN_TITLE and i+1 < len(assessment_list):
            merged_assessment_list.append("\n".join(assessment_list[i:i+2]))
            i=i+2
        else:
            merged_assessment_list.append(assessment_list[i])
            i=i+1
    
    truth_table = np.zeros((len(all_predicted_paragraphs), len(merged_assessment_list)), dtype=bool)
    
    for i,predicted_paragraph in enumerate(all_predicted_paragraphs):
        for j,merged_assessment in enumerate(merged_assessment_list):
            relevant_words = get_relevant_words(merged_assessment+predicted_paragraph)
            vector1 = get_bow_vector(predicted_paragraph, relevant_words)
            vector2 = get_bow_vector(merged_assessment, relevant_words)
            if vector1 is None or vector2 is None:
                continue
            distance = np.dot(vector1, vector2)
            
            if distance >= COSINE_SIMILARITY_THRESHOLD:
                truth_table[i,j] = True
                break
    
    precisions = {}
    recalls = {}
    
    for n in PRECISION_AT:
        true_positives = np.sum(truth_table[ 0: min(len(all_predicted_paragraphs),n) , :])
        precision = true_positives/min(len(all_predicted_paragraphs), n)
        if type(precision)==str:
            continue
        precisions[n] = precision 
        total_precision[n] += precision
        recall = true_positives/len(merged_assessment_list)
        if type(recall)==str:
            continue
        print("Recall@"+str(n)+":"+str(recall))
        recalls[n] = recall
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
    print("index=", index)
    total_valid_samples = total_valid_samples+1
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
    
timestring = datetime.now().strftime("%H:%M:%S")

output_path = sys.argv[3] if len(sys.argvs) >= 4 else os.getcwd()
writer = pd.ExcelWriter(output_path + '/wmd_top'+
                               str(NUMBER_OF_PARAGRAPHS)+"_"+timestring+'.xlsx', engine='xlsxwriter')
df_output.to_excel(writer,sheet_name='Main Worksheet',index=False)
writer.save()  
