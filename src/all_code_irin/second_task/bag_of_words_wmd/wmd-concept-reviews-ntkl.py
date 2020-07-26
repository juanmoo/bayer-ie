import os
import pandas as pd
import json
from nltk.corpus import stopwords
from nltk import download
download('stopwords') 
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import gensim.downloader as api
from datetime import datetime

corpus = api.load('text8')
model = api.load("glove-wiki-gigaword-50")
#model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

path = os.getcwd()
directory = "/Desktop/Spring2020UROP"
data_type = "VendorEMAforMIT"

def get_drug_name(drug_file_name):
    drug_name = ""
    for e in drug_file_name[1 :]:
        if e=="-":
            break
        drug_name += e
    return drug_name 

file_name = path + directory+"/data/bayer/"+data_type+"/BayerLabelAnnotationWorksheet_Genpact_MasterEMA.xlsx"
xl_file = pd.ExcelFile(file_name)
dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}
    

df = dfs["Main Worksheet"]


broad_concepts = list(df['Broad Concept Paragraph'])
assessment = list(df['Text in assessment report 1'])
file_names = list(df['Link to label (or filename if using files sent by file transfer)'])

file_concept = [(file_names[i], broad_concepts[i], assessment[i]) for i in range(len(file_names))
                                                    if type(file_names[i])!=float 
                                                    if type(broad_concepts[i])!=float]

review_files_xls = os.listdir(path + directory + "/reviews-json/"+data_type)  

file_concept_paragraph_wmd = []                                              
stop_words = stopwords.words('english')

N = 5
M = 10
for i, (file_name, broad_concept, assessment) in enumerate(file_concept):
    sentence_a = broad_concept
    sentence_a = sentence_a.lower().split()
    drug_name =  get_drug_name(file_name)
    
    min_paragraph_wmd = []
    
    try:
        print(str(i+1)+". drug_name=", drug_name)
        file_path = list(filter(lambda f: f[: len(drug_name)]==drug_name, review_files_xls))[0]
        with open(path + directory + "/reviews-json/"+data_type+"/"+file_path) as f:
            data = json.load(f)
            for paragraph in data:
                sentence_b = paragraph
                if len(list(paragraph.split())) < N:
                        continue
            
                sentence_b = sentence_b.lower().split()
                sentence_a = [w for w in sentence_a if w not in stop_words]
                sentence_b = [w for w in sentence_b if w not in stop_words]
                distance = model.wmdistance(sentence_a, sentence_b)
                min_paragraph_wmd.append((paragraph, distance))
#                if distance < min_wmd:
#                    min_wmd = distance
#                    min_paragraph = paragraph
            min_paragraph_wmd.sort(key=lambda t: t[1]) 
            try:
                top_M_paragraphs = [min_paragraph_wmd[i][0] for i in range(M)]
                joiner = "\n"
                top_M_paragraphs = joiner.join(top_M_paragraphs)
                file_concept_paragraph_wmd.append([file_name, broad_concept, top_M_paragraphs, assessment])
            except:
                print("Less than ", str(M), "paragraphs found")
    except:
        print("exception")
        pass            
           
        


df = pd.DataFrame(file_concept_paragraph_wmd)
timestring = datetime.now().strftime("%H:%M:%S")
writer = pd.ExcelWriter(path + directory + '/output_xls_files/testEMA-ntkl-top'+str(M)+"-"+timestring+'.xlsx', engine='xlsxwriter')
df.to_excel(writer,sheet_name='test',index=False)
writer.save()    
