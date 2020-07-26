import sys
import os
import nltk
import pandas as pd
import json
from datetime import datetime
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

path = os.getcwd()
directory = "/Desktop/Spring2020UROP"
data_type = "VendorEMAforMIT"
filename = sys.argv[1] 

def get_drug_name(drug_file_name):
    drug_name = ""
    for e in drug_file_name[1 :]:
        if e=="-":
            break
        drug_name += e
    return drug_name 
    
file_name = path + directory+"/output_xls_files/"+filename
xls_file = pd.ExcelFile(file_name)
dfs = {sheet_name: xls_file.parse(sheet_name) 
          for sheet_name in xls_file.sheet_names}
    
df = dfs[list(dfs.keys())[0]]


broad_concepts = list(df['Broad Concept Paragraph'])
assessment = list(df['Text in assessment report 1'])
file_names = list(df['Link to label (or filename if using files sent by file transfer)'])
predicted_paragraphs = list(df['Paragraphs'])

file_concept = [(file_names[i], broad_concepts[i], predicted_paragraphs[i], assessment[i]) for i in range(len(file_names))
                                                    if type(file_names[i])!=float 
                                                    if type(broad_concepts[i])!=float]

output = []
review_files_xls = os.listdir(path + directory + "/reviews-json/"+data_type) 

THRESHOLD = 0.1

for i, (file_name, prediction, broad_concept, assessment) in enumerate(file_concept):
    
    print("i=", str(i))
    if type(assessment) == float:
        print("Invalid inf")
        continue
    all_predicted_paragraphs = "\n".split(prediction)
    assessment_paragraphs = "\n".split(assessment)
    found = {}
    for j,p1 in enumerate(all_predicted_paragraphs):
        for k,p2 in enumerate(assessment_paragraphs):
            s1 = set(word_tokenize(p1))
            s2 = set(word_tokenize(p2))
            try:
                distance = nltk.jaccard_distance(s1, s2) 
                print("distance=", distance)
            except: 
                print("jaccard problem")
                continue
            print("distance=", distance)
            if distance < THRESHOLD:
                found[j] = (k, distance)
                continue
    drug_name =  get_drug_name(file_name)        
    file_path = list(filter(lambda f: f[: len(drug_name)]==drug_name, review_files_xls))[0]
    
    total_paragraphs = 0
    with open(path + directory + "/reviews-json/"+data_type+"/"+file_path) as f:
        data = json.load(f)
        total_paragraphs = len(data)
            
        
    false_positive = len(all_predicted_paragraphs) - len(list(found.keys()))
    true_positive = len(list(found.keys()))
    false_negative = len(assessment_paragraphs) -  len(set([v1 for k,(v1, _) in found])) 
       
    precision = true_positive/(true_positive+false_positive)
    recall =  true_positive/(true_positive+false_negative)
    f1 = "Invalid"
    try:
        f1 = 2*precision*recall/(precision + recall)
    except:
        print("F1 problem")
    
    output.append((file_name, prediction, broad_concept, assessment, precision, recall, f1))      
          
           
        


df = pd.DataFrame(output, columns = ['Link to label (or filename if using files sent by file transfer)', 
                                     'Broad Concept Paragraph','Paragraphs','Text in assessment report 1',
                                     'Precision', 'Recall', 'F1 score'])
timestring = datetime.now().strftime("%H:%M:%S")
writer = pd.ExcelWriter(path + directory + '/output_xls_files/testEMA-ntkl-prf1'+timestring+'.xlsx', engine='xlsxwriter')
df.to_excel(writer,sheet_name='test',index=False)
writer.save()    
