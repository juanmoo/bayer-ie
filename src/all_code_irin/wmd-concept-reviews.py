import os
import pandas as pd
import spacy
import wmd
import json



nlp = spacy.load('en_core_web_lg', create_pipeline=wmd.WMD.create_spacy_pipeline)

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
                                           
N = 20
for i, (file_name, broad_concept, assessment) in enumerate(file_concept):
    sentence1 = nlp(broad_concept)
    drug_name =  get_drug_name(file_name)
    min_wmd = float('inf')
    min_paragraph = None
    try:
        print(str(i+1)+". drug_name=", drug_name)
        file_path = list(filter(lambda f: f[: len(drug_name)]==drug_name, review_files_xls))[0]
        with open(path + directory + "/reviews-json/"+data_type+"/"+file_path) as f:
            data = json.load(f)
            for paragraph in data:
                sentence2 = nlp(paragraph)
                if len(list(paragraph.split())) < N:
                        continue
                wmd = sentence1.similarity(sentence2)
                if wmd < min_wmd:
                    min_wmd = wmd
                    min_paragraph = paragraph
        file_concept_paragraph_wmd.append([file_name, broad_concept, min_paragraph, min_wmd, assessment])            
    except:
        print("exception for drug ", drug_name)
        pass        
        


df = pd.DataFrame(file_concept_paragraph_wmd)
writer = pd.ExcelWriter(path + directory + '/testEMA.xlsx', engine='xlsxwriter')
df.to_excel(writer,sheet_name='test',index=False)
writer.save()    
