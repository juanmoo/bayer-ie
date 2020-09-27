import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import *
from sklearn.svm import LinearSVC
import numpy as np
import random
import re
from utils import parse_spreadsheet
import warnings
warnings.filterwarnings("ignore")


annotations, all_rationales = parse_spreadsheet(['/data/rsg/nlp/yujieq/data/bayer/VendorEPAforMIT/CS Annotations_2020-01-20.xlsx', 
                                             '/data/rsg/nlp/yujieq/data/bayer/VendorEPAforMIT/CS Annotations_Additional rows.xlsx'])

all_labels = list(all_rationales.keys())


default_config = {
    'ngram_range': (1, 3),
    'stop_config': 'english',
    'tfidf_config': False,
    'min_df': 5
}

## Custom Featurizers ## 
class RationaleFeaturizer:
    
    def __init__(self, rationales):
        self.rationales = [re.sub('\s+', ' ', r).lower() for r in rationales]
        
    def fit_transform(self, text):
        x = self.transform(text)
        s = x.sum(axis=0) > 0
        self.rationales = [r for i,r in enumerate(self.rationales) if s[i]]
        x = x[:, s]
        return x
        
    def transform(self, text):
        text = list(text)
        n = len(text)
        m = len(self.rationales)
        x = np.zeros((n, m))
        for i in range(n):
            txt = text[i].lower()
            txt = re.sub('\s+', ' ', txt)
            for j in range(m):
                if self.rationales[j] in txt:
                    x[i,j] = 1
        return x

def add_rationale_features(train_data, test_data, X_train, X_test, rationales):
    r_featurizer = RationaleFeaturizer(rationales)
    
    X_r = r_featurizer.transform(train_data['text'])
    X_r1 = r_featurizer.transform(train_data['b1'])
    X_r2 = r_featurizer.transform(train_data['b2'])
    X_train = np.hstack([X_train, X_r, X_r1, X_r2])
    
    X_r = r_featurizer.transform(test_data['text'])
    X_r1 = r_featurizer.transform(test_data['b1'])
    X_r2 = r_featurizer.transform(test_data['b2'])
    X_test = np.hstack([X_test, X_r, X_r1, X_r2])
    
    return X_train, X_test


def prepare_train_test_data(train_data, test_data, config=default_config):
    
    # Instantiate Vectorizers #
    b_vectorizer = CountVectorizer(ngram_range=config['ngram_range'], \
                                         stop_words=config['stop_config'],
                                         min_df=config['min_df'])

    text_vectorizer = CountVectorizer(ngram_range=config['ngram_range'], \
                                      stop_words=config['stop_config'],
                                      min_df=config['min_df'])

    b_vectorizer.fit(train_data['b1'].to_list() + train_data['b2'].to_list())
    
    X_b1 = b_vectorizer.transform(train_data['b1']).toarray()
    X_b2 = b_vectorizer.transform(train_data['b2']).toarray()
    X_text = text_vectorizer.fit_transform(train_data['text']).toarray()
    
    X_train = np.hstack([X_b1, X_b2, X_text])
    
    X_b1 = b_vectorizer.transform(test_data['b1']).toarray()
    X_b2 = b_vectorizer.transform(test_data['b2']).toarray()
    X_text = text_vectorizer.transform(test_data['text']).toarray()
    
    X_test = np.hstack([X_b1, X_b2, X_text])
    
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    
    return X_train, X_test
    

## Model Training ##
def svm_train(X_train, Y_train, config=default_config):
    print(X_train.shape)
    
    model = Pipeline([('tfidf', TfidfTransformer(use_idf=config['tfidf_config'])), \
                        ('clf', LinearSVC(class_weight="balanced"))])
    model.fit(X_train, Y_train)
    
    return model

## Model Testing ##
def svm_test(model, X_test, Y_test, test_data, verbose=False):
    
    pred = np.array(model.predict(X_test))
    cm = np.array(confusion_matrix(Y_test, pred))
    
    # Diagonal elemetns were correctly classified
    diagonal = cm.diagonal()
    
    # Input class Counts
    class_sum = cm.sum(axis=1)
    
    # Predicted class counts
    pred_sum = cm.sum(axis=0)
    
    # Per-class performance w/ no-examples -> 0 perf
    recall = np.where(class_sum == 0, 0, diagonal/class_sum)
    precision = np.where(pred_sum == 0, 0, diagonal/pred_sum)
    
    if verbose:
        output = {
            'precision': precision[1:].sum(),
            'recall': recall[1:].sum(),
            'cm': cm,
            'prediction': pred,
            'actual_positive': test_data.loc[Y_test > 0][['doc_name', 'b1', 'b2', 'text']],
            'true_positive': test_data.loc[Y_test * pred > 0][['doc_name', 'b1', 'b2', 'text']],
            'false_positive': test_data.loc[pred * (1 - Y_test) > 0][['doc_name', 'b1', 'b2', 'text']],
            'false_negative': test_data.loc[Y_test * (1 - pred) > 0][['doc_name', 'b1', 'b2', 'text']]
        }
    else:
        output = {
            'precision': precision[1:].sum(),
            'recall': recall[1:].sum(),
            'cm': cm,
            'prediction': pred
        }
    
    return output


data = pd.read_pickle("processed_data.pkl")

random.seed(123)
docs = pd.unique(data['doc_name'])
random.shuffle(docs)

n_train = int(len(docs) * 0.8)
train_docs = docs[:n_train]
test_docs = docs[n_train:]

train_data = data.loc[data['doc_name'].isin(train_docs)]
test_data = data.loc[data['doc_name'].isin(test_docs)]


X_train_0, X_test_0 = prepare_train_test_data(train_data, test_data)

eval_results = {}
pred_results = {}

for label in all_labels:
    print()
    print(label)
    n_labeled = train_data[label].sum()
    print("#label", n_labeled)
    if n_labeled == 0:
        continue
    X_train, X_test = add_rationale_features(train_data, test_data, X_train_0, X_test_0, all_rationales[label])
    Y_train = train_data[label]
    Y_test = test_data[label]
    model = svm_train(X_train, Y_train)
    output = svm_test(model, X_test, Y_test, test_data, verbose=True)
    prec, recall = output['precision'], output['recall']
    f1 = 0 if prec+recall == 0 else 2*prec*recall/(prec+recall)
    output['f1'] = f1
    eval_results[label] = output
    pred_results[label] = output['prediction']
    print(f"Prec: {prec}   Recall: {recall}   F1: {f1}")


# Print evaluation
doc_list = pd.unique(data['doc_name'])
for label in all_labels:
    n_matched = data[label].sum()
    if label not in eval_results:
        continue
    res = eval_results[label]
    print(label, n_matched, res['precision'], res['recall'], res['f1'], sep='\t')

    
# Save predictions
df_output = pd.DataFrame(columns=['doc_name', 'b1', 'b2', 'text', 'matched', 'predicted'])
for i, x in enumerate(test_data.iterrows()):
    predicted = '||'.join([l for l, pred in pred_results.items() if pred[i] == 1])
    df_output = df_output.append({
        'doc_name': x[1]['doc_name'],
        'b1': x[1]['b1'],
        'b2': x[1]['b2'],
        'text': x[1]['text'],
        'matched': x[1]['labels'],
        'predicted': predicted
    }, ignore_index=True)

writer = pd.ExcelWriter('predictions_EPA.xlsx', engine='xlsxwriter')
df_output.to_excel(writer,sheet_name='Main Worksheet',index=False)
writer.save()
