'''
Module containing definitions and procedures for linear model.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import *
from sklearn.svm import LinearSVC
import numpy as np
import re

## Constants ##
default_config = {
    'ngram_config': (1, 4),
    'stop_config': 'english',
    'tfidf_config': False,
    'min_df': 0.0001
}

## Custom Featurizers ## 
class RationaleFeaturizer:
    
    def __init__(self, rationales):
        self.rationales = [re.sub('\s', '', r) for r in rationales]
        
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
            txt = re.sub('\s', '', txt)
            for j in range(m):
                if self.rationales[j] in txt:
                    x[i,j] = 1
        return x


## Model Training ##
def svm_train(train_data, label, rationales=None, config=default_config):
    
    # Instantiate Vectorizers #
    section_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                         stop_words=config['stop_config'],
                                         min_df=config['min_df'])

    subsection_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                            stop_words=config['stop_config'],
                                            min_df=config['min_df'])

    header_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                           stop_words=config['stop_config'],
                                           min_df=config['min_df'])
    
    subheader_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                           stop_words=config['stop_config'],
                                           min_df=config['min_df'])
    
    text_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                      stop_words=config['stop_config'],
                                      min_df=config['min_df'])
    
    if label.startswith('populations'):
        header_vectorizer = RationaleFeaturizer(rationales)
        subheader_vectorizer = RationaleFeaturizer(rationales)
        text_vectorizer = RationaleFeaturizer(rationales)
        X_header = header_vectorizer.fit_transform(train_data['header'])
        X_subheader = subheader_vectorizer.fit_transform(train_data['subheader'])
        X_text = text_vectorizer.fit_transform(train_data['text'])
        if label == "populations - paediatric":
            X_train = np.hstack([X_header, X_subheader, X_text])
        elif label == "populations - adolescent":
            X_train = np.hstack([X_subheader, X_text])
        else:
            X_train = np.hstack([X_text])
    else:
        X_section = section_vectorizer.fit_transform(train_data['section']).toarray()
        X_subsection = subsection_vectorizer.fit_transform(train_data['subsection']).toarray()
        X_header = header_vectorizer.fit_transform(train_data['header']).toarray()
        X_subheader = subheader_vectorizer.fit_transform(train_data['subheader']).toarray()
        X_text = text_vectorizer.fit_transform(train_data['text']).toarray()
        X_train = np.hstack([X_section, X_subsection, X_header, X_subheader, X_text])
        
    Y_train = train_data[label]

    model = Pipeline([('tfidf', TfidfTransformer(use_idf=config['tfidf_config'])), \
    ('clf', LinearSVC(class_weight="balanced"))])
    model.fit(X_train, Y_train)
    
    return {
        'model': model,
        'sec_vec': section_vectorizer,
        'subsec_vec': subsection_vectorizer,
        'head_vec': header_vectorizer,
        'subh_vec': subheader_vectorizer,
        'text_vec': text_vectorizer,
        'label': label
    }

## Model Testing ##
def svm_test(test_data, params, verbose=False):
    
    label = params['label']
    
    if label.startswith('populations'):
        X_header = params['head_vec'].transform(test_data['header'])
        X_subheader = params['subh_vec'].transform(test_data['subheader'])
        X_text = params['text_vec'].transform(test_data['text'])
        if label == "populations - paediatric":
            X_test = np.hstack([X_header, X_subheader, X_text])
        elif label == "populations - adolescent":
            X_test = np.hstack([X_subheader, X_text])
        else:
            X_test = np.hstack([X_text])
    else:
        X_section = params['sec_vec'].transform(test_data['section']).toarray()
        X_subsection = params['subsec_vec'].transform(test_data['subsection']).toarray()
        X_header = params['head_vec'].transform(test_data['header']).toarray()
        X_subheader = params['subh_vec'].transform(test_data['subheader']).toarray()
        X_text = params['text_vec'].transform(test_data['text']).toarray()    
        X_test = np.hstack([X_section, X_subsection, X_header, X_subheader, X_text])
    
    Y_test = np.array((test_data[params['label']])).reshape(-1) * 1.0
    
    pred = np.array(params['model'].predict(X_test)).reshape(-1) * 1.0
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
            'actual_positive': test_data.loc[Y_test > 0][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']],
            'true_positive': test_data.loc[Y_test * pred > 0][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']],
            'false_positive': test_data.loc[pred * (1 - Y_test) > 0][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']],
            'false_negative': test_data.loc[Y_test * (1 - pred) > 0][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']]
        }
    else:
        output = {
            'precision': pres.sum()/c_freq.sum(),
            'recall': rec.sum()/c_freq.sum(),
            'cm': cm 
        }
    
    return output

def svm_predict(data, model):

    label = model['label']
    if label.startswith('populations'):
        X_header = model['head_vec'].transform(data['header'])
        X_subheader = model['subh_vec'].transform(data['subheader'])
        X_text = model['text_vec'].transform(data['text'])
        if label == "populations - paediatric":
            X_test = np.hstack([X_header, X_subheader, X_text])
        elif label == "populations - adolescent":
            X_test = np.hstack([X_subheader, X_text])
        else:
            X_test = np.hstack([X_text])
    else:
        X_section = model['sec_vec'].transform(data['section']).toarray()
        X_subsection = model['subsec_vec'].transform(data['subsection']).toarray()
        X_header = model['head_vec'].transform(data['header']).toarray()
        X_subheader = model['subh_vec'].transform(data['subheader']).toarray()
        X_text = model['text_vec'].transform(data['text']).toarray()    
        X_test = np.hstack([X_section, X_subsection, X_header, X_subheader, X_text])


    pred = np.array(model['model'].predict(X_test)).reshape(-1)
    return pred