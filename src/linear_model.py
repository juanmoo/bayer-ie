'''
Module containing definitions and procedures for linear model.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import *
from sklearn.svm import LinearSVC
import numpy as np

## Constants ##
default_config = {
    'ngram_config': (1, 4),
    'stop_config': 'english',
    'tfidf_config': True,
    'min_df': 0.0001
}


## Model Training ##
def svm_train(train_data, label, config=default_config):
    
    # Instantiate Vectorizers #
    section_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                         stop_words=config['stop_config'],
                                         min_df=config['min_df'])

    subsection_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                            stop_words=config['stop_config'],
                                            min_df=config['min_df'])

    subheader_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                           stop_words=config['stop_config'],
                                           min_df=config['min_df'])

    header_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                           stop_words=config['stop_config'],
                                           min_df=config['min_df'])

    text_vectorizer = CountVectorizer(ngram_range=config['ngram_config'], \
                                      stop_words=config['stop_config'],
                                      min_df=config['min_df'])

    
    X_section = section_vectorizer.fit_transform(train_data['section']).toarray()
    X_subsection = subsection_vectorizer.fit_transform(train_data['subsection']).toarray()
    X_header = header_vectorizer.fit_transform(train_data['header']).toarray()
    X_subheader = subheader_vectorizer.fit_transform(train_data['subheader']).toarray()
    X_text = text_vectorizer.fit_transform(train_data['text']).toarray()

    X_train = np.hstack([X_section, X_subsection, X_header, X_subheader, X_text])
    Y_train = train_data[label]
    
    model = Pipeline([('tfidf', TfidfTransformer(use_idf=config['tfidf_config'])), ('clf', LinearSVC(class_weight="balanced"))])

    model.fit(X_train, Y_train)
    
    return {
        'model': model,
        'sec_vec': section_vectorizer,
        'subsec_vec': subsection_vectorizer,
        'header_vec': header_vectorizer,
        'subh_vec': subheader_vectorizer,
        'text_vec': text_vectorizer,
        'label': label
    }

## Model Testing ##
def svm_test(test_data, params, verbose=False):
    
    X_section = params['sec_vec'].transform(test_data['section']).toarray()
    X_subsection = params['subsec_vec'].transform(test_data['subsection']).toarray()
    X_header = params['header_vec'].transform(test_data['header']).toarray()
    X_subheader = params['subh_vec'].transform(test_data['subheader']).toarray()
    X_text = params['text_vec'].transform(test_data['text']).toarray()

    X_test = np.hstack([X_section, X_subsection, X_header, X_subheader, X_text])
    Y_test = np.array((test_data[params['label']])).reshape(-1, 1) * 1.0
    
    pred = np.array(params['model'].predict(X_test)).reshape(-1, 1) * 1.0
    cm = np.array(confusion_matrix(Y_test, pred))
    
    
    # Diagonal elemetns were correctly classified
    diagonal = cm.diagonal()
    
    # Input class Counts
    class_sum = cm.sum(axis=0)
    
    # Predicted class counts
    pred_sum = cm.sum(axis=1)
    
    # Per-class performance w/ no-examples -> 0 perf
    precision = np.where(class_sum == 0, 0, diagonal/class_sum)
    recall = np.where(pred_sum == 0, 0, diagonal/pred_sum)
    
    # Frequency Weighted Performance
    c_freq = cm.sum(axis=1)/cm.sum()
    pres = c_freq * precision
    rec = c_freq * recall
    
    # Remove 'other' Category
    c_freq = c_freq[1:]
    pres = pres[1:] 
    rec = rec[1:]
    
    if verbose:
        output = {
            'precision': pres.sum()/c_freq.sum(),
            'recall': rec.sum()/c_freq.sum(),
            'cm': cm,
            'all_predicted': test_data.loc[pred > 0][['doc_name', 'section', 'subsection', 'header', 'subheader', 'text']],
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