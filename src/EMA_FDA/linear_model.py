'''
Module containing definitions and procedures for linear model.
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import *
from sklearn.svm import LinearSVC
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import sys

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
        self.rationales = [re.sub('\s', '', r.lower()) for r in rationales]

    def fit_transform(self, text):
        x = self.transform(text)
        s = x.sum(axis=0) > 0
        self.rationales = [r for i, r in enumerate(self.rationales) if s[i]]
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
                    x[i, j] = 1
        return x


class DummyFeaturizer:
    def fit_transform(self, text):
        return self.transform(text)

    def transform(self, text):
        n = len(text)
        return np.zeros((n, 1))

## Model Training ##


def svm_train_ema(train_data, label, rationales=None, config=default_config, ignore_rationales=False):

    # Instantiate Vectorizers #
    section_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                         stop_words=config['stop_config'],
                                         min_df=config['min_df'])

    subsection_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                            stop_words=config['stop_config'],
                                            min_df=config['min_df'])

    header_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                        stop_words=config['stop_config'],
                                        min_df=config['min_df'])

    subheader_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                           stop_words=config['stop_config'],
                                           min_df=config['min_df'])

    text_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                      stop_words=config['stop_config'],
                                      min_df=config['min_df'])

    try:
        X_section = section_vectorizer.fit_transform(
            train_data['section']).toarray()
    except:
        print('No data for sections. Skipping sections')
        print('Training docs: ', list(pd.unique(train_data['doc_name'])))
        section_vectorizer = DummyFeaturizer()
        X_section = section_vectorizer.fit_transform(train_data['section'])

    try:
        X_subsection = subsection_vectorizer.fit_transform(
            train_data['subsection']).toarray()
    except:
        print('No data for subsections. Skipping subsections')
        print('Training docs: ', list(pd.unique(train_data['train_docs'])))
        subsection_vectorizer = DummyFeaturizer()
        X_subsection = subsection_vectorizer.fit_transform(
            train_data['subsection'])

    if not ignore_rationales and label.startswith('populations'):
        header_vectorizer = RationaleFeaturizer(rationales)
        subheader_vectorizer = RationaleFeaturizer(rationales)
        text_vectorizer = RationaleFeaturizer(rationales)

        X_header = header_vectorizer.fit_transform(train_data['header'])
        X_subheader = subheader_vectorizer.fit_transform(
            train_data['subheader'])
        X_text = text_vectorizer.fit_transform(train_data['text'])

        X_train = np.hstack(
            [X_section, X_subsection, X_header, X_subheader, X_text])
    else:

        X_header = header_vectorizer.fit_transform(
            train_data['header']).toarray()
        X_subheader = subheader_vectorizer.fit_transform(
            train_data['subheader']).toarray()
        X_text = text_vectorizer.fit_transform(train_data['text']).toarray()
        X_train = np.hstack(
            [X_section, X_subsection, X_header, X_subheader, X_text])

    Y_train = train_data[label]

    model = Pipeline([('tfidf', TfidfTransformer(use_idf=config['tfidf_config'])),
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


def svm_train_fda(train_data, label, rationales=None, config=default_config, ignore_rationales=False):

    # Instantiate Vectorizers #
    parent_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                        stop_words=config['stop_config'],
                                        min_df=config['min_df'])

    text_vectorizer = CountVectorizer(ngram_range=config['ngram_config'],
                                      stop_words=config['stop_config'],
                                      min_df=config['min_df'])

    if not ignore_rationales and label.startswith('populations'):
        parent_vectorizer = RationaleFeaturizer(rationales)
        text_vectorizer = RationaleFeaturizer(rationales)
        X_parent = parent_vectorizer.fit_transform(train_data['parent'])
        X_text = text_vectorizer.fit_transform(train_data['text'])
        X_train = np.hstack([X_parent, X_text])
    else:
        try:
            X_parent = parent_vectorizer.fit_transform(
                train_data['parent']).toarray()
        except:
            print('No data for parent. Skipping parent')
            print('Training docs: ', list(pd.unique(train_data['file'])))
            parent_vectorizer = DummyFeaturizer()
            X_parent = parent_vectorizer.fit_transform(train_data['parent'])

        X_text = text_vectorizer.fit_transform(train_data['text']).toarray()
        X_train = np.hstack([X_parent, X_text])

    Y_train = train_data[label]

    model = Pipeline([('tfidf', TfidfTransformer(use_idf=config['tfidf_config'])),
                      ('clf', LinearSVC(class_weight="balanced"))])
    model.fit(X_train, Y_train)

    return {
        'model': model,
        'par_vec': parent_vectorizer,
        'text_vec': text_vectorizer,
        'label': label
    }


# Predict

def svm_predict_ema(data, model, ignore_rationales=False):

    label = model['label']
    if not ignore_rationales and label.startswith('populations'):
        X_section = model['sec_vec'].transform(data['section']).toarray()
        X_subsection = model['subsec_vec'].transform(
            data['subsection']).toarray()
        X_header = model['head_vec'].transform(data['header'])
        X_subheader = model['subh_vec'].transform(data['subheader'])
        X_text = model['text_vec'].transform(data['text'])
        X_test = np.hstack(
            [X_section, X_subsection, X_header, X_subheader, X_text])
    else:
        X_section = model['sec_vec'].transform(data['section']).toarray()
        X_subsection = model['subsec_vec'].transform(
            data['subsection']).toarray()
        X_header = model['head_vec'].transform(data['header']).toarray()
        X_subheader = model['subh_vec'].transform(data['subheader']).toarray()
        X_text = model['text_vec'].transform(data['text']).toarray()
        X_test = np.hstack(
            [X_section, X_subsection, X_header, X_subheader, X_text])

    pred = np.array(model['model'].predict(X_test)).reshape(-1)
    return pred


def svm_predict_fda(data, model):

    label = model['label']
    if label.startswith('populations'):
        X_parent = model['par_vec'].transform(data['parent'])
        X_text = model['text_vec'].transform(data['text'])
        X_test = np.hstack([X_parent, X_text])
    else:
        X_parent = model['par_vec'].transform(data['parent']).toarray()
        X_text = model['text_vec'].transform(data['text']).toarray()
        X_test = np.hstack([X_parent, X_text])

    pred = np.array(model['model'].predict(X_test)).reshape(-1)
    return pred