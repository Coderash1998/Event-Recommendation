from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from flashtext import KeywordProcessor
import numpy as np
import pandas as pd
import string
import spacy
nlp1=spacy.load('event')
nlp2=spacy.load('Domain')
de=pd.read_pickle("m2.pickle")
dd=pd.read_pickle("m1.pickle")
kd = KeywordProcessor()
ke = KeywordProcessor()
def get_domain(s):
    kd.add_keywords_from_dict(dd)
    return kd.extract_keywords(" ".join(s))    
def get_event(s):
    ke.add_keywords_from_dict(de)
    return ke.extract_keywords(" ".join(s))
def get_event_and_domain(word_data):
    stemmer = SnowballStemmer(language='english')  
    lemmatizer = WordNetLemmatizer() 
    nltk_tokens=[word.strip(string.punctuation) for word in word_data.split(" ")]
    doc1=nlp1(word_data)
    doc2=nlp2(word_data)
    ents1 = [eve.label_ for eve in doc1.ents]
    ents2 = [dom.label_ for dom in doc2.ents]
    e=get_event(nltk_tokens)
    e.extend(ents1)
    d=get_domain(nltk_tokens)
    d.extend(ents2)
    stem_word=[]
    lem_word=[]
    for w in nltk_tokens:
        lem_word.append(lemmatizer.lemmatize(w))
    e.extend(get_event(lem_word))
    d.extend(get_domain(lem_word))
    for w in lem_word:
        stem_word.append(stemmer.stem(w))
    e.extend(get_event(stem_word))
    d.extend(get_domain(stem_word))
    return list(np.unique(np.array(d))),list(np.unique(np.array(e)))
def give_input(word_data):
    return get_event_and_domain(word_data)

