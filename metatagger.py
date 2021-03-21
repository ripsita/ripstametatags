
import re
import string
from math import log
from math import sqrt
import numpy as np
from numpy import array
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = WordNetLemmatizer()


def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    # Removing prefixed 'b'
    sentence = re.sub(r'^b\s+', '', sentence)

    return sentence


def remove_stop_words(sen):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sen)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    lamatised = [stemmer.lemmatize(word) for word in filtered_sentence]
    result = (" ").join(lamatised)
    return result


def getNgram(n, sen, ngrams):
    for i in range(len(sen)-n+1):
        ngram = sen[i:i+n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams


def getCatWiseNgramData(n):
    category_wise_ngram = {}
    for i in range(len(wiki_data)):
        cat_temp = wiki_data.loc[i, "category"]
        if cat_temp in category_wise_ngram:
            category_wise_ngram[cat_temp] = getNgram(
                n, wiki_data.loc[i, "clean_desc"], category_wise_ngram[cat_temp])
        else:
            category_wise_ngram[cat_temp] = getNgram(
                n, wiki_data.loc[i, "clean_desc"], {})
    return category_wise_ngram


def contain_punctuation(s):
    punctuation = [c for c in string.punctuation]
    punctuation.append(' ')
    r = any(c in s for c in punctuation)
    return r


def normalise_tfs(tfs, total):
    for k, v in tfs.items():
        tfs[k] = v / total
    return tfs


def log_idfs(idfs, num_cats):
    for k, v in idfs.items():
        idfs[k] = log(num_cats / v)
    return idfs


def cosine_similarity(v1, v2):
    num = np.dot(v1, v2)
    den_a = np.dot(v1, v1)
    den_b = np.dot(v2, v2)
    return num / (sqrt(den_a) * sqrt(den_b))


def read_category_vectors():
    vectors = {}
    for l in vector_list:
        l = l.rstrip('\n')
        fields = l.split()
        cat = fields[0]
        vec = np.array([float(v) for v in fields[1:]])
        vectors[cat] = vec
    return vectors


def get_ngrams(l, n):
    l = l.lower()
    ngrams = {}
    for i in range(0, len(l)-n+1):
        ngram = l[i:i+n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return ngrams


def normalise_tfs(tfs, total):
    for k, v in tfs.items():
        tfs[k] = v / total
    return tfs


def mk_vector(vocab, tfs):
    vec = np.zeros(len(vocab))
    for t, f in tfs.items():
        if t in vocab:
            pos = vocab.index(t)
            vec[pos] = f
    return vec


def findMatches(query):
    print('trying to find match for ', query)
    vectors = read_category_vectors()
    result = {}
    ngrams = {}
    cosines = {}
    for i in range(4, 7):
        n = get_ngrams(query, i)
        ngrams = {**ngrams, **n}
    qvec = mk_vector(vocab, ngrams)
    for cat, vec in vectors.items():
        cosines[cat] = cosine_similarity(vec, qvec)
    for cat in sorted(cosines, key=cosines.get, reverse=True):
        result[cat] = float(cosines[cat])*100
    return result


wiki_data = pd.read_csv("workingdata.csv")

wiki_data["full_decription"] = wiki_data["description"].astype(
    str) + wiki_data["description2"].astype(str)
wiki_data['clean_desc'] = wiki_data['full_decription'].apply(preprocess_text)
wiki_data['clean_desc'] = wiki_data['clean_desc'].apply(remove_stop_words)
cat_array = pd.unique(wiki_data['category'])
cat_data = pd.DataFrame(cat_array, columns=['category'])

allngrams = []
for i in range(4, 7):
    allngrams.append(getCatWiseNgramData(i))


cat_tfs = {}
cat_tf_idfs = {}
idfs = {}

for i in range(len(cat_data)):
    tfs = {}
    sum_freqs = 0
    cat_temp = cat_data.loc[i, "category"]
    for j in range(0, len(allngrams)):
        temp_ngrams_dict = allngrams[j][cat_temp]
        for key in temp_ngrams_dict:
            #print(key, '->', temp_ngrams_dict[key])
            ngram = key
            freq = int(temp_ngrams_dict[key])
            tfs[ngram] = freq
            sum_freqs += freq

            if ngram in idfs:
                idfs[ngram] += 1
            else:
                idfs[ngram] = 1
    tfs = normalise_tfs(tfs, sum_freqs)
    cat_tfs[cat_temp] = tfs

idfs = log_idfs(idfs, len(cat_data))
vocab = []

for i in range(len(cat_data)):
    cat = cat_data.loc[i, "category"]
    tf_idfs = {}
    tfs = cat_tfs[cat]
    for ngram, tf in tfs.items():
        tf_idfs[ngram] = tf * idfs[ngram]
    cat_tf_idfs[cat] = tf_idfs

    c = 0
    for k in sorted(tf_idfs, key=tf_idfs.get, reverse=True):
        # only keep top 100 dimensions per category. Also, we won't keep ngrams with spaces
        if c == 100:
            break
        if k not in vocab and not contain_punctuation(k):
            vocab.append(k)
            c += 1


vector_list = []
for i in range(len(cat_data)):
    cat = cat_data.loc[i, "category"]
    vec = np.zeros(len(vocab))
    cat_speci_tfidf = cat_tf_idfs[cat]
    for key in cat_speci_tfidf:
        #print(key, '->', cat_speci_tfidf[key])
        ngram = key
        tf_idf = float(cat_speci_tfidf[key])
        if ngram in vocab:
            pos = vocab.index(ngram)
            vec[pos] = tf_idf
    vector_list.append(cat+' '+' '.join([str(v) for v in vec]))
