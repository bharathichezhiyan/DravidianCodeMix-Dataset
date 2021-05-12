
# coding: utf-8

# In[43]:

###################### If it is not running 
#Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

import re
import nltk
#nltk.download('punkt', 'stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn.utils import class_weight

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfVectorizer

# In[27]:


df = pd.read_csv('tamil_sentiment_full.csv',names=['category','text'],sep='\t')
text=df[['text']]
labels=df[['category']]
#text





# In[5]:




# In[6]:


def take_data_to_shower(tweet):
    noises = ['URL', '@USER', '\'ve', 'n\'t', '\'s', '\'m']
    for noise in noises:
        tweet = tweet.replace(noise, '')
    return re.sub(r'[^a-zA-Z]', ' ', tweet)


def tokenize(tweet):
    lower_tweet = tweet.lower()
    return word_tokenize(lower_tweet)


def remove_stop_words(tokens):
    clean_tokens = []
    stopWords = set(stopwords.words('english'))
    for token in tokens:
        if token not in stopWords:
            if token.replace(' ', '') != '':
                if len(token) > 1:
                    clean_tokens.append(token)
    return clean_tokens


def stem_and_lem(tokens):
    clean_tokens = []
    for token in tokens:
        token = wordnet_lemmatizer.lemmatize(token)
        token = lancaster_stemmer.stem(token)
        if len(token) > 1:
            clean_tokens.append(token)
    return clean_tokens





##EMBEDDING##

###################### If it is not running  then data problem bring all into \t from first stage of data itself. 
clean_texts = copy.deepcopy(text)
tqdm.pandas(desc="Cleaning Data Phase I...")
clean_texts['text'] = text['text'].progress_apply(take_data_to_shower)

tqdm.pandas(desc="Tokenizing Data...")
clean_texts['tokens'] = clean_texts['text'].progress_apply(tokenize)

tqdm.pandas(desc="Cleaning Data Phase II...")
clean_texts['tokens'] = clean_texts['tokens'].progress_apply(remove_stop_words)

tqdm.pandas(desc="Stemming And Lemmatizing")
clean_texts['tokens'] = clean_texts['tokens'].progress_apply(stem_and_lem)

text_vector = clean_texts['tokens'].tolist()


# In[36]:



def tfid(text_vector):
    vectorizer = TfidfVectorizer()
    untokenized_data =[' '.join(tweet) for tweet in tqdm(text_vector, "Vectorizing...")]
    vectorizer = vectorizer.fit(untokenized_data)
    vectors = vectorizer.transform(untokenized_data).toarray()
    return vectors
  
def get_vectors(vectors, labels, keyword):
    if len(vectors) != len(labels):
        print("Unmatching sizes!")
        return
    result = list()
    for vector, label in zip(vectors, labels):
        if label == keyword:
            result.append(vector)
    return result


vectors_a = tfid(text_vector) # Numerical Vectors A
labels_a = labels['category'].values.tolist() # Subtask A Labels


def compute_class_weight_dictionary(y):
    # helper for returning a dictionary instead of an array
    classes = np.unique(y)
    class_weights = class_weight.compute_class_weight("balanced", classes, y)
    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict

def classify(vectors, labels, type="DT"):
    # Random Splitting With Ratio 3 : 1
    train_vectors, test_vectors, train_labels, test_labels = train_test_split(vectors, labels, random_state=5, test_size=0.1)
    class_weights=compute_class_weight_dictionary(train_labels)
    #class_weights = class_weight.compute_class_weight('balanced',np.unique(train_labels),train_labels)
    # In[25]:

    print(class_weights)


    # Initialize Model
    classifier = None
    if(type=="MNB"):
        classifier = MultinomialNB(alpha=0.7)
        classifier.fit(train_vectors, train_labels)
    elif(type=="KNN"):
        classifier = KNeighborsClassifier(n_jobs=4)
        classifier.fit(train_vectors, train_labels)
        
    elif(type=="SVM"):
        classifier = SVC()
        classifier.fit(train_vectors, train_labels)
        
    elif(type=="DT"):
        classifier = DecisionTreeClassifier(max_depth=800, min_samples_split=5)
        classifier.fit(train_vectors, train_labels)
        
    elif(type=="LR"):
        classifier = LogisticRegression(multi_class='auto', solver='newton-cg',class_weight=class_weights)
        classifier.fit(train_vectors, train_labels)
        
    elif(type=="RF"):
        classifier = RandomForestClassifier(n_estimators=100, max_depth=800, min_samples_split=5)
        classifier.fit(train_vectors, train_labels)
        
    else:
        print("Wrong Classifier Type!")
        return

    accuracy = accuracy_score(train_labels, classifier.predict(train_vectors))
    print("Training Accuracy:", accuracy)
    test_predictions = classifier.predict(test_vectors)
    accuracy = accuracy_score(test_labels, test_predictions)
    print("Test Accuracy:", accuracy)
    print("Confusion Matrix:", )
    print(confusion_matrix(test_labels, test_predictions))
    print(classification_report([i for i in test_labels], 
                            [i for i in test_predictions]))

print("\nBuilding Model LR...")
classify(vectors_a, labels_a, "LR")

print("\nBuilding Model SVM...")
classify(vectors_a, labels_a, "SVM") # 

print("\nBuilding Model MNB...")
classify(vectors_a, labels_a, "MNB")


print("\nBuilding Model KNN...")
classify(vectors_a, labels_a, "KNN")


print("\nBuilding Model DT...")
classify(vectors_a, labels_a, "DT")


print("\nBuilding Model RF...")
classify(vectors_a, labels_a, "RF")
