# -*- coding: utf-8 -*-

import pandas as pd

read_file = pd.read_csv (r'SMSSpamCollection.txt',header=None,error_bad_lines=False, dtype={'col1' : str , 'col2' : str}, index_col=None)
read_file.columns = ['Spam','Ham']
#read_file.SentimentText=read_file.SentimentText.astype(str)
read_file.to_csv (r'dataset.csv', index= False)

import pandas as pd
import csv
#Data Loading
messages = [line.rstrip() for line in open('dataset.csv')]
print (len(messages))
#Appending column headers
messages = pd.read_csv('dataset.csv', sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"]) 

#Droping first record because it has float
messages= messages.iloc[1:]

data_size=messages.shape
print(data_size)   

messages_col_names=list(messages.columns)
print(messages_col_names)   

print(messages.groupby('label').describe())

print(messages.head(3))

#Identifying the outcome/target variable.
message_target=messages['label'] 
print(message_target)

target

#Tokenize the message
import nltk
#nltk.download('all')
from nltk.tokenize import word_tokenize
def split_tokens(message):
  message=message.lower()
  #message = str(message, 'utf8') #convert bytes into proper unicode But as it is python 3 so no need to convert
  word_tokens =word_tokenize(message)
  return word_tokens
messages['tokenized_message'] = messages.apply(lambda row: split_tokens(row['message']),axis=1)


#Lemmatization
from nltk.stem.wordnet import WordNetLemmatizer
def split_into_lemmas(message):
    lemma = []
    lemmatizer = WordNetLemmatizer()
    for word in message:
        a=lemmatizer.lemmatize(word)
        lemma.append(a)
    return lemma
messages['lemmatized_message'] = messages.apply(lambda row: split_into_lemmas(row['tokenized_message']),axis=1)

print('Tokenized message:',messages['tokenized_message'][11])
print('Lemmatized message:',messages['lemmatized_message'][11])


#Stop Word Removal
from nltk.corpus import stopwords
def stopword_removal(message):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    filtered_sentence = ' '.join([word for word in message if word not in stop_words])
    return filtered_sentence
messages['preprocessed_message'] = messages.apply(lambda row: stopword_removal(row['lemmatized_message']),axis=1)
Training_data=pd.Series(list(messages['preprocessed_message']))
Training_label=pd.Series(list(messages['label']))

#Create TDM
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tf_vectorizer = CountVectorizer(ngram_range=(1, 2),min_df = (1/len(Training_label)), max_df = 0.7)
Total_Dictionary_TDM = tf_vectorizer.fit(Training_data)
message_data_TDM = Total_Dictionary_TDM.transform(Training_data)


#Create TFIDF can be used in place of TDM
# =============================================================================
# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
# tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2),min_df = (1/len(Training_label)), max_df = 0.7)
# Total_Dictionary_TFIDF = tfidf_vectorizer.fit(Training_data)
# message_data_TFIDF = Total_Dictionary_TFIDF.transform(Training_data)
# =============================================================================

#Making Training data and test data
from sklearn.model_selection import train_test_split#Splitting the data for training and testing
train_data,test_data, train_label, test_label = train_test_split(message_data_TDM, Training_label, test_size=.1)



# =============================================================================
# 
# Classifire Algorithms
# 
# =============================================================================
#Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier#Creating a decision classifier model
classifier=DecisionTreeClassifier() #Model training
classifier = classifier.fit(train_data, train_label) #After being fitted, the model can then be used to predict the output.
message_predicted_target = classifier.predict(test_data)
score = classifier.score(test_data, test_label)
print('Decision Tree Classifier : ',score)

#Stochastic Gradient Descent Classifier
seed=7
from sklearn.linear_model import SGDClassifier
classifier =  SGDClassifier(loss='modified_huber', shuffle=True,random_state=seed)
classifier = classifier.fit(train_data, train_label)
score = classifier.score(test_data, test_label)
print('SGD classifier : ',score)


#Support Vector Machine
from sklearn.svm import SVC
classifier = SVC(kernel="linear", C=0.025,random_state=seed)
classifier = classifier.fit(train_data, train_label)
score = classifier.score(test_data, test_label)
print('SVM Classifier : ',score)


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10,random_state=seed)
classifier = classifier.fit(train_data, train_label)
score = classifier.score(test_data, test_label)
print('Random Forest Classifier : ',score)

#Model Tuning test for different parameter value
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=6, n_estimators=25, max_features=50,random_state=seed)
classifier = classifier.fit(train_data, train_label)
score=classifier.score(test_data, test_label)
print('Random Forest classification after model tuning',score)


#Stratified Shuffle Split
#The StratifiedShuffleSplit splits the data by taking an equal number of samples from each class in a random manner.
seed=7
from sklearn.model_selection import StratifiedShuffleSplit
###cross validation with 10% sample size
sss = StratifiedShuffleSplit(n_splits=1,test_size=0.1, random_state=seed)
sss.get_n_splits(message_data_TDM,Training_label)
print(sss)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
classifiers = [
    DecisionTreeClassifier(),
    SGDClassifier(loss='modified_huber', shuffle=True),
    SVC(kernel="linear", C=0.025),
    KNeighborsClassifier(),
    OneVsRestClassifier(svm.LinearSVC()),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10),
   ]
for clf in classifiers:
    score=0
    for train_index, test_index in sss.split(message_data_TDM,Training_label):
       X_train, X_test = message_data_TDM [train_index], message_data_TDM [test_index]
       y_train, y_test = Training_label[train_index], Training_label[test_index]
       clf.fit(X_train, y_train)
       score=score+clf.score(X_test, y_test)
    print(score)


from sklearn.metrics import accuracy_score
print('Accuracy Score',accuracy_score(test_label,message_predicted_target))  
classifier = classifier.fit(train_data, train_label)
score=classifier.score(test_data, test_label)
test_label.value_counts()


from sklearn.metrics import confusion_matrix
print('Confusion Matrix',confusion_matrix(test_label,message_predicted_target))


from sklearn.metrics import classification_report
target_names = ['spam', 'ham']
print(classification_report(test_label, message_predicted_target, target_names=target_names))










