#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:12:17 2017

@author: Moulya
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#read the reviews and their polarities from a given file // 2 lists one for file reviews and labels. 
def loadData(fname): 
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')  
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    return reviews,labels

rev_train,labels_train=loadData('reviews_train.txt') 
rev_test,labels_test=loadData('reviews_test.txt')


#Build a counter based on the training dataset// count the number of times the word appears, in practise sklearn does it for you. it
#finds all the unique terms, i.e. the count of words. you can ignore stopwords,lower case, consider 2 grams, you can 
#make your model smarter by changing line 29 here.  
counter = CountVectorizer()
counter.fit(rev_train)


#count the number of times each term appears in a document and transform each doc into a count vector// for every review it counts the number 
#of times it appears in the document, the cell 
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data'''

#train classifier // this is the only line that we have to change according to model, sometimes you may have to change the parameters
clf = RandomForestClassifier(n_estimators=2100,n_jobs=20,criterion="entropy",max_features='log2',random_state=150,max_depth=500,min_samples_split=170)

#train all classifier on the same datasets
clf.fit(counts_train,labels_train)

#use hard voting to predict (majority voting)
pred=clf.predict(counts_test)

#print accuracy the print will be full of zeros, in practise python only stores the non-zeroes, 
print (accuracy_score(pred,labels_test))



