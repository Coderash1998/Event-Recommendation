import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import geteventanddomain as ged
df = pd.read_csv("CCMLEmployeeData.csv")
df['Index']=range(0,len(df))
features1 = ['Domain','Event1']
features2 = ['Domain','Event2']
def combine_features1(row):
    try:
        return row['Domain'] + " " + row['Event1']
    except:
        print("Error:", row)
def combine_features2(row):
    try:
        return row['Domain'] + " " + row['Event2']
    except:
        print("Error:", row)        
df["combined_features1"] = df.apply(combine_features1,axis=1)
df["combined_features2"] = df.apply(combine_features2,axis=1)
cv = CountVectorizer()
count_matrix1 = cv.fit_transform(df['combined_features1'])
count_matrix2 = cv.fit_transform(df['combined_features2'])
count_matrix3 = cv.fit_transform(df['Event1'])
count_matrix4 = cv.fit_transform(df['Event2'])
cosine_sim1 = cosine_similarity(count_matrix1)
cosine_sim2 = cosine_similarity(count_matrix2)
cosine_sim3 = cosine_similarity(count_matrix3)
cosine_sim4 = cosine_similarity(count_matrix4)
def get_index_from_feature1(text):
    try:
        return df[df.combined_features1 == text]["Index"].values[0]
    except:
        return -1
def get_index_from_feature2(text):
    try:
        return df[df.combined_features2 == text]["Index"].values[0]
    except:
        return -1
def get_index_from_feature3(text):
    try:
        return df[df.Event1 == text]["Index"].values[0]
    except:
        return -1
def get_index_from_feature4(text):
    try:
        return df[df.Event2 == text]["Index"].values[0]
    except:
        return -1
def get_name_from_index(index):
	return df[df.index == index]["Name"].values[0]
def getnames1(a,b,s):
    similar_people1=[]
    similar_people2=[]
    similar_people=[]
    if get_index_from_feature1(s)==-1 and get_index_from_feature2(s)==-1:
        return []        
    else:
        if get_index_from_feature1(s)!=-1:
            similar_people1 = list(enumerate(cosine_sim1[get_index_from_feature1(s)]))
        if get_index_from_feature2(s)!=-1:
            similar_people2 = list(enumerate(cosine_sim2[get_index_from_feature2(s)]))        
        similar_people.extend(similar_people1)
        similar_people.extend(similar_people2)
        unique_people=[]        
        sorted_similar_people = sorted(similar_people, key = lambda x:x[1], reverse=True)
        for c in sorted_similar_people:
            if (get_name_from_index(c[0]) not in unique_people) and (c[1]>=0.9):
                unique_people.append(get_name_from_index(c[0]))        
        return unique_people
def getnames2(b):    
    similar_people1=[]
    similar_people2=[]
    similar_people=[]    
    if get_index_from_feature4(b)==-1 and get_index_from_feature3(b)==-1:
        return []
    else:
        if get_index_from_feature3(b)!=-1:
            similar_people1 = list(enumerate(cosine_sim3[get_index_from_feature3(b)]))
        if get_index_from_feature4(b)!=-1:
            similar_people2 = list(enumerate(cosine_sim4[get_index_from_feature4(b)]))       
        similar_people.extend(similar_people1)
        similar_people.extend(similar_people2)
        unique_people=[]       
        sorted_similar_people = sorted(similar_people, key = lambda x:x[1], reverse=True)
        for c in sorted_similar_people:
            if (get_name_from_index(c[0]) not in unique_people) and (c[1]>=0.9):
                unique_people.append(get_name_from_index(c[0])) 
        return unique_people
def inputline(text):
    d,e=ged.give_input(text)
    if (len(e)!=0 and len(d)!=0):
        name_list=[]
        for a in d:
            for b in e:
                s= a+" "+b
                name_each_event=getnames1(a,b,s)
               
                name_list.extend(name_each_event)
        return text,",".join(list(np.unique(np.array(name_list))))    
    elif (len(d)==0 and len(e)!=0):
        name_list=[]
        for b in e:
            name_each_event = getnames2(b)
            name_list.extend(name_each_event)
        return text,",".join(list(np.unique(np.array(name_list))))
    else:
        return text,""
