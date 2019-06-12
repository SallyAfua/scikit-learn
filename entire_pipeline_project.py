# import the needed package for the tasks

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler



import time
import random

then = time.time() #Time before the operations start

#DO YOUR OPERATIONS HERE



# parameters for the pipeline

num_reviews = 20
eps = 0.3
min_samples = 4

# read the tsv file
obs=[]
filename = "/home/salomey/Salomay/ESSAY_PHASE_A/Data_for_project/amazon_reviews_us_Musical_Instruments_v1_00.tsv/data_musical.tsv"
file = open(filename, mode='r') 
for line in file.readlines():
    obs.append(line.split('\t'))



# visualizing the data in a dataframe using pandas
data=pd.DataFrame(obs,columns=["marketplace ","customer_id" ,"review_id ","product_id ","product_parent" ,"product_title","product_category" ,"star_rating" ,"helpful_votes" ,"total_votes","vine", "verified_purchase" ,"review_headline" ,"review_body", "review_date"])


# drop the first row
data = data.drop([0], axis=0)
#print("These are the amazon review data for video",data)

# extracting positive reviews based on star ratings>2

positive_review= data[data['star_rating']>'2']



# extracting the positive review and converting into a list
positives = positive_review.review_body.tolist()
#print("These are the positive reviews:",positives)

# extracting negative reviews based on star ratings<=2

negative_df=data[data['star_rating']<='2']


# converting negative reviews into a list
negatives = negative_df.review_body.tolist()
#print("These are the negative reviews:",negatives)
# pre-processing reviews(remove stop words and non-alphabetic characters)

def process(review, remove_stopwords=True):
   
    review_text = re.sub("[^a-zA-Z]"," ", review)
   
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    cleaned_reviews=[]
    
    for word in words:
        cleaned_reviews.append(word)

    
    return(cleaned_reviews)

cleaned_review_body = []
for review in negatives:# change "negatives" to "positives" if you are interested in cleaning the positive reviews
    cleaned_review_body.append( " ".join(process(review)))

tokkenn=[]
for r in cleaned_review_body:
    tokkenn.append(process(r))
#print("these are the pre-processed words", tokkenn)
# converting the words into vectors using ELMo

elmo = ElmoEmbedder()
word_embeddings = []
for reviews in tokkenn[:num_reviews]:
    word_embeddings.append(elmo.embed_sentence(reviews)[0]) # select the top layer of the ELMo vector

#print("These are the word embeddings for the reviews",word_embeddings)
#print(len(word_embeddings))
X=(np.concatenate(word_embeddings))
#print(type(X)) 
#print("This is the shape of our vector X",np.shape(X))

#################################################################################

# Now we cluster our vector X using sklearn DBSCAN
    
X = StandardScaler().fit_transform(np.asarray(X))
plt.rcParams.update({'figure.max_open_warning': 0})

   
db = DBSCAN(eps, min_samples,algorithm="kd_tree").fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
#print("these are labels",labels)

#print("These are the core sample indices",db.core_sample_indices_)

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
#print('Estimated number of clusters: %d' % n_clusters_)
#print('Estimated number of noise points: %d' % n_noise_)

# Black removed and is used for noise instead.
unique_labels = set(labels)
fig = plt.figure()
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)


plt.title('Estimated number of clusters: %d' % n_clusters_)

fig.savefig('cluster_image.png', bbox_inches='tight')

# extracting the reviews from the clusters

index2 = []
counter = 0
for i in labels:
    if i != -1: # remove the noise labels
        index2.append(counter)
    counter+=1

label2 = []
for j in index2:
    k = labels[j]
    label2.append(k) # these are the labels for points in each cluster


d = {"Index":index2, "Labels":label2}

df = pd.DataFrame(d)

s = pd.Series(["Labels"])


# extracting the indices of the labels in the cluster
index_kk = []

for i in range(len(label2)):
    p= df[df['Labels']==i]
    
    index_kk.append(p.Index.tolist())
#print("index for each label:",index_kk )
index_labels = [x for x in index_kk if x != []]
#print ("These are the indices for each label",index_labels)

# extracting the reviews in each cluster
empty_lst = []
for i in index_labels:
    empty_lst.append([cleaned_review_body[j] for j in i])

#print ("these are the reviews found in each cluster",empty_lst)	

for i in range(n_clusters_):
    #print("Cluster %d:" % i,' %s' % empty_lst[i])
    print("Cluster %d:" % i,' %s' % len(empty_lst[i]))

# all reviews in a cluster is now a single token
clust = []
for i in range(len(empty_lst)):
    t= ' '.join(empty_lst[i])
    clust.append(t)
#print("These are the reviews in a cluster",clust)

######################################################################


# building word-cluster count matrix

def fn_tdm_df(docs, xColNames = None):
    ''' create a term document matrix as pandas DataFrame
    with **kwargs you can pass arguments of CountVectorizer
    if xColNames is given the dataframe gets columns Names'''

    #initialize the  vectorizer
    vectorizer = CountVectorizer()
    x1 = vectorizer.fit_transform(clust)
    #create dataFrame
    df = pd.DataFrame(x1.toarray().transpose(), index = vectorizer.get_feature_names()) 
    
    if xColNames is not None:
        df.columns =  xColNames

    return df

for i in range(len(clust[:5])):
        
    count_matrix=fn_tdm_df(docs=clust, xColNames =None)
print("This is the count matrix",count_matrix.head())

# building word-cluster PMI matrix

def pmi(df):
    '''
    Calculate the positive pointwise mutal information score for each entry
    https://en.wikipedia.org/wiki/Pointwise_mutual_information
    We use the log( p(y|x)/p(y) ), y being the column, x being the row
    '''
    # Get numpy array from pandas df
    arr = df.as_matrix()

    # p(y|x) probability of each t1 overlap within the row
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T

    # p(y) probability of each t1 in the total set
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    # PMI: log( p(y|x) / p(y) )
    # This is the same data, normalized
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001
    _pmi = np.log(ratio)
    _pmi[_pmi < 0] = 0

    return _pmi




# PMI matrix into a data frame
c=pmi(count_matrix)
vectorizer = CountVectorizer()
x1 = vectorizer.fit_transform(clust)
PMI_df = pd.DataFrame(c, index = vectorizer.get_feature_names()) 

# extracting the labels of each cluster with a positive pmi score into a list
List = []
for c in range(len(empty_lst)):
    col=PMI_df[c]
    List.append(col[col>0].sort_values(ascending=False).to_dict())
#print('These are the label vectors for each cluster')

for i in range(n_clusters_):
    print("Cluster_labels %d:\n" % i,' %s' % List[i].keys())



now = time.time() #Time after it finished

print("It took: ", now-then, " seconds")


                                









