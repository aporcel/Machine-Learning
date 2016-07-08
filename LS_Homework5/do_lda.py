import pandas as pd
import numpy as np
import lda
import matplotlib.pyplot as plt
import operator
import sys
import time

# this function randomly chooses n_per_doc words per document to hold-out
# DO NOT MODIFY THIS FUNCTION
def choose_heldout_words(X, n_per_doc=5):
    heldout_dict = {}
    np.random.seed(0) # set random seed to zero so that everyone is using the same
                      # held-out words 
    n_heldout = 0

    for row in range(X.shape[0]): # each row of X corresponds to a document
        words = []
        for word in range(X.shape[1]): # each column of X corresponds to a specific word
            if X[row,word]==1:         # we're only considering words that appear exactly once
                words.append(word)     

        heldout_words = []
        for _ in range(n_per_doc):
            heldout_word = words[np.random.randint(0,len(words))]
            heldout_words.append(heldout_word)

        # because all the random draws above are done independently, we might get repeats
        # so here we get rid of repeats
        heldout_words = np.unique(heldout_words)
        for heldout_word in heldout_words:
            X[row, heldout_word] -= 1 # remove held-out word from the X used for training

        heldout_dict[row] = heldout_words 
        n_heldout += len(heldout_words)

    print "held-out a total of %d words" % n_heldout 

    # we return two objects:
    # first we return a word-doc co-occurence matrix with the held-out words removed (X).
    # then we return a dictionary that encodes which words were held-out
    # in each document (heldout_dict).
    # each key in the dict corresponds to a document.
    # each value in the dict is a list of the held-out words (represented by their index
    # in the vocabulary, i.e. the column they correspond to in X).
    return X, heldout_dict

#######################################
#           ##           ##           #
#   TO DO   ##   TO DO   ##   TO DO   #
#           ##           ##           #
#######################################
# this function is supposed to return the log likelihood on the held out test words
# note that we pass it the fitted model and the dictionary which encodes which
# words were held-out.
# i've included a few lines of code to get you started.
# you need to fill in the loop by calculating the log likelihood of each held-out word
def calculate_log_likelihood_on_heldout_words(model, heldout_dict):
    ll = 0.0
    for row in heldout_dict: # this loops through all the documents
        for word in heldout_dict[row]: # this loops through all the held-out words
            sum_ = 0
            for t in range(0,model.n_topics):      # this loops through all the topics
                theta = model.doc_topic_[row,t]    # Obtaining Theta, row: Document, t: topic
                beta = model.components_[t,word]   # Obtaining Beta, t: topic, word: word
                sum_ = sum_ + beta * theta
            # now you just need to extract the appropriate components of theta and beta from
            # the model and add the appropriate term to the loglikelihood ll.
            # in particular you need to access the matrix model.components_[] as well
            # as the matrix model.doc_topic_[]
            ll += np.log(sum_) + 0.0 ### TO COMPLETE !!! 

    return ll

print "******************************************"
print "* DOING DATA INGESTION AND PREPROCESSING *"
print "******************************************"

# here we read in the preprocessed reuters data, in particular the doc-word co-occurence matrix X
X = lda.datasets.load_reuters()
print "loaded doc-word co-occurence matrix" 
print "there are %d words in the entire corpus" % np.sum(X)
n_docs = X.shape[0]
print "there are %d documents in the corpus" % n_docs

# here we load the unique words used in the corpus, i.e. the vocabulary
vocab = lda.datasets.load_reuters_vocab()
print "there are %d words in the vocabulary" % len(vocab)

# here we randomly choose which words we're going to hold-out
n_per_doc = 10 # do not change
X, heldout = choose_heldout_words(X, n_per_doc=n_per_doc)
print "after holding out %d words per doc there are %d words in the training corpus" % (n_per_doc,
       np.sum(X) )

# the number of iterations we do model fitting (do not change)
n_iter = 1200

print "\n**********************************************"
print "*                 DOING INFERENCE            *"
print "**********************************************"

n_topicslist = [4,8,12,16,20,24]
ll_obtained = []
for n_topics in n_topicslist:
    t_inf_start = time.time()
    print "fitting model with n_topics=%02d" % n_topics
    # here we define the model and do fitting
    # do not change the hyperparametersi alpha/eta or random_state
    model = lda.LDA(n_topics=n_topics, random_state=0, n_iter=n_iter, alpha=0.1, eta=0.01)
    model.fit(X) 
    # using the model we calculate the log likelihood of the held-out words
    # (at least once you've implemented this function!)
    ll_test = calculate_log_likelihood_on_heldout_words(model, heldout)
    t_inf_end = time.time()
    print "model fitting took a total of %.1f seconds, with a test of %s" % (t_inf_end-t_inf_start,ll_test)
    ll_obtained.append(ll_test)

#Plotting Log Likelihood vs. Number of Topics
plt.figure(figsize=(8,8))
plt.plot(n_topicslist,ll_obtained,'r-',marker='o')
plt.title('Log Likelihood per Number of Topics',fontsize=16)
plt.xlabel('Number of Topics',fontsize=14)
plt.ylabel('Log Likelihood',fontsize=14)
plt.xticks([0,4,8,12,16,20,24])
#plt.ylim([-30550,-30250])
plt.grid()
plt.show();

# Selecting the best N as the largest:
n_topics = n_topicslist[np.array(ll_obtained).argsort()[-1:][::-1][0]]
t_inf_start = time.time()
print "fitting model with n_topics=%02d, as the best one" % n_topics
# here we define the model and do fitting
# do not change the hyperparametersi alpha/eta or random_state
model = lda.LDA(n_topics=n_topics, random_state=0, n_iter=n_iter, alpha=0.1, eta=0.01)
model.fit(X) 
t_inf_end = time.time()
print "model fitting took a total of %.1f seconds" % (t_inf_end-t_inf_start)
# using the model we calculate the log likelihood of the held-out words
# (at least once you've implemented this function!)
ll_test = calculate_log_likelihood_on_heldout_words(model, heldout)

print "Topics' 5 most used words in best model"
for t in range(0,model.n_topics):
    words = 'topic %d: ' % (t)
    for wi in model.components_[t].argsort()[-5:][::-1]:
        words = words + vocab[wi] + ','
    print words[:-1]

#################
### TO ANSWER ###
#################

### for what value of n_topics do you get the best model ??
# I get the best model for the value of n_topics = 8

### what are the top 5 words in each topic in the best model ??
# In the best model, the top 5 words in each topic are:
# topic 0: charles,prince,king,diana,royal
# topic 1: church,people,life,years,n't
# topic 2: last,government,church,country,political
# topic 3: elvis,police,music,east,fans
# topic 4: mother,teresa,order,heart,charity
# topic 5: pope,vatican,surgery,john,pontiff
# topic 6: harriman,u.s,yeltsin,clinton,churchill
# topic 7: city,germany,german,french,century

### which topic makes up the largest proportion of articles 77 and 78 ??
# topic 3  makes the largest proportion for article 77
# topic 3  makes the largest proportion for article 78

titles = lda.datasets.load_reuters_titles()
print "\narticle #77:", titles[77],"\n topic ",model.doc_topic_[77].argsort()[-1:][::-1][0]," makes the largest proportion"
print "\narticle #78:", titles[78],"\n topic ",model.doc_topic_[78].argsort()[-1:][::-1][0]," makes the largest proportion"
