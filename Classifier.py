import pandas as pd
import numpy as np
#import nltk
import re
#nltk.download()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import NaiveBayes as nb

train = pd.read_csv("train2.tsv", header=0, delimiter="\t", quoting=3)

# removes punctuation and other characters from a review
# removes stopwords from the review
def review_to_words(raw_review):
    r = re.sub("[^a-zA-Z]", " ", raw_review)
    r = r.lower()
    words = r.split()

    words = [w for w in words if not w in stopwords.words("english")]
    return(" ".join(words))

# clean all reviews in an array
def clean_all(reviews):
    num_reviews = reviews["Phrase"].size
    clean_reviews = []
    
    for i in range(0, num_reviews):
        if( (i+1)%1000 == 0 ):
            print("Cleaned %d/%d Reviews" % ( i+1, num_reviews ))
        clean_reviews.append(review_to_words(reviews["Phrase"][i]))
    return(clean_reviews)


# -- process the training set --
clean_reviews = clean_all(train)

# create a vocabulary from the clean reviews using Scikit-Learn
vec = CountVectorizer(analyzer = "word", tokenizer = None, \
                             preprocessor = None, stop_words = None, \
                             max_features = 5000) # change max features to change vocab size
    
features = vec.fit_transform(clean_reviews)
features = features.toarray()
vocab = vec.get_feature_names()


# -- print info for debugging --
#print(features.shape)
#print(vocab)


# -- train the classifier that uses Multinomial Naive Bayes from NLTK --
classifier = MultinomialNB()
classifier.fit(features, train["Sentiment Value"])


# -- classify the test set --
# change which test set to use by commenting/uncommenting
#test = pd.read_csv("test-train.tsv", header=0, delimiter="\t", quoting=3) # 49 sentences
test = pd.read_csv("unmarked-include.csv", header=0, delimiter="\t", quoting=3) # 347 sentences

clean_test = clean_all(test)
num_test = len(clean_test)

# get array of features using the vocabulary
test_features = vec.transform(clean_test)
test_features = test_features.toarray()

# predict the sentiment score using the NB classifier from Scikit-Learn
result = classifier.predict(test_features)

# print statistics of result of testing
def result_stats(result, num_test):
    result_diff = result - test["Sentiment Value"]
    # -- print results of classification --
    for i in range(0, num_test):
        print(test["Phrase"][i])
        print("Classified: %d, Actual: %d, Difference: %d" %
              (result[i], test["Sentiment Value"][i], result_diff[i]))
    result_diff = abs(result_diff)
    
    count = 0
    for i in result_diff:
        if i == 0:
            count = count+1
            
    result_diff = np.sum(result_diff)
    result_diff = result_diff/num_test
    print("Average difference in scores: %f" % (result_diff))
    print("Percent Correct: %f" % (count/num_test * 100))

result_stats(result, num_test)

print("------------------------------\nNow our own NB\n")
# -- train the classifier using our own Naive Bayes --
classes = [0, 1, 2, 3, 4] # classes to classify into
classifier2 = nb.NaiveBayes(clean_reviews, train["Sentiment Value"],
                             classes, vocab, clean_test, test["Sentiment Value"])
