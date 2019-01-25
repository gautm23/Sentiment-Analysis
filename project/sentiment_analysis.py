import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import numpy as np
############################# Reading  the csv file for data/reviews #################################
data = pd.read_csv('./Data/product.csv', sep = ',', quotechar = '"', encoding = 'utf8',dtype={'Reviews':str},
    usecols=['Reviews'],nrows=20000,na_values=[' ','.', '??']       # Take any '.' or '??' values as NA
)
data.columns = [col.replace(' ', '_').lower() for col in data.columns]

################################# processing to remove the punctuations and storing the data in form of words    ############################
processing_data1=[]
for d in data['reviews']:
    processing_data1.append(TextBlob(str(d).replace('[^\w\s]','').replace(",", "").replace(".", "").replace("!", "").replace("?", "")      .replace(";", "").replace(":", "")
                                .replace("*", "").replace("(", "").replace(")", "").replace("/", "").lower()).split(" "))

##########################   Removing the commonly used words in a sentence and lemmatizing the words         ###############################
stop_words1=set(stopwords.words('english'))
stop_words1.add('I')
stop_words1.add('(')
stop_words1.add(')')
stop_words1.add(',')
rem=["won't","aren't","don't",'hadn''mightn','didn',"hadn't",'shouldn','no',"couldn't","doesn't",'nor', 'hasn',"shouldn't",'mustn','wasn','than',
     "shan't",'don','aren',"wasn't",'not','against', 'haven',"needn't","isn't",'weren',"wouldn't","weren't","you're", "mightn't","didn't","haven't","hasn't","mustn't"]

stop_words=[]
for w in stop_words1:
    if w not in rem:
        stop_words.append(w)

lemmatizer = WordNetLemmatizer()
processing_data2=[]
for a in processing_data1:
  temp=[]
  for b in a:
      if b not in stop_words:
       temp.append(lemmatizer.lemmatize(b,pos='a'))
  processing_data2.append(temp)
#################### Calculating the sentiment score of the sentences >=0 for positive and <0 for negative sentences ##################
processing_data_response=[]
processing_data3=[]
all_words=[]
aaaa=0
for d in processing_data2:
    sum=0
    for a in d:
        ppp=float(TextBlob(a).sentiment[0])
        sum += ppp

    processing_data_response.append(sum)
    processing_data3.append([d, processing_data_response[aaaa]])
    aaaa+=1

#######################  Storing the sentiment score  of reviews in sentiment_score.csv file #########################
kd=pd.DataFrame({'Reviews':data['reviews'],'Sentiment_score':processing_data_response})
kd.to_csv('./Data/Processed_Data/sentiment_score.csv', sep=',', encoding='utf-8', index=False, header=True,columns=['Reviews','Sentiment_score'])

################### Shuffling the data before selecting training and testing data set #################################
np.random.shuffle(processing_data3)
training_data=processing_data3[:15000]
testing_data=processing_data3[15000:]
classes=[1,0]
print("size of the training data set is: ",len(training_data))
print("size of the testing data set is: ",len(testing_data))
positive_words={}
negative_words={}

############################   Training of data using naive_bayes algorithm     ###############################
def train_naive_bayes(training, classes):
    """Given a training dataset and the classes that categorize
    each observation, return  vocabulary of unique words,
    log_prior_prob: a list of P(c), and loglikelihood: a list of P(fi|c)s
    for each word
    """
    #Initialize all_data[ci]: a list of all documents of class i
    #E.g. all_data[1] is a list of [reviews, sentiment_score] of class 1
    all_data = [[]] * len(classes)

    #Initialize no_doc[ci]: number of documents of class i
    no_doc = [None] * len(classes)

    #Initialize log_prior_prob[ci]: stores the prior probability for class i
    log_prior_prob = [None] * len(classes)

    #Initialize loglikelihood: loglikelihood[ci][wi] stores the likelihood probability for wi given class i
    loglikelihood = [None] * len(classes)
    # Partition documents into classes. all_data[0]: negative docs, all_data[1]: positive docs
    for obs in training:  # obs: a [review, sentiment_score] pair
        # if sentiment_score >=0, classify the review as positive
        if obs[1] >=0:
           all_data[1] = all_data[1] + [obs]  # Can also write as all_data = all_data.append(obs)
        # else, classify review as negative
        else :
            all_data = all_data + [obs]
    vocabulary= []
    for dd in training:
        if dd[1]>=0:
           for word in dd[0]:
               if word not in vocabulary:
                   vocabulary.append(word)
               if positive_words.get(word)is not None:
                   positive_words[word]=positive_words[word]+1
               else:
                    positive_words[word]=1
        else:
            for word in dd[0]:
                if word not in vocabulary:
                    vocabulary.append(word)
                if negative_words.get(word)is not None:
                   negative_words[word] += 1
                else:
                    negative_words[word]=1

    vocabulary_size = len(vocabulary)
    total = len(training)
    for index in range(len(classes)):
        # Store no_doc value for each class
        no_doc[index] = len(all_data[index])

        # Compute P(c)
        log_prior_prob[index] = np.log((no_doc[index] + 1) / total)

        # Counts total number of words in class c
        count_w_in_vocabulary = 0
        for d in all_data[index]:
            count_w_in_vocabulary = count_w_in_vocabulary + len(d[0])
        denom = count_w_in_vocabulary + vocabulary_size

        dict = {}
        # Compute P(w|c)
        for wi in vocabulary:
            # Count number of times wi appears in all_data[index]
            count_wi_in_all_data = 0
            for d in all_data[index]:
                for word in d[0]:
                    if word == wi:
                        count_wi_in_all_data = count_wi_in_all_data + 1
            numer = count_wi_in_all_data + 1
            dict[wi] = np.log((numer) / (denom))
        loglikelihood[index] = dict

    return (vocabulary, log_prior_prob, loglikelihood)

p,q,r=train_naive_bayes(training_data,classes)

############################  Testing of data  ###################################################
def test_naive_bayes(test_docs, log_prior, log_likeli_hood, vocabulary):
    # Initialize log_post_prob[index]: stores the posterior probability for class index
  count = 0

  for ddd in test_docs:
    log_post_prob = [None] * len(classes)
    for index in classes:
        sumloglikelihoods = 0
        for word in ddd:
            if word in vocabulary:
                # This is sum represents log(P(w|c)) = log(P(w1|c)) + log(P(wn|c))
                sumloglikelihoods += log_likeli_hood[index][word]

        # Computes P(c|d)
        log_post_prob[index] = log_prior[index] + sumloglikelihoods

    # Return the class that generated max ĉ
    pp = 0
    if ddd[1] >= 0:
       pp = 1
    if log_post_prob.index(max(log_post_prob)) == pp:
           count += 1


  return count * 100 / len(test_docs)
pp=test_naive_bayes(testing_data,q,r,p)
print("The accuracy percentage of the naive baye's classifier is: ",pp)
############## Storing the words with their frequency in different polarity(i.e positive or negative polarity ) ########################
kd=pd.DataFrame({'Positive_words':list(positive_words.keys())[:],'Frequency':list(positive_words.values())[:]})
kd.to_csv('./Data/Processed_Data/positive_words.csv', sep=',', encoding='utf-8', index=False, header=True,columns=['Positive_words','Frequency'])
kd=pd.DataFrame({'Negative_words':list(negative_words.keys())[:],'Frequency':list(negative_words.values())[:]})
kd.to_csv('./Data/Processed_Data/negative_words.csv', sep=',', encoding='utf-8', index=False, header=True,columns=['Negative_words','Frequency'])
