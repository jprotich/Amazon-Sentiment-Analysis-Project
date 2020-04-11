import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from contractions import CONTRACTION_MAP
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist, classify, NaiveBayesClassifier
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import re, string, random, pprint
import numpy as np

font = {'weight': 'bold'}

data = pd.read_csv('electronic_products.csv', skip_blank_lines=True, dtype='unicode')
data.drop_duplicates(inplace=True)
data.to_csv('electronic_products.csv', index=False)

#Removing certain columns from dataset
data.drop("id", axis=1, inplace=True)
data.drop("asins", axis=1, inplace=True)
data.drop("categories", axis=1, inplace=True)
data.drop("colors", axis=1, inplace=True)
data.drop("dateAdded", axis=1, inplace=True)
data.drop("dateUpdated", axis=1, inplace=True)
data.drop("dimension", axis=1, inplace=True)
data.drop("ean", axis=1, inplace=True)
data.drop("imageURLs", axis=1, inplace=True)
data.drop("manufacturer", axis=1, inplace=True)
data.drop("manufacturerNumber", axis=1, inplace=True)
data.drop("name", axis=1, inplace=True)
data.drop("primaryCategories", axis=1, inplace=True)
data.drop("reviews.date", axis=1, inplace=True)
data.drop("reviews.dateSeen", axis=1, inplace=True)
data.drop("reviews.doRecommend", axis=1, inplace=True)
data.drop("reviews.numHelpful", axis=1, inplace=True)
data.drop("reviews.sourceURLs", axis=1, inplace=True)
data.drop("reviews.username", axis=1, inplace=True)
data.drop("sourceURLs", axis=1, inplace=True)
data.drop("upc", axis=1, inplace=True)
data.drop("weight", axis=1, inplace=True)
data.drop("keys", axis=1, inplace=True)

#Removing NA types/empty rows that would cause errors during reading
data.dropna(how="all", inplace=True)

#Dealing with contractions, Ex: don't -> dont
#Resource used: https://www.linkedin.com/pulse/processing-normalizing-text-data-saurav-ghosh/
def expand_contractions(text):
    pattern = re.compile("({})".format("|".join(CONTRACTION_MAP.keys())),flags = re.DOTALL| re.IGNORECASE)
    
    def replace_text(t):
        txt = t.group(0)
        if txt.lower() in CONTRACTION_MAP.keys():
            return CONTRACTION_MAP[txt.lower()]
        
    expand_text = pattern.sub(replace_text,text)
    return expand_text 

singleReviews = []
review_tokens = []
#EDITING DATA FORMAT AND SPLITTING UP INTO SPECIFIC LISTS
for i in range(len(data.index)):
    #Turning into single reviews (combine title and reviews); Expanding contractions
    singleReviews.append(expand_contractions(str(data['reviews.title'][i]) + " " + str(data['reviews.text'][i])))

#Tokenization
for t in range(len(singleReviews)):   
    white_space_tokenizer = nltk.WhitespaceTokenizer()
    review_tokens.append(white_space_tokenizer.tokenize(str(singleReviews[t])))

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def remove_noise(text):
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    text = [re.sub('[^a-zA-Z]+', '', word) for word in text]
    stop_words = set(stopwords.words('english'))
    stop_words.remove('no')
    stop_words.remove('not')
    text = [x for x in text if x not in stop_words]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return(text)

for i in range(len(review_tokens)):
    review_tokens[i] = remove_noise(str(review_tokens[i]))
    filtered_reviews = list(filter(None, review_tokens))

data['reviews.rating'] = data['reviews.rating'].astype(int)

#Preparing data to be fed into training and testing models for Naive Bayes Classifier

#Sorting original reviews into categorized lists by rating
positiveRatings = []
negativeRatings = []
neutralRatings = []
trueRatings = []

for i in range(len(filtered_reviews)):
    if(data['reviews.rating'][i] == 5 or data['reviews.rating'][i] == 4):  
        positiveRatings.append(filtered_reviews[i])
        trueRatings.append('positive')
    elif(data['reviews.rating'][i] == 3):
        neutralRatings.append(filtered_reviews[i])
        trueRatings.append('neutral')
    else:
        negativeRatings.append(filtered_reviews[i])    
        trueRatings.append('negative')

#############################################################################################
#Naive Bayes Classifier
def features(words):
    return dict([(word, True) for word in words.split()])

pos_data = [(features(f), 'positive') for f in positiveRatings]
ntr_data = [(features(f), 'neutral') for f in neutralRatings]
neg_data = [(features(f), 'negative') for f in negativeRatings]

classifyingData = pos_data + ntr_data + neg_data
random.shuffle(classifyingData)

train_data = classifyingData[:int((len(classifyingData)/2))]
test_data = classifyingData[int((len(classifyingData)/2)):]
classifier = NaiveBayesClassifier.train(train_data)

print(classifier.show_most_informative_features(20)) 

nbPos = 0
nbNtr = 0
nbNeg = 0
nbScores = []

for review in filtered_reviews:
    nbScore = classifier.classify(features(review))
    if(nbScore == 'positive'):
        nbScores.append('positive')
        nbPos += 1
    elif(nbScore == 'neutral'):
        nbScores.append('neutral')
        nbNtr += 1
    else:
        nbScores.append('negative')
        nbNeg += 1

dlen = len(classifyingData)
print("\nNaive Bayes Results\n")
print('Actual Positive: {:<20} NB Positive: {:<20}'.format(len(positiveRatings)/dlen, nbPos/dlen))
print('Actual Neutral: {:<20} NB Neutral: {:<20}'.format(len(neutralRatings)/dlen, nbNtr/dlen))
print('Actual Negative: {:<20} NB Negative: {:<20}'.format(len(negativeRatings)/dlen, nbNeg/dlen))
print("\nAccuracy is:", classify.accuracy(classifier, test_data))

###################################################################################################
#SVM
filtered_tokens = [review.split() for review in filtered_reviews]

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
trainingData = vectorizer.fit_transform(filtered_reviews[:3500])
testingData = vectorizer.transform(filtered_reviews[3500:])

# Perform classification with SVM, kernel=linear
classifier = SVC(kernel='linear')
classifier.fit(trainingData, trueRatings[:3500])
predictTesting = classifier.predict(testingData)

print("\nLSVM Results\n")
report = classification_report(trueRatings[3500:], predictTesting, output_dict=True)
print('positive: ', report['positive'])
print('neutral: ', report['neutral'])
print('negative: ', report['negative'])
print("Accuracy is:", str(classifier.score(testingData, trueRatings[3500:])))

vectorizedReviews = []
for review in filtered_reviews:
    review_vector = vectorizer.transform([review]) # vectorizing
    vectorizedReviews.append(classifier.predict(review_vector))

##########################################################################################################
#VADER - sentiment lexicon
analyser = SentimentIntensityAnalyzer()

#Removing duplicates of brands
allBrands = dict.fromkeys(data['brand'])

for k, v in allBrands.items():
    if v is None:
        allBrands[k] = int('0')

vaderScores = []
vPos = 0
vNeg = 0
vNtr = 0
#Compound threshold +/- 0.3
for r in range(len(filtered_reviews)):
    score = analyser.polarity_scores(str(filtered_reviews[r]))
    total = score['compound']
    allBrands[data['brand'][r]] += total
    if score['compound'] >= 0.3: 
        vaderScores.append("positive") 
        vPos+=1
    elif score['compound'] <= - 0.3: 
        vaderScores.append("negative") 
        vNeg+=1
    else: 
        vaderScores.append("neutral")
        vNtr+=1

#Calculating accuracy
match = 0
for i in range(len(trueRatings)):
    if(trueRatings[i] == vaderScores[i]):
        match += 1

print("\nVADER Results\n")
print('Actual Positive: {:<20} VADER Positive: {:<20}'.format(len(positiveRatings)/dlen, vPos/dlen))
print('Actual Neutral: {:<20} VADER Neutral: {:<20}'.format(len(neutralRatings)/dlen, vNtr/dlen))
print('Actual Negative: {:<20} VADER Negative: {:<20}'.format(len(negativeRatings)/dlen, vNeg/dlen))
print("Accuracy is:", str(match/dlen))

allScores = []
#Calculating Averages for Brand Positivity Results
brandCount = Counter(data['brand'])
for k,v in allBrands.items():
    allScores.append((allBrands[k]/brandCount[k])*100)
    allBrands[k] = round(allBrands[k]/brandCount[k]*100)
    
#Sorting by Percents
allBrands = OrderedDict(sorted(allBrands.items(), key=lambda x: x[1], reverse=True))

print('{:<40} {:<40}'.format("\nBrand Name", "Positivity (In Percent Form)"))
print("-----------------------------------------------------------------------------")
count = 0
for k,v in allBrands.items():
    allBrands[k] = str(allBrands[k]) + "%"
    print('{:<55} {:<55}'.format(k, allBrands[k]))
    count += 1

xValues = [0] * len(allScores)
for i in range(len(xValues)):
    xValues[i] = i

plt.style.use('seaborn-whitegrid')
plt.scatter(xValues, allScores, marker = 'o', edgecolors='b')
plt.ylabel("Positivity (out of 100)")
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
plt.title("Positivity of All Electronic Brands")
plt.ylim(0,100)
plt.xlabel("Brands")
plt.show()



