import nltk 
import pandas
import preprocessing as pp

# define language (de/fr/en) or "" for any
language = "en"
preprocessing = pp.PreProcessData(language)

# file paths
corpusFile = '/path/to/trainingData.csv'
testFile = '/path/to/testData.csv'
outputFile = '/path/to/destination/file_'+language+'.csv'

# read files =>check encoding and delimiter!
allTrainingData = pandas.read_csv(corpusFile, encoding='UTF-8', delimiter=';')
allTestData = pandas.read_csv(testFile, encoding='UTF-8', delimiter=';', index_col=0)

# language filter
trainingData = preprocessing.langSelection(allTrainingData,language)
testData = preprocessing.langSelection(allTestData,language)
 
def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment,index) in ppTrainingData:
        all_words.extend(words)
    wordlist=nltk.FreqDist(all_words) # list with each word and its frequency
    word_features=wordlist.keys() # list of unique words in the corpus 
    return word_features

def extract_features(text):
    text_words=set(text)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in text_words)
    return features 

# preprocessing
ppTrainingData=preprocessing.processData(trainingData)
ppTestData=preprocessing.processData(testData)

# Extract features and train the classifier
word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)
NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)

# classify
for t in ppTestData:
    testData.at[t[2],'classified'] = NBayesClassifier.classify(extract_features(t[0]))

testData.to_csv(outputFile, sep=';', encoding='latin1')