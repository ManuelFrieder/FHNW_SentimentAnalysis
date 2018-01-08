import pandas
import preprocessing as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.pipeline import Pipeline

# define language (de/fr/en) or "" for any
language = ""
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

#Generating the training and testing vectors
def prepareTrainingData(trainingData):
    X = []
    y = []
    
    ppTrainingData=preprocessing.processData(trainingData)
    
    for wordlist,sentiment,index in ppTrainingData:
        y.append(0 if (sentiment=='negative') else 1)
        X.append(' '.join(wordlist))
    return X, y

# training
X_train, y_train = prepareTrainingData(trainingData)

vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True, use_idf = True, ngram_range=(1, 2))
svm_clf =svm.LinearSVC(C=0.1)
vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
vec_clf.fit(X_train,y_train)

# classify
for index, row in testData.iterrows():
    ppTestData = ' '.join(preprocessing.processComment(row.text))
    testData.at[index,'classified'] = ('negative' if (vec_clf.predict([ppTestData])==0) else 'positive')

testData.to_csv(outputFile, sep=';', encoding='latin1')