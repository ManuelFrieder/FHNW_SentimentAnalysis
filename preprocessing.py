# Class to preprocess test and training data

class PreProcessData:
    def __init__(self, lang):
        from string import punctuation 
        from nltk.corpus import stopwords 
        from nltk.stem.snowball import SnowballStemmer

        self.stopwords=set(list(punctuation)+['at_us','url','rt'])

        if lang == "de":
            self.language = 'german'
        elif lang == "fr":
            self.language = 'french'
        elif lang == "en":    
            self.language = 'english'

        try:
            self.stemmer = SnowballStemmer(self.language, ignore_stopwords=True)
            self.stopwords.update(stopwords.words(self.language))
        except AttributeError:
            self.german_stemmer = SnowballStemmer('german', ignore_stopwords=True)
            self.french_stemmer = SnowballStemmer('french', ignore_stopwords=True)
            self.english_stemmer = SnowballStemmer('english', ignore_stopwords=True)
            self.stopwords.update(stopwords.words('german')+stopwords.words('french')+stopwords.words('english'))

    def processData(self,list_of_comments):
        processedComments=[]
        # List of tuples. Each tuple is a comment which is a list of words and its label
        for index, row in list_of_comments.iterrows():
            if row.classified == "neutral": #ignore neutral comments
                continue
            else:
                processedComments.append((self.processComment(row.text),row.classified,index))
        return processedComments
    
    def processComment(self,comment):
        import re
        from nltk.tokenize import wordpunct_tokenize
        
        # Convert to lower case
        comment=str(comment).lower()
        # Replace links with "URL" 
        comment=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',comment)     
        # Replace @username with "AT_USER"
        comment=re.sub('@[^\s]+','AT_USER',comment)
        # Replace #word with word 
        comment=re.sub(r'#([^\s]+)',r'\1',comment)
        # Tokenize the comment into a list of words 
        comment=wordpunct_tokenize(comment)
        # Stemming
        ppcomment = comment
        try:
            comment[:] = [self.stemmer.stem(word) for word in comment]
        except AttributeError:
            comment[:] = [self.german_stemmer.stem(word) for word in comment]
            comment[:] = [self.french_stemmer.stem(word) for word in comment]
            comment[:] = [self.english_stemmer.stem(word) for word in comment]
        # return the list without stopwords 
        return [word for word in ppcomment if word not in self.stopwords]

    def langSelection(self,allData,language):
        if language != "": #select only the defined language
            langData = allData.lang == language
            allData = allData[langData]
        return allData

    def filterNeutral(self,allData):
        selection = allData[allData.classified != 'neutral']
        selection = allData[selection]
        return selection