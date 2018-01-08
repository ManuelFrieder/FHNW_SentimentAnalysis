import pandas
tweet_ids = '/.../canephore-corpus/tweets-ids' #adjust path
tweets = pandas.read_csv(tweet_ids, encoding="latin1", names = ['id', 'text', 'sentiment'])

pandas.set_option('display.precision',4)
collected = tweets
for index, row in tweets.iterrows():
    try:
        file='/path/to/retrievedTweets/folder/'+str(int(row.id))+'.txt' #adjust path
        collected.loc[[index], 'text'] = value=open(file,'r').read()
        annFile = '/.../canephore-corpus/tweets-annotations/all/'+str(int(row.id))+'.ann' #adjust path
        annotation = pandas.read_table(annFile, delimiter='\t', header=None)
        compare = pandas.DataFrame(annotation.iloc[:,1].str[2:10].value_counts())
        compare = compare.transpose()
        try:
            compare.iloc[0]['Positive']
            compare.iloc[0]['Negative'] 
        except KeyError as err:
            if str(err) == "'Negative'":
                collected.loc[[index], 'sentiment'] = 'positive'
            else:
                try:
                    if compare.iloc[0]['Positive'] > compare.iloc[0]['Negative']:
                        collected.loc[[index], 'sentiment'] = 'positive'
                    elif compare.iloc[0]['Positive'] < compare.iloc[0]['Negative']:
                        collected.loc[[index], 'sentiment'] = 'negative'
                    else:
                            collected.loc[[index], 'sentiment'] = 'neutral'
                except KeyError:
                    if str(err) == "'Positive'":
                        collected.loc[[index], 'sentiment'] = 'negative'
                    else:
                        collected.loc[[index], 'sentiment'] = 'empty'
    except: 
        continue
        
collected.to_csv('/export/path/frenchCorpusFile.csv', sep=';', encoding='utf-8') #ajdust path