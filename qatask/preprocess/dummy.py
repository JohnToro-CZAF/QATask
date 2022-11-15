def preprocess(article):
    wikipage = article['title']
    wikipage = 'wiki/' + wikipage.replace(' ', '_')
    return {'id': article['id'], 'text': article["text"], 'wikipage': wikipage}