import numpy as np

class NB_Sentiment_Classifier:
    def __init__(self, texts:list, sentiments:list) -> None:
        assert len(texts) == len(sentiments), 'Za svaki tekst postoji tačno jedan sentiment.'
        self.texts = texts
        self.sentiments = sentiments
        self.pos_word_counts = {} 
        self.neg_word_counts = {}
        self.text_counts = {'pos': 0, 'neg': 0} # broj ✅ tekstova i broj ❌ tekstova
        self.n_words = {'pos': 0, 'neg': 0} # ukupan broj ✅ reči i ukupan broj ❌ reči
        self.prior = {'pos': 0, 'neg': 0} # P(✅) i P(❌)

    def _preprocess(self, text:str) -> str:
        '''Preprocess and returns text.'''
        import re
        text = re.sub(r'[^\w\s]', '', text) # uklonimo znakove
        words = text.lower().split() # svedemo na mala slova i podelimo na reči
        return words
    
    def fit(self) -> None:
        '''Train a classifier.'''
        # pravimo tabelu ponavljanja za svaku rec - Bag-of-words
        for text, sentiment in zip(self.texts, self.sentiments):
            words = self._preprocess(text)
            for word in words: 
                if sentiment == 'pos': self.pos_word_counts[word] = self.pos_word_counts.get(word, 0) + 1
                if sentiment == 'neg': self.neg_word_counts[word] = self.neg_word_counts.get(word, 0) + 1

        # broj ✅ tekstova i broj ❌ tekstova
        self.text_counts['pos'] = len([s for s in self.sentiments if s=='pos'])
        self.text_counts['neg'] = len([s for s in self.sentiments if s=='neg'])

        # ukupan broj ✅ reči i ukupan broj ❌ reči
        self.n_words['pos'] = sum(self.pos_word_counts.values())
        self.n_words['neg'] = sum(self.neg_word_counts.values())
        
        # nadji P(✅) i P(❌)
        n_total_texts = sum(self.text_counts.values())
        self.prior['pos'] = self.text_counts['pos'] / n_total_texts
        self.prior['neg'] = self.text_counts['neg'] / n_total_texts


    def predict(self, text:str) -> tuple[float, float]:
        '''Returns a list of: [P(text|✅), P(❌|text)].'''
        words = self._preprocess(text)
        p_words_given_pos_sentiment = []
        p_words_given_neg_sentiment = []
        # iteraramo kroz sve reci u recenici i racunamo P(word|✅)) i P(word|❌)
        for word in words:
            # verovatnoća da se reč nađe u pozitivnoj recenziji
            p_word_given_pos = self.pos_word_counts.get(word, 0) + 1 / (self.n_words['pos'] + len(self.pos_word_counts)) # Laplace Smoothing
            p_words_given_pos_sentiment.append(p_word_given_pos)

            # verovatnoća da se reč nađe u negativnoj recenziji
            p_word_given_neg = self.neg_word_counts.get(word, 0) + 1 / (self.n_words['neg'] + len(self.neg_word_counts)) # Laplace Smoothing
            p_words_given_neg_sentiment.append(p_word_given_neg)

        # računamo P(text|✅) i P(text|❌) tako sto pomnozimo verovatnoću za svaku reč
        p_text_given_pos = np.prod(p_words_given_pos_sentiment)
        p_text_given_neg = np.prod(p_words_given_neg_sentiment)

        # iskoristimo Bajesovu formulu da nadjemo P(✅|text) i P(❌|text)
        p_text_is_pos = self.prior['pos'] * p_text_given_pos
        p_text_is_neg = self.prior['neg'] * p_text_given_neg
        
        return p_text_is_pos, p_text_is_neg


if __name__ == '__main__':
    reviews = {
    'The movie was great': 'pos',
    'That was the greatest movie!': 'pos',
    'I really enjoyed that great movie.': 'pos',
    'The acting was terrible': 'neg',
    'The movie was not great at all...': 'neg'}

    reviews_texts = list(reviews.keys())
    reviews_sentiments = list(reviews.values())
    clf = NB_Sentiment_Classifier(reviews_texts, reviews_sentiments)
    clf.fit()
    
    text = 'The movie was terrible, terrible, terrible,...'
    p_text_is_pos, p_text_is_neg = clf.predict(text)
    
    print(f'P(✅|{text}) = {p_text_is_pos:.5f}')
    print(f'P(❌|{text}) = {p_text_is_neg:.5f}')
    if p_text_is_pos > p_text_is_neg: print('Recenzija je pozitivna')
    else: print('Recenzija je negativna')