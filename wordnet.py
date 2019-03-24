from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.wsd import lesk

print(wn.synsets('dogs'))
print(wn.synsets('running'))

dog = wn.synset('dog.n.01')
print(dog.definition())

dog = wn.synset('run.n.05')
print(dog.definition())

synonyms = []
antonyms = []

for syn in wn.synsets("active"):
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
		    antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


sent = 'people should be able to marry a person of their choice'.split()

for i in range(len(sent)):
    word = sent[i]
    print(lesk(sent, word))