import nltk
# import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
# from nltk.corpus import wordnet
import pickle
'''POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent\'s
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''
with open('RawText.txt', 'r') as data:
    rawtext = data.read()
for line in sent_tokenize(rawtext):
    # print(line)
    pass
# print(word_tokenize(rawtext))
stop_words = set(stopwords.words("english"))
print(f'Stop Words: \n *************** \n{stop_words}')
words = word_tokenize(rawtext)
filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
    for word in filtered_sentence:
          #TODO print results in file to reuse it 
        pass

train_text = state_union.raw("2005-GWBush.txt")
sample_text = rawtext
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(rawtext)
# chunked =[]


def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt= nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()
            chunk_gram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}
                                             }<VB.?|IN|DT|TO>+{"""
            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree,'\n','---------------------------')
                chunked.draw()

    except Exception as e:
        print(str(e))

# process_content()


# set that we'll train our classifier with
training_set = train_text
# set that we'll test against.
testing_set = sample_text
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()