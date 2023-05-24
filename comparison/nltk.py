import time

t = open("texts/thePerks.txt", "r")
text = t.read()

t0 = time.time()

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

tokens = nltk.word_tokenize(text)
res = nltk.pos_tag(tokens)
print(res)
print(len(tokens), "количество тэгов")

print (time.time() - t0, "время обработки в секундах")
#print(res)