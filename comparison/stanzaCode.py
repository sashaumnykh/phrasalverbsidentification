import stanza

# Download the language model
stanza.download('en')

import time

t = open("../texts/thePerks.txt", "r")
text = t.read()
t0 = time.time()
nlp = stanza.Pipeline('en', processors="tokenize,mwt,pos,lemma,depparse")

sentence = text

# Build a Neural Pipeline


# Pass the sentence through the pipeline
doc = nlp(sentence)

# Print the dependencies of the first sentence in the doc object
# Format - (Token, Index of head, Nature of dependency)
# Index starts from 1, 0 is reserved for ROOT
#doc.sentences[0].print_dependencies()

#print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for word in sent.words], sep='\n')
#print(*[f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats if word.feats else "_"}' for sent in doc.sentences for word in sent.words], sep='\n')
#print("-" * 50)

print("{:<15} | {:<15} | {:<10} | {:<15} ".format('Token', 'POS', 'Relation', 'Head'))
print("-" * 50)

# Convert sentence object to dictionary

for sent in doc.sentences:
    sent_dict = sent.to_dict()
# iterate to print the token, relation and head
    for word in sent_dict:
        print("{:<15} | {:<15} | {:<10} | {:<15} "
              .format(str(word['text']), str(word['upos']), str(word['deprel']),
                      str(sent_dict[word['head'] - 1]['text'] if word['head'] > 0 else 'ROOT')))

print (time.time() - t0, "время обработки в секундах")