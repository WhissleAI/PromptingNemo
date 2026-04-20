import nltk
nltk.download('punkt_tab')

from nltk import tokenize

def nmt_large_text_split(input_text, max_chars = 615):
    """Function to split large input text"""
    
    def nmt_text_split_sentence_splitter(sentence_text, max_chars):
        """Function to split a sentence while respecting word boundaries, if sentence length > max_chars"""
        sentence_splits = []
        if len(sentence_text) > max_chars:
            words = sentence_text.split()
            for word in words:
                if len(sentence_splits) > 0 and (len(sentence_splits[-1]) + len(word) <= max_chars):
                    sentence_splits[-1] += word
                else:
                    sentence_splits.append(word)
        else:
            sentence_splits.append(sentence_text)
        return sentence_splits
    
    sentences = tokenize.sent_tokenize(input_text) 
    nmt_input_texts = []
    for i in range(len(sentences)):
        sentence_splits = nmt_text_split_sentence_splitter(sentences[i], max_chars)
        sentences = sentences[:i] + sentence_splits + sentences[i+1:]
        if len(nmt_input_texts) > 0 and (len(nmt_input_texts[-1]) + len(sentences[i]) <= max_chars):
            nmt_input_texts[-1] += sentences[i]
        else:
            nmt_input_texts.append(sentences[i])    
    return nmt_input_texts