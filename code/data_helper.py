import os
import xlrd
import json
import re
import numpy as np 
import codecs
from nltk import FreqDist
import nltk
from collections import Counter
import itertools

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    if len(sentences)>1:
        sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def load_data(_filename, n_classes):
    binary = False
    if n_classes==2:
        binary= True
        n_classes = 3
    
    tmp_x=[];tmp_y=[]
    
    workbook = xlrd.open_workbook(_filename)
    
    worksheet = workbook.sheet_by_index(0)
    # preprocess the data
    for i in range(0, len(worksheet.col(0))):
        tmp=json.dumps(worksheet.cell(i, 0).value)
        tmp = re.sub('"', '', tmp)
        tmp =re.sub('-', '',tmp)
        tmp = re.sub(r'\d+', '',tmp)
        tmp=tmp.lower()
        tmp_x.append(tmp)
        tmp_y.append(worksheet.cell(i, n_classes-1).value)
    
    if binary:
       idx= find_indices(tmp_y, lambda e: e == 0 or e==2)
       tmp_x = [tmp_x[i] for i in idx]
       tmp_y =[tmp_y[i] for i in idx]
       tmp_y = [1 if y == 2 else 0 for y in tmp_y]
       
    if n_classes==5:
        tmp_y = [t-1 for t in tmp_y]
    return tmp_x,tmp_y


def maxLength(_x):
    # get the average for each sequence
    list =[]
    for _doc in _x:
       for j in _doc:
           list.append(len(j))
    return np.max(list)//1
        
def maxSentences(_x):
    list = []
    for _doc in _x:
        list.append(len(_doc))
    return np.max(list)//1

def avgLength(_x):
    # get the average for each sequence
    list =[]
    for _doc in _x:
       for j in _doc:
           list.append(len(j))
    return np.average(list)//1
    
def avgSentences(_x):
    list = []
    for _doc in _x:
        list.append(len(_doc))
    return np.average(list)//1

def seq_sent_doc_lengths(x,max_sentences):
    tmp_doc =[]
    for i in range(len(x)):
        tmp_s = x[i]
        tmp_sent =[]
        for j in range(len(tmp_s)):
            tmp_sent.append(len(tmp_s[j]))
        padded_zeros=np.array(np.zeros(max_sentences-len(tmp_sent)))
        tmp_padded_doc = np.concatenate((np.array(tmp_sent) ,padded_zeros),axis=0)
        tmp_doc.append(tmp_padded_doc)
    return tmp_doc
    
def split_load_data_sentences(_x):
    # store document with sentences
    tmp_x = [] 
    # store text only
    _text=''
    for document in range(len(_x)):
        # split document into sentences
        sentences = split_into_sentences(_x[document])
        # store temp sentences
        tmp_sentences = []
        # clear sentences 
        for s in range(len(sentences)):
            # get only the words in the sentence
            tmp=re.sub('\W+',' ', sentences[s]).strip()
            # avoid empty sentences
            if len(tmp)>0:
                # store text only
                _text +=tmp + ' '  
                # store split sentence
                tmp_sentences.append(tmp)
        # store tmp sentences
        tmp_x.append(tmp_sentences)
    return tmp_x, _text

def convert_load_data_to_word_sequences(_x):
    # store text only 
    _text=''
    # store 
    tmp_sequence = []
    for document in range(len(_x)):
        # get only the words in the sentence
        tmp=re.sub('\W+',' ', _x[document]).strip()
        tmp_s = []
        if len(tmp)>0:
            _text +=tmp + ' '  
            tmp_sequence.append(tmp)
    return tmp_sequence, _text

def create_vocabulary(_text, rmv_stop_wrds):
    # create an empty network of model's vocabulary
    def init_vocab_network(n_inputs):
        network = list()
        for i in range(0,n_inputs):
            layer={'value':0,'token':''}
            network.append(layer)
        return network

    # given a list of words, return a dictionary of word-frequency pairs.
    def wordlist_to_freq_dict(wrdlist):
        wordfreq = [wrdlist.count(p) for p in wrdlist]
        return dict(zip(wrdlist,wordfreq))

    # sort the dictionary of word-frequency pairs in descending order
    def sort_freq_dict(freqdict):
        aux = [(freqdict[key], key) for key in freqdict]
        aux.sort()
        aux.reverse()
        return aux

    if rmv_stop_wrds:
        print('removing stop words...')
        tokenized_text = nltk.word_tokenize(_text)
        stopwords = nltk.corpus.stopwords.words('english')
        word_freq = nltk.FreqDist(tokenized_text)
        dict_filter = lambda word_freq, stopwords: dict( (word,word_freq[word]) for word in word_freq if word not in stopwords)
        wordlist = dict_filter(word_freq, stopwords)
    else :
        wordlist = FreqDist()
        wordlist.update(_text.split())
        
    sort_freq_list=sort_freq_dict(wordlist)   
    # initiate model's vocabulary
    _voc=init_vocab_network(len(sort_freq_list))
    
    # update vocabulary values
    j=0
    
    for index in sort_freq_list:
        # plus one to avoid the zero padding
        _voc[j]['value']=j+1 
        _voc[j]['token']=index[1]
        j+=1
    return _voc, len(_voc)


def data_to_integer(tmp_x, _y_labels, vocabulary, _max_seqlen, _max_opinionlen):
    # initial max sequence length
    max_sequence = 0
    def word_to_integer(str,dictionary):
        for index in dictionary:
            tmp_value = 0
            if index['token'] == str:
                tmp_value =index['value']
                break
        return tmp_value  
                
    # store data to integer
    _x_int = []
    
    for i in range(len(tmp_x)):
        # store tmp sentences
        sentences = tmp_x[i]
        # store tmp data to int sentences
        tmp_int_sentences = []
        
        # iterate through tmp sentences 
        for j in range(len(sentences)):
            # cut opinion greater than _max_opinionlen sentenses
            if j >= _max_opinionlen:
                break
            # get tmp sentence    
            sentence = sentences[j]
            # cut sequence length greater than _max_seqlen value
            if len(sentence.split()) > _max_seqlen:
                tmp_s =''
                for sent in sentence.split()[:_max_seqlen]:
                   tmp_s += sent + ' '
                # get tmp cut sentence
                sentence = tmp_s
             # map sentence word to integers acording to vocabulary values
            seq_integer = [word_to_integer(token,vocabulary) for token in sentence.split()]
            
            # update tmp maximum sequence 
            if max_sequence < len(seq_integer):
                max_sequence = len(seq_integer)
            # store converted to integer tmp sentence    
            tmp_int_sentences.append(seq_integer)
        # store converted to integer tmp sentences
        _x_int.append(tmp_int_sentences)
        
    return _x_int,_y_labels, max_sequence

def data_to_integer_document(tmp_x, _y_labels, vocabulary, _max_seqlen):
    # store tmp max sequence
    max_sequence = 0
    
    def word_to_integer(str,dictionary):
        for index in dictionary:
            tmp_value = 0
            if index['token'] == str:
                tmp_value =index['value']
                break
        return tmp_value
                 
    # store word to integers
    _x_int = []
    
    # loop through sentences
    for i in range(len(tmp_x)):
        # store tmp sentence
        sentence = tmp_x[i]
        # store converted to integer sentence
        tmp_int_sentences = []
        
        # cut sentence length greater than _max_seqlen value
        if len(sentence.split()) > _max_seqlen:
            # store text only
            tmp_s =''
            for sent in sentence.split()[:_max_seqlen]:
                tmp_s += sent + ' '
            # store cut tmp sentence
            sentence = tmp_s
        # map sentence sequence to vocabulary integers
        seq_integer = [word_to_integer(token,vocabulary) for token in sentence.split()]
        
        # update maximum sequence 
        if max_sequence < len(seq_integer):
            max_sequence = len(seq_integer)
        
        # store tmp to integer sentence
        _x_int.append(seq_integer)
    return _x_int,_y_labels, max_sequence

def calculate_document_length(documents):
    return max(len(x) for x in documents)
    
def calculate_sequence_length(num):
    if not num%2==0:
        num+=1
    return num


def pad_documents(documents, padding_word="0"):
    #  calculate maximum sequence length
    sequence_length = calculate_document_length(documents)
    
    padded_documents = []
    # loop through opinions
    for i in range(len(documents)):
        sentence = documents[i]
        num_padding = sequence_length - len(sentence)      
        tmp_array = np.concatenate([np.array(sentence),np.zeros(num_padding)])
        padded_documents.append(tmp_array)
        
    return padded_documents, sequence_length

def seqlengths(x):
    tmp_x =[]
    for i in range(len(x)):
        tmp_x.append(len(x[i]))
    return tmp_x


def pad_documents_sentence_document(documents,_seq_length, padding_word="0"):
    # calculate maximum sentences per opinion 
    document_length = calculate_document_length(documents)
    # calculate maximum sentence length
    sequence_length = calculate_sequence_length(_seq_length)
    
    padded_documents = []
    
    for i in range(len(documents)):
        tmp_padded_document =[]
        tmp_sentences = documents[i]
        if len(tmp_sentences) is 0:
            tmp_sentences = [[0]]
        tmp_padded_sentences=[]
        for j in range(len(tmp_sentences)):
            padded_zeros_words=np.array(np.zeros(sequence_length-len(tmp_sentences[j])))
            tmp_padded_sentence = np.concatenate((np.array(tmp_sentences[j]) ,padded_zeros_words),axis=0)
            tmp_padded_sentences.append(tmp_padded_sentence)
      
        tmp_padded_document = np.concatenate((np.array(tmp_padded_sentences),
        np.array(np.zeros((document_length-len(tmp_sentences),sequence_length)))),axis=0)
            
        padded_documents.append(tmp_padded_document)
        
    return padded_documents, document_length, sequence_length

def load_word_list(filename):
    word_list = []
    with open(filename,'r') as f:
         list = f.readlines()
         for word in list:
              word = word.rstrip('\n').lower()
              word_list.append(word)
    f.close()
    return word_list

def next_batch(num, data, labels,seqlens,_has_seqns):
    
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    seqlens_shufle = [seqlens[ i] for i in idx]
    if _has_seqns is True:
        return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(seqlens_shufle)
    else :
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def _run_document_mode(fl_name, max_seqlen,rmv_stop_wrds,n_classes):
    # set data path folder
    os.chdir(os.environ['USERPROFILE'] +'\\Downloads\\HyCoR-master\\data')
    print('loading dataset...')
    data_x,data_y = load_data(fl_name,n_classes)
    print('converting to sequences...')
    data_x,tmp_x= convert_load_data_to_word_sequences(data_x)
    print('creating vocabulary...')
    vocabulary,vocab_size = create_vocabulary(tmp_x,rmv_stop_wrds)
    print('vocabulary size: %d' % vocab_size)
    print('converting to sequence of integers...')
    x, y, max_sequence_length =  data_to_integer_document(data_x,data_y,vocabulary,max_seqlen)
    data_seqlens = seqlengths(x)
    print('zero padding...')
    x,  max_sequence_length = pad_documents(x)    
    print('end of preprocessing...')
    
    return x,y,max_sequence_length,data_seqlens,vocab_size


def _run_sentence_document_mode_pre_stage(fl_name, rmv_stop_wrds,n_classes,dataset_base):
    # set  data path folder
    os.chdir(os.environ['USERPROFILE'] +'\\Downloads\\HyCoR-master\\data')
    print('loading dataset...')
    data_x,data_y = load_data(fl_name,n_classes) # e.g 'PG.xlsx' 
    print('calculating ' + dataset_base + ' values...')
    data_x,tmp_x= split_load_data_sentences(data_x)
       
    vocabulary,vocab_size = create_vocabulary(tmp_x,rmv_stop_wrds)
    
    x, y, sequence_length =  data_to_integer(data_x,data_y,vocabulary,1000,1000)
    
    if dataset_base == 'max':
        # calculate max sentences/document
        _Sentences = int(maxSentences(x))
        # calculate max sequences in the corpus
        _Sequences = int(maxLength(x))
    elif dataset_base=='avg':
        # calculate avg sentences/document
        _Sentences = int(avgSentences(x))
        # calculate max sequences in the corpus
        _Sequences = int(maxLength(x))
        
    return _Sentences, _Sequences
    
def _run_sentence_document_mode(fl_name, max_seqlen, max_opinionlen,rmv_stop_wrds,n_classes):
    # set path folder
    os.chdir(os.environ['USERPROFILE'] +'\\Downloads\\HyCoR-master\\data')
    data_x,data_y = load_data(fl_name,n_classes) # e.g 'PG.xlsx' 
    print('converting to sequences...')
    data_x,tmp_x= split_load_data_sentences(data_x)
    
    print('creating vocabulary...')
    vocabulary,vocab_size = create_vocabulary(tmp_x,rmv_stop_wrds)
    print('vocabulary size: %d' % vocab_size)
    print('converting to sequences of integers...')
    x, y, sequence_length =  data_to_integer(data_x,data_y,vocabulary,max_seqlen,max_opinionlen)
    
    data_seqlens = seq_sent_doc_lengths(x,max_opinionlen)
    print('zero padding...')
    x, document_size, max_sequence_length = pad_documents_sentence_document(x,sequence_length)    
    print('end of preprocessing...')
    
    return x,y,max_sequence_length, document_size,data_seqlens,vocab_size
  
