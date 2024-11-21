import pickle
import random
import nltk
import os
import re
import string
import numpy as np
import logging


from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import shutil

#filename = "./text_data/20newsgroup.zip"
  
# Target directory 
#extract_dir = "./text_data"
  
# Format of archie file 
#archive_format = "zip"
  
# Unpack the archive file  
#shutil.unpack_archive(filename, extract_dir, archive_format)  
#print("Archive file unpacked successfully.")

#creating a list of folder names to make valid pathnames later
my_path = './text_data/20newsgroup/'
folders = [f for f in os.listdir(my_path)]
print(folders)

#We are having 20 subfolders
print(len(folders))

#creating a  list to store list of all files from different folders

folder_files = []
for folder_name in folders:
    folder_path = os.path.join(my_path, folder_name)
    z=[]
    for f in os.listdir(folder_path):
      z.append(f)
    folder_files.append(z)

print("Here we are getting total number of files we have in all subfolders",sum(len(folder_files[i]) for i in range(20)))
no_of_files_perfolder=int(input("No of files you want for each folder"))
#creating a list of pathnames of 50 documents from each 20 subfolders
pathname_list = []
for fo in range(len(folders)):
    print(folders[fo],len(folder_files[fo]))
    count=0
    for fi in folder_files[fo]:
        pathname_list.append(os.path.join(my_path, os.path.join(folders[fo], fi)))
        count+=1
        if count==no_of_files_perfolder:
            break
        

#function to preprocess the words list to remove punctuations
def preprocess(words):
    #we'll make use of python's translate function,that maps one set of characters to another
    #we create an empty mapping table, the third argument allows us to list all of the characters 
    #to remove during the translation process
    
    #first we will try to filter out some  unnecessary data like tabs
    table = str.maketrans('', '', '\t')
    words = [word.translate(table) for word in words]
    
    punctuations = (string.punctuation).replace("'", "") 
    # the character: ' appears in a lot of stopwords and changes meaning of words if removed
    #hence it is removed from the list of symbols that are to be discarded from the documents
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in words]
    
    #some white spaces may be added to the list of words, due to the translate function & nature of our documents
    #we remove them below
    words = [str for str in stripped_words if str]
    
    #some words are quoted in the documents & as we have not removed ' to maintain the integrity of some stopwords
    #we try to unquote such words below
    p_words = []
    for word in words:
        if (word[0] and word[len(word)-1] == "'"):
            word = word[1:len(word)-1]
        elif(word[0] == "'"):
            word = word[1:len(word)]
        else:
            word = word
        p_words.append(word)
    
    words = p_words.copy()
        
    #we will also remove just-numeric strings as they do not have any significant meaning in text classification
    words = [word for word in words if not word.isdigit()]
    
    #we will also remove single character strings
    words = [word for word in words if not len(word) == 1]
    
    #after removal of so many characters it may happen that some strings have become blank, we remove those
    words = [str for str in words if str]
    
    #we also normalize the cases of our words
    words = [word.lower() for word in words]
    

    words = [word for word in words if len(word) > 1 and len(word) < 20]
    
    #words = [word for word in words if len(word) < 20]
    
    #lemma = WordNetLemmatizer()
    #words = [lemma.lemmatize(word) for word in words]
    
    return words

#function to remove stopwords

def remove_stopwords(words):
    words = [word for word in words if not word in stop_words]
    return words

#function to convert a sentence into list of words
def tokenize_sentence(line):
    print(line,type(line))
    words = line[0:len(line)-1].strip().split(" ")
    words = preprocess(words)
    words = remove_stopwords(words)
    return words

#function to remove metadata
def remove_metadata(lines):
    for i in range(len(lines)):
        if(lines[i] == '\n'):
            start = i+1
            break
    new_lines = lines[start:]
    return new_lines

#function to convert a document into list of words
def tokenize(path):
    #load document as a list of lines
    f = open(path, 'r',encoding="utf-8", errors="replace")
    text_lines = f.readlines()
    
    #removing the meta-data at the top of each document
    text_lines = remove_metadata(text_lines)
    
    #initiazing an array to hold all the words in a document
    doc_words = []
    
    #traverse over all the lines and tokenize each one with the help of helper function: tokenize_sentence
    for line in text_lines:
        #pattern = re.compile(r'[^a-z]+')
        #line = line.lower()
        #line = pattern.sub(' ', line).strip()   
        line = line[0:len(line)-1].strip().split(" ")
        line = preprocess(line)
        line = remove_stopwords(line)
        doc_words.append(line)

    return doc_words

def flatten(list):
    new_list = []
    for i in list:
        for j in i:
            new_list.append(j)
    return new_list
    
    
list_of_words = []
i=1
for document in pathname_list:
  print(i)
  i+=1
  list_of_words.append(flatten(tokenize(document)))
  

new_stopwords=['about','above','after','again','against','all','and','any','are',
 "aren't",'because','been','before','being','below','between','both',
 'but','can',"can't",'cannot','could',"couldn't",'did',"didn't",'does',"doesn't",'doing',
 "don't",'down','during','each','few','for','from','further','had',"hadn't",'has',"hasn't",'have',
 "haven't",'having',"he'd","he'll","he's",'her','here',"here's",'hers','herself','him','himself','his',
 'how',"how's","i'd","i'll","i'm","i've",'into','is',"isn't",'it',"it's",'its','itself',"let's",'more',
 'most',"mustn't",'myself','nor','not','off','once','only','other','ought','our','oursourselves',
 'out','over','own','same',"shan't",'she',"she'd","she'll","she's",'should',"shouldn't",'some','such','than','that',"that's",
 'the','their','theirs','them','themselves',
 'then','there',"there's",'these','they',"they'd","they'll","they're",
 "they've",'this','those','through','too','under','until',
 'very','was',"wasn't","we'd","we'll","we're","we've",
 'were',"weren't",'what',"what's",'when',"when's",'where',"where's",'which','while','who',"who's",
 'whom','why',"why's",'will','with',"won't",'would',"wouldn't",
 'you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves',
 'one','two','three','four','five','six','seven','eight','nine','ten','hundred','thousand',
 '1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th',
 "ax'ax'ax'ax'ax'ax'ax'ax'ax'ax'ax'ax'ax'ax'ax",'get','also','use','like','anyone','know','need',
 'want','using','may','new','aah','aap','aai','aas','abc','abcdef','aaplayexe',
 'even','good','bad','article','bus','work','think','help','please','thanks','writes','time','many','much','used','well',
 'say','lot','place','example','nice','first','really','thing','might','someone','look','fact','right','guns','believe',
  'way','batf','atf','January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December',
 'jim','miller','robert','smith','wall','actually','make','part','seems','tell','rate','must','see','going','never','keep','david','koresh','rights',
 'things','said','something','day','since','however','point','still','better','best','worst','anything','little','number','years','year','read','write',
 'true','false','take','made','every','far','back','long','find','come','without','high','low','says','mean','different','etc','enough','hard','sure',
 'real','around','got','least','seen','rather','last','second','done','put','possible','frank','whether','probably','wrong','else','another','away','either',
 'bos','nyi','pit','tor','buf','cal','det','pts','next','john','clinton','already','dod','leds','give','kind','ever','led','soon','men','getting','quite','nhl','san','generally','let','some','som','later','likely','have','hav','ioccc','thi','tim','wil','considered','interesting'
,'ther','although','consider','near','present','coming','edu','wit','upon','included','related','hold','apr','ways','full','com'
,'nothing','almost','perhaps','therefore','becaus','third','though','mostly','early','otherwise','started','makes','small'
,'yet','anyway','includes','rea','about','abou','tha','that','less','trying','today','tomorrow','apparently','othe','shall'
,'within','following','able','abov','according','across','afte','agai','ago','allowed','allows','always','among','ask','asked','asking'
,'assume','assuming','gets','given','gives','giving','goes','goo','indeed'
]
# removing 2 length characters,single quote as well as new stopwords and again rerun lda
def removing_2l_newstop_words(list_of_words):
  new_list_of_words=[]
  for ch in list_of_words:
    x=[]
    for hc in ch:
      if len(hc)>2:
        if hc not in new_stopwords:
          hc=re.sub("'s","",hc)
          x.append(hc)
    new_list_of_words.append(x)
  return new_list_of_words
  
list_of_words=removing_2l_newstop_words(list_of_words)
list_of_words = list(filter(None, list_of_words))

for ch in list_of_words:
    for hc in ch:
      if len(hc)<=2:
        ch.remove(hc)
        
new_low=[]
for ch1 in list_of_words:
  ch1=[x for x in ch1 if not any(c.isdigit() for c in x)]
  #ch1=list(set(ch1))
  new_low.append(ch1)

#print("No of docs",len(new_low))

np_list_of_words = np.asarray(flatten(new_low))
words, counts = np.unique(np_list_of_words, return_counts=True)
#print("No of unique words",len(words))
words=list(words)

#####################################
'''#put those words which we want to give guidance to gmbm.
checklist=['atheism','graphics','windows','bios','cmo','mac', 'apple','xterm','sale','cars',\
'motorcycle','baseball','hockey','encryption','power', 'circuit','disease','space','christian',\
'gun','israel', 'armenian','government', 'president','religion', 'muslim']
#since below we are selecting words based on frequency to reduce vocab size
###########################################
vocabulary=[]
maxfreq=int(input("Enter max freq selection word"))
for i in range(0,len(words)):
    if words[i] in checklist:
        vocabulary.append(words[i])
    elif counts[i]>maxfreq:
        vocabulary.append(words[i])

docs=[]
count=0
for d in new_low:
  wlist=[w for w in d if w in vocabulary]
  docs.append(wlist)
  count+=len(wlist)

#print("Total words",count)
'''

docs=new_low
vocabulary=words
  
#with open(save_path+"np_list_of_words",'wb')as f5:
  #pickle.dump(np_list_of_words,f5)
  
#map word2id  
array=[]
M=len(docs)
for x1 in range(0,M):
  N=len(docs[x1])
  #print(x1)
  for x2 in range(0,N):
    array.append(vocabulary.index(docs[x1][x2]))
    
print("no of docs",len(docs))
print("vocab size",len(vocabulary))
print("total words",len(array))

save_path='./text_data/'
save_path+='20ns_docs'+str(len(docs))+'_v'+str(len(vocabulary))+'_w'+str(len(array))+'/'
os.mkdir(save_path)
print("output path",save_path)

print("Save in docs notations press 1 otherwise press 0 for image name conventions\n")
notation=int(input())

#with open(save_path+"list_of_words",'wb')as f1:
  #pickle.dump(list_of_words,f1)

with open(save_path+"vocab",'wb')as f2:
  pickle.dump(vocabulary,f2)

if notation==1:
    with open(save_path+"docs",'wb')as f3:
      pickle.dump(docs,f3)
    with open(save_path+"array",'wb')as f4:
      pickle.dump(array,f4)
else:
    with open(save_path+"images",'wb')as f3:
      pickle.dump(docs,f3)  
    with open(save_path+"map_patch_to_id",'wb')as f3:
      pickle.dump(array,f3)


# clean text data
# remove non alphabetic characters
# remove stopwords and lemmatize

def input_text_clean(list_of_tokens):
    word_list = [word.lower() for word in list_of_tokens]
    '''import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    
    # lemmatize
    lemma = WordNetLemmatizer()
    word_list = [lemma.lemmatize(word) for word in word_list] ''' 
    return word_list

      



'''checklist=input_text_clean(checklist)
print(checklist)
for ch in checklist:
  try:
    print(ch,vocabulary.index(ch))
  except:
    print(ch,'nf')'''
