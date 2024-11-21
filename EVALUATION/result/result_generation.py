import copy
import pickle
import numpy as np
import pandas as pd
import os
import sys

#twords is no of top words
#for printing top words we need vocabulary,phi probabilities ,no of topics K 
def temp_save_model_twords(twords,K,phi,voc,jpath,filename,iteration_no):
  probs_words=[]
  top_word_list=[]
  V=len(voc)
  f=open(jpath+str(iteration_no)+'topics.txt','w')
  f1=open(jpath+str(iteration_no)+'topics_inlineprint.txt','w')
  if twords>V:
    twords=V
  for k in range(0,K):
    words_probs=[]
    for w in range(0,V):
      words_probs.append([w,phi[k][w]])
    words_probs.sort(key = lambda x: x[1],reverse=True)
    
  
    print("\n----------------Topic ",k,"----------------------\n")
    print("\n----------------Topic ",k,"----------------------\n",file=f)
    print("\n----------------Topic ",k,"----------------------\n",file=f1)
    temp_word=[]
    temp_prob=[]
    index=1
    for idx in words_probs[:twords]:
      print("Word "+str(index)+"=",voc[idx[0]],"  probability=",idx[1])
      try:
        print("Word "+str(index)+"=",voc[idx[0]],"  probability=",idx[1],file=f)
        print(voc[idx[0]],end=',',file=f1)
      except:
        print(voc[idx[0]].encode('utf8')[0:],"  probability=",idx[1],file=f)
        print(voc[idx[0]].encode('utf8')[0:],end=',',file=f1)
      temp_word.append(voc[idx[0]])
      temp_prob.append(idx[1])
      index+=1
    top_word_list.append(temp_word)
    probs_words.append(temp_prob)
  f.close()
  return probs_words,top_word_list
#we are also returning two things top words for each topic and their corresponding probabilities

def create_df(topic_word_list,probs_words,V):
    df=pd.DataFrame(columns=['Topic '+str(i) for i in range(1,len(topic_word_list)+1)])
    for topic in range(0,len(topic_word_list)):
        #column=np.asarray(topic_word_list[topic])
        column=np.stack((topic_word_list[topic], probs_words[topic]),axis=1)
        new_col=['nan' for i in range(V)]
        index=0
        for i in column:
            new_col[index]=i[0]+', '+str(i[1])
            index+=1
        df['Topic '+str(topic+1)]=new_col
    return df

def main():
    try:
        print("\nDataset path is ", sys.argv[1])
    except:
        print("Dataset file path not found")
        return 
    try:
        print("\n Input path is ",sys.argv[2])
    except:
        print("Input file path not found")
        return
        
    jpath=sys.argv[1]     #####gmbm generated files in gmbm output folder
    hpath = sys.argv[2]   #####resources present in gmbm input folder
    iteration_no= int(sys.argv[3])
    no_of_topics=int(sys.argv[4])
    filename=sys.argv[5]
    no_of_words=int(sys.argv[6])
    
    with open(jpath+'BETA_'+str(iteration_no),'rb') as f1:
        phi_topic10=pickle.load(f1)
    with open(hpath+'vocab','rb') as f4:
        voc=pickle.load(f4)
    
    probs_words_10,top_word_list_10=temp_save_model_twords(no_of_words,no_of_topics,phi_topic10,voc,jpath,filename,iteration_no)
    #df_10=create_df(top_word_list_10,probs_words_10,len(probs_words_10))
    #df_10.to_excel(jpath+filename+"Topics"+str(no_of_topics)+".xlsx",index=False)
    #print(df_10)
    
if __name__ == "__main__":
    main()