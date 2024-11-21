from sampling import *
from compute_BETA import *
from compute_theta import *
import logging
from datetime import datetime
import time 
import pickle 
import os 
import settings
import sys

def sanity_check(N,nigijk,nigij,iteration):
    print("Sanity check for iteration no:",iteration)
    #############################################################
    l=[]
    nigijk=np.asarray(nigijk)
    nigij=np.asarray(nigij)
    for i in range(N):
        l.append(nigijk[i,:].sum(axis=1)==nigij[i])
    l=np.asarray(l)
    l=l.flatten()
    if len(set(l))==1 and set(l)=={True}:
        print('summation of nigijk and nigij are equal')
        text='summation of nigijk and nigij are equal'
    else:
        print("summation of nigijk and nigij are not equal")
        text='summation of nigijk and nigij are not equal'

    #############################################################
    nkj=np.asarray(settings.nkj)
    nk=np.asarray(settings.nk)
    l2=nkj.sum(axis=1)==settings.nk
    if len(set(l2))==1 and set(l2)=={True}:
        print('summation of nkj and nk are equal')
        text+=' summation of nkj and nk are equal'
    else:
        print("summation of nkj and nk are not equal")
        text+=' summation of nkj and nk are not equal'
    print("nk")
    print(nk)
    ##########################################################
    l3=[] 
    nihk=np.asarray(settings.nihk)
    n_d_s_wcounts=np.asarray(settings.n_d_s_wcounts)
    for i in range(N):
        l3.append(nihk[i,:].sum(axis=1)==n_d_s_wcounts[i])
    l3=np.asarray(l3)
    l3=l3.flatten()
    if len(set(l3))==1 and set(l3)=={True}:
        print('summation of nihk and n_d_s_wcounts are equal')
        text+=' summation of nihk and n_d_s_wcounts are equal'
    else:
        print("summation of nihk and n_d_s_wcounts are not equal")
        text+=' summation of nihk and n_d_s_wcounts are not equal'

    return text    

def topwords(twords,jpath,iteration):
    probs_words=[]
    top_word_list=[]
    with open(settings.res_path+'vocab', 'rb') as d7:
        voc = pickle.load(d7)
    with open(jpath+'BETA_'+str(iteration),'rb') as f1:
        BETA=pickle.load(f1)
    
    V=len(voc)
    f=open(jpath+str(iteration)+'_topics.txt','w',encoding='utf-16')
    f1=open(jpath+str(iteration)+'_topics_inline.txt','w',encoding='utf-16')
    if twords>V:
        twords=V
    for k in range(0,settings.K):
        words_probs=[]
        for w in range(0,settings.V):
          words_probs.append([w,BETA[k][w]])
        words_probs.sort(key = lambda x: x[1],reverse=True)
        #print("\n----------------Topic ",k,"----------------------\n")
        print("\n----------------Topic ",k,"----------------------\n",file=f)
        print("\n----------------Topic ",k,"----------------------\n",file=f1)
        #temp_word=[]
        #temp_prob=[]
        index=1
        for idx in words_probs[:twords]:
          #print("Word "+str(index)+"=",voc[idx[0]],"  probability=",idx[1])
          try:
            print("Word "+str(index)+"=",voc[idx[0]],"  probability=",idx[1],file=f)
            print(voc[idx[0]],end=',',file=f1)
          except:
            print(voc[idx[0]].encode('utf-16')[0:],"  probability=",idx[1],file=f)
            print(voc[idx[0]].encode('utf-16')[0:],end=',',file=f1)
          #temp_word.append(voc[idx[0]])
          #temp_prob.append(idx[1])
          index+=1
        #top_word_list.append(temp_word)
        #probs_words.append(temp_prob)
    f.close()
    f1.close()
    return 
    
def topvisualwords(twords,jpath,iteration):
    probs_words=[]
    top_word_list=[]
    with open(settings.res_path+'voc_index_list', 'rb') as d7:
        voc = pickle.load(d7)
    with open(jpath+'BETA_'+str(iteration),'rb') as f1:
        BETA=pickle.load(f1)
    
    V=len(voc)
    f=open(jpath+str(iteration)+'_events.txt','w')
    f1=open(jpath+str(iteration)+'_events_inline.txt','w')
    if twords>V:
        twords=V
    for k in range(0,settings.K):
        words_probs=[]
        for w in range(0,settings.V):
          words_probs.append([w,BETA[k][w]])
        words_probs.sort(key = lambda x: x[1],reverse=True)
        #print("\n----------------Event ",k,"----------------------\n")
        print("\n----------------Event ",k,"----------------------\n",file=f)
        print("\n----------------Event ",k,"----------------------\n",file=f1)
        #temp_word=[]
        #temp_prob=[]
        index=1
        for idx in words_probs[:twords]:
          #print("Word "+str(index)+"=",voc[idx[0]],"  probability=",idx[1])
          try:
            print("Visual Word "+str(index)+"=",voc[idx[0]],"  probability=",idx[1],file=f)
            print(voc[idx[0]],end=',',file=f1)
          except:
            print(voc[idx[0]].encode('utf8')[0:],"  probability=",idx[1],file=f)
            print(voc[idx[0]].encode('utf8')[0:],end=',',file=f1)
          #temp_word.append(voc[idx[0]])
          #temp_prob.append(idx[1])
          index+=1
        #top_word_list.append(temp_word)
        #probs_words.append(temp_prob)
    f.close()
    f1.close()
    return 

def estimate(topic_path,savestep,niters):
  logfilename_=datetime.now().strftime('/GMBM_events_%H_%M_%d_%m_%Y.log')
  logger = logging.getLogger()
  fhandler = logging.FileHandler(filename=topic_path+logfilename_, mode='a')
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fhandler.setFormatter(formatter)
  logger.addHandler(fhandler)
  logger.setLevel(logging.DEBUG)

  '''settings.z
  settings.D
  settings.p
  settings.result_path
  settings.K
  settings.V
  settings.J
  settings.nda
  settings.checkpoint_load_flag
  settings.which_iteration_start'''

  start_iteration=1
  if settings.checkpoint_load_flag:
    start_iteration=settings.which_iteration_start+1
    
  logger.info("No of images,No of events and Vocab_size:"+str(settings.N)+" "+str(settings.K)+" and "+str(settings.V))
  logger.info("alpha,kappa,lambda,delta,nu and mew:"+str(settings.alpha)+","+str(settings.kappa)+","+str(settings.lamb)+","+str(settings.delta)+","+str(settings.nu)+" and "+str(settings.mew))
  text=sanity_check(settings.N,settings.nigijk,settings.nigij,0)
  logger.info(text)

  for iteration in range(start_iteration,niters+1):
    print("iteration number",iteration)
    c=0

    for i in range(0,settings.N):

      for h in range(0,settings.H):
        length=settings.n_d_s_wcounts[i][h]
        for j in range(0,length):
          print("itertation no:",iteration,"img no:",i,"word no:",j,"total visual words in curr img:",length,"sampling event")
          event=sampling(i,h,j,c,settings.H)
          settings.z[i][h][j]=event
          c+=1
    if settings.image_dataset_flag:
        print("PHIKW")
        print(settings.phikw)
    #compute_BETA(iteration)
    #print(settings.BETA)
      #print("Doc no",d,"completed and cumulative time taken=",ctime)
    logger.info("iteration number:"+str(iteration)+" No of events and vocabsize:"+str(settings.K)+" and "+str(settings.V))
    if (iteration%savestep==0) or (iteration==1):
      old_path=settings.result_path
      settings.result_path=settings.result_path+str(iteration)+"/"
      os.mkdir(settings.result_path)
      compute_theta(iteration)
      #compute_BETA(iteration)
      compute_BETA(iteration)
      with open(settings.result_path+"nkj_"+str(iteration), 'wb') as L4:
        pickle.dump(settings.nkj,L4)

      with open(settings.result_path+"nk_"+str(iteration), 'wb') as L5:
        pickle.dump(settings.nk,L5)

      with open(settings.result_path+"nigijk_"+str(iteration), 'wb') as L6:
        pickle.dump(settings.nigijk,L6)

      with open(settings.result_path+"nigij_"+str(iteration), 'wb') as L7:
        pickle.dump(settings.nigij,L7)

      with open(settings.result_path+"nihk_"+str(iteration), 'wb') as L8:
        pickle.dump(settings.nihk,L8)

      with open(settings.result_path+"nih_"+str(iteration), 'wb') as L9:
        pickle.dump(settings.nih,L9)
      
      with open(settings.result_path+"n_d_s_wcounts_"+str(iteration), 'wb') as L9:
        pickle.dump(settings.n_d_s_wcounts,L9)

      with open(settings.result_path+"z_"+str(iteration), 'wb') as L10:
        pickle.dump(settings.z,L10)

      with open(settings.result_path+"ksi_"+str(iteration), 'wb') as L11:
        pickle.dump(settings.ksi,L11)

      with open(settings.result_path+"g_"+str(iteration), 'wb') as L12:
        pickle.dump(settings.g,L12)
      
      with open(settings.result_path+"phikw_"+str(iteration), 'wb') as L12:
        pickle.dump(settings.phikw,L12)
      
      with open(settings.result_path+"phi_k_"+str(iteration), 'wb') as L12:
        pickle.dump(settings.phi_k,L12)
        
      with open(settings.result_path+"phiEta_v_"+str(iteration), 'wb') as L12:
        pickle.dump(settings.phiEta_v,L12)
        
      with open(settings.result_path+"pair_k_v_list_"+str(iteration), 'wb') as L12:
        pickle.dump(settings.pair_k_v_list,L12)
    
      with open(settings.result_path+"np_ksi_"+str(iteration), 'wb') as L12:
        pickle.dump(settings.np_ksi,L12)
      
      text=sanity_check(settings.N,settings.nigijk,settings.nigij,iteration)
      logger.info(text)  
      if settings.image_dataset_flag==False:
        topwords(20,settings.result_path,iteration)  
      else:
        topvisualwords(10,settings.result_path,iteration)
      print("\nGibbs sampling completed! iteration no:\n",iteration)
      logger.info("\nGibbs sampling completed! iteration no: "+str(iteration))
      settings.result_path=old_path  


  return
