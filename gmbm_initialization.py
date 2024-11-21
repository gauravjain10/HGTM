import pickle 
import numpy as np 
import random 
import settings
import time
import sys
import math
from math import log
from collections import Counter

def z_INI(phikw,kappa,ksi,alpha,j,gdai,i,V,K):
    #k=int(random.random()*K)
    z_k_probabilities=[]
    for k in range(K):
        numerator=phikw[k][j]*kappa * ksi[i][gdai][k]*alpha
        #print("numerator",numerator)
        summation_phikv_dot_kappa=sum(phikw[k])*kappa
        #for v in range(0,V):
            #summation_phikv_dot_kappa+=(phikw[k][v]*kappa)
        
        summation_gbdaik_dot_alpha=sum(ksi[i][gdai])*alpha
        #for gk in range(0,K):
            #summation_gbdaik_dot_alpha+=(ksi[i][gdai][k]*alpha)
        try:
            ini=numerator/(summation_phikv_dot_kappa+summation_gbdaik_dot_alpha)
        except:
            ini=0
        z_k_probabilities.append(ini)
        #print(numerator,summation_phikv_dot_kappa,summation_gbdaik_dot_alpha)

    for k in range(1,K):
        z_k_probabilities[k]+=z_k_probabilities[k-1]
  
  #scaled sample because of unnormalized p[]
    u=random.random()*z_k_probabilities[K-1]

    for newevent in range(0,K):
        if z_k_probabilities[newevent]>u:
            break    
    #topic=np.argmax(z_k_probabilities)
    #topic=np.argmax(np.random.multinomial(1, [ini]*K, size=1))
    #print("topic prob",z_k_probabilities)
    #print("topic",topic)
    
    return newevent
    
    
    
def gmbm_initialization():
    kappa=settings.kappa
    V=settings.V
    H=settings.H
    K=settings.K
    N=settings.N
    lamb=settings.lamb
    nu=settings.nu
    mew=settings.mew
    delta=settings.delta
    alpha=settings.alpha
    images=settings.images #images x visual word matrix
    checkpoint_load_flag=settings.checkpoint_load_flag
    which_iteration_start=settings.which_iteration_start
    ksi_flag=settings.ksi_flag
    #phikw_flag=settings.phikw_flag
    result_path=settings.result_path
    old_array=settings.old_array
    

    
    if checkpoint_load_flag:
        print("Found checkpoint True\n")
        
        with open(result_path+"new_array", 'rb') as L12:
            settings.array=pickle.load(L12)
            
        BETA_path=result_path+str(which_iteration_start)+"/BETA_"+str(which_iteration_start)
        with open(BETA_path, 'rb') as L1:
            settings.BETA=pickle.load(L1)

        theta_path=result_path+str(which_iteration_start)+"/theta_"+str(which_iteration_start)
        with open(theta_path, 'rb') as L2:
            settings.theta=pickle.load(L2)

        with open(result_path+str(which_iteration_start)+"/nkj_"+str(which_iteration_start), 'rb') as L3:
            settings.nkj=pickle.load(L3)

        with open(result_path+str(which_iteration_start)+"/nk_"+str(which_iteration_start), 'rb') as L4:
            settings.nk=pickle.load(L4)

        with open(result_path+str(which_iteration_start)+"/nigijk_"+str(which_iteration_start), 'rb') as L5:
            settings.nigijk=pickle.load(L5)

        with open(result_path+str(which_iteration_start)+"/nigij_"+str(which_iteration_start), 'rb') as L6:
            settings.nigij=pickle.load(L6)

        with open(result_path+str(which_iteration_start)+"/nihk_"+str(which_iteration_start), 'rb') as L7:
            settings.nihk=pickle.load(L7)

        with open(result_path+str(which_iteration_start)+"/n_d_s_wcounts_"+str(which_iteration_start), 'rb') as L9:
            settings.n_d_s_wcounts=pickle.load(L9)
            
        with open(result_path+str(which_iteration_start)+"/nih_"+str(which_iteration_start), 'rb') as L9:
            settings.nih=pickle.load(L9)

        with open(result_path+str(which_iteration_start)+"/z_"+str(which_iteration_start), 'rb') as L10:
            settings.z=pickle.load(L10)

        with open(result_path+str(which_iteration_start)+"/ksi_"+str(which_iteration_start), 'rb') as L11:
            settings.ksi=pickle.load(L11)

        with open(result_path+str(which_iteration_start)+"/g_"+str(which_iteration_start), 'rb') as L12:
            settings.g=pickle.load(L12)

        with open(result_path+str(which_iteration_start)+"/np_ksi_"+str(which_iteration_start), 'rb') as L12:
            settings.np_ksi=pickle.load(L12)
            
        with open(result_path+str(which_iteration_start)+"/phikw_"+str(which_iteration_start), 'rb') as L12:
            settings.phikw=pickle.load(L12)
        
        with open(result_path+str(which_iteration_start)+"/phi_k_"+str(which_iteration_start), 'rb') as L12:
            settings.phi_k=pickle.load(L12)
            
        with open(result_path+str(which_iteration_start)+"/phiEta_v_"+str(which_iteration_start), 'rb') as L12:
            settings.phiEta_v=pickle.load(L12)
            
        with open(result_path+str(which_iteration_start)+"/pair_k_v_list_"+str(which_iteration_start), 'rb') as L12:
            settings.pair_k_v_list=pickle.load(L12)

        print("Checkpoint loaded successfully!!")
        return

    print("No checkpoint found")
    total_time=0
    start=time.time()
    print("Initialization Process is started\n")
    np.random.seed(100)
    
    ksi=[]#values between 0 and 1
    np_ksi=np.zeros((N,H,K))
    for i in range(0,N):
        each_img_list=[]
        for h in range(0,H):
            each_level_list=[]
            r=i+1
            l=h+1  
            rdotl=r*l
            m_d_K=mew*delta/K
            rdotl_plus_m_d_K_plus_delta=rdotl+m_d_K+ delta 
            summation_ksirl=np.sum(np_ksi[0:l,0:r],axis=(0,1))
            for k in range(0,K):
                if ksi_flag==True:
                    each_level_list.append(1)
                else:

                    summation_ksirlk=summation_ksirl[k]
                    pval=((summation_ksirlk+m_d_K) / rdotl_plus_m_d_K_plus_delta)
                    X=np.random.binomial(1,pval)
                    each_level_list.append(X)
                    np_ksi[i][h][k]=X
            each_img_list.append(each_level_list)
        ksi.append(each_img_list)
    total_time+=time.time()-start
    ksinp=np.asarray(ksi)
    print("ksi shape",ksinp.shape)
    print(ksinp.sum(axis=(0,1)))
    
    settings.pair_k_v_list=[]
    
    if settings.no_guidance_flag==False:
        rare_words_for_each_event=settings.rare_words_for_each_event
        if len(rare_words_for_each_event)==1:   
            #v_vector_size=np.zeros((1,V))
            v_vector_size=np.full((K, V), 0,dtype=int)
            for rare_w in rare_words_for_each_event[0]:
                try:
                    if settings.image_dataset_flag:
                        rare_id=rare_w
                    else:
                        rare_id=settings.voc.index(rare_w.lower())
                    v_vector_size[settings.single_rare_event_index][rare_id]=1
                    settings.pair_k_v_list.append(rare_id)
                except:
                    print(rare_w," is not present in vocab")
            settings.rare_visual_words_indexes=list(v_vector_size)
        else:
            set_rare_visual_words_indexes=[]
            #v_vector_size=np.zeros((K,V))
            v_vector_size=np.full((K, V), 0,dtype=int)
            for var_k in range(0,K):
                if var_k in settings.multi_rare_event_indexes:
                    for rare_w in rare_words_for_each_event[var_k]:
                        try:
                            if settings.image_dataset_flag:
                                rare_id=rare_w
                            else:
                                rare_id=settings.voc.index(rare_w.lower())
                            v_vector_size[var_k][rare_id]=1
                            settings.pair_k_v_list.append(rare_id)
                            set_rare_visual_words_indexes.append(rare_id)
                        except:
                            print(rare_w," is not present in vocab")
            settings.rare_visual_words_indexes=list(v_vector_size)

    
    
    
    settings.phi_k=[]
    for iv in range(0,settings.V):
        settings.phi_k.append(0)
        
    settings.phiEta_v=[]
    for ik in range(0,settings.K+1):
        settings.phiEta_v.append(0)
		
    BETA=[]
    settings.phikw=[]
    for xBETA in range(0,K):
        temp_BETA=[]
        temp_phikw=[]
        for yBETA in range(0,V):
            temp_BETA.append(0)
            temp_phikw.append(0)
        BETA.append(temp_BETA)
        settings.phikw.append(temp_phikw)
    print("BETA shape",np.asarray(BETA).shape)
    print("phikw shape",np.asarray(settings.phikw).shape)
    print("upsilon",settings.upsilon)
    print("Total words,vocab size",len(old_array),settings.V)
    
    if settings.V>settings.K:######for text dataset############
        weightage_value=max(int(math.ceil(settings.V/settings.K)),int(len(old_array)/settings.V))
    else:
        weightage_value=max(int(math.ceil(settings.K/settings.V)),int(len(old_array)/settings.V))
        
    if settings.image_dataset_flag:########for image dataset##########    
        weightage_value=max(Counter(old_array).values()) ##taking most frequent visual word count########
        #weightage_value=max(1,int(weightage_value/(settings.V*settings.upsilon)))
        
    for ik in range(0,settings.K):
        settings.phiEta_v[k] = 0
        sumval = 0
        for iv in range(0,settings.V):
            if settings.no_guidance_flag==False:
                value_input=settings.rare_visual_words_indexes[ik][iv]         
                if value_input==1:
                    if settings.image_dataset_flag:
                        settings.phikw[ik][iv] = int(weightage_value/sum(settings.rare_visual_words_indexes[ik]))
                    else:
                        settings.phikw[ik][iv] = weightage_value
                elif iv not in settings.pair_k_v_list: #condition check that not assign 1 to those pair which is already use by other kth event
                    nu_lambda=settings.nu*settings.lamb/ settings.V
                    nu_lambda_K_val=settings.lamb + settings.K - settings.phi_k[iv] - 1
                    #prob_val_ph=log(nu_lambda + settings.phi_k[iv]) - log(nu_lambda_K_val)
                    prob_val_ph=((nu_lambda + settings.phi_k[iv])/nu_lambda_K_val)
                    if settings.image_dataset_flag:
                        prob_val_ph=prob_val_ph+np.random.uniform()
                    if prob_val_ph <= settings.upsilon:
                        settings.phikw[ik][iv] = 1 
                    else:
                        settings.phikw[ik][iv] = 0
            else:
                #for no guidance 
                settings.phikw[ik][iv] =1
                
            settings.phi_k[iv] += settings.phikw[ik][iv]
            settings.phiEta_v[ik] += settings.phikw[ik][iv] * settings.kappa
            sumval += settings.phikw[ik][iv]
        if (sumval == 0):
            print("in ini_phi sum 0")
    #print(settings.phikw)
    phikw_np=np.array(settings.phikw)
    print('phikw sum',phikw_np.sum(axis=1))
    if settings.image_dataset_flag:
        print('phikw',settings.phikw)
    del phikw_np
    start=time.time()
    
    nkj=[]#is the number of times patch type j is associated with event k
    #Assuming V is vocabulary and K is no of events
    for ik in range(0,K):
        temp=[]
        for iv in range(0,V):
            temp.append(0)
        nkj.append(temp)
    total_time+=time.time()-start
    print("nkj shape",np.asarray(nkj).shape)
    start=time.time()

    nk=[]#represents the number of times event k is used in the whole data
    for ink in range(0,K):
        nk.append(0)
    total_time+=time.time()-start
    print("nk shape",np.asarray(nk).shape)
    start=time.time()

    nigijk=[]#is the number of times event k and gij are used. 
    for id in range(0,N):
        img_temp=[]
        for ij in range(0,H):
            level_temp=[]
            for ik in range(0,K):
                level_temp.append(0)
            img_temp.append(level_temp)
        nigijk.append(img_temp)
    total_time+=time.time()-start
    print("nigijk shape",np.asarray(nigijk).shape)
    start=time.time()

    nigij=[]#denotes the number of times gij is used. 
    for id in range(0,N):
        img_temp=[]
        for ij in range(0,H):
            img_temp.append(0)
        nigij.append(img_temp)
    total_time+=time.time()-start
    print("nigij shape",np.asarray(nigij).shape)
    start=time.time()

    nihk=[]# is number of times event vector indexed by k is used. 
    for id in range(0,N):
        img_temp=[]
        for ij in range(0,H):
            level_temp=[]
            for ik in range(0,K):
                level_temp.append(0)
            img_temp.append(level_temp)
        nihk.append(img_temp)
    total_time+=time.time()-start
    print("nihk shape",np.asarray(nihk).shape)
    start=time.time()

    nih=[]#is the number of visual words in the image level h. 
    for id in range(0,N):
        img_temp=[]
        for ik in range(0,K):
            img_temp.append(0)
        nih.append(img_temp)
        
    array=[]
    
    cc=0
    n_d_s_wcounts=[]
    for id in range(0,N):
        #print(id)
        img_temp=[]
        array_temp=[]
        img_i=images[id]#visual words of image no id
        #words_per_Sent=(len(doc_i)//J)+1
        for j in range(0,H):
            img_temp.append([0])
            array_temp.append([])
        for each_w in range(len(img_i)):
            x=np.random.randint(H,size=1)
            #print(x,x[0])
            #print(doc_temp,doc_temp[x[0]])
            #x[0] means x is vector of size 1 thats why we use 0
            #next doc_temp[x[0]] is a list with one element
            img_temp[x[0]][0]+=1
            array_temp[x[0]].append(old_array[cc])
            cc+=1
        #for j in range(J):
            #doc_temp.append(len(doc_i[j*words_per_Sent :(j+1)*words_per_Sent]))
        n_d_s_wcounts.append([wc[0] for wc in img_temp])
        for each_w_col_ind in array_temp:
            for col_ind in each_w_col_ind:
                array.append(col_ind)
    total_time+=time.time()-start
    #print("n_d_s_wcounts shape",np.asarray(n_d_s_wcounts).shape)
    with open(result_path+"new_array", 'wb') as L12:
            pickle.dump(array,L12)
            
    print("Total words",len(array))
    
    g=[]#values between 1 and H
    for i in range(0,N):
        each_img_list=[]
        for a in range(0,H):
            each_level_list=[]
            total_words=n_d_s_wcounts[i][a]
            for i in range(0,total_words):
                each_level_list.append(np.random.randint(H))
            each_img_list.append(each_level_list)
        g.append(each_img_list)
    total_time+=time.time()-start
    #print("g shape",np.asarray(g).shape)


    start=time.time()
    z=[]
    c=0
    for i in range(0,N):
        print("init for image no",i)
        img_temp=[]
        for h in range(0,H):
            level_temp=[]
            total_words=n_d_s_wcounts[i][h]#using n_d_s_wcounts initialization above
            for n in range(0,total_words):
                try:
                    tindex=array[c]
                except:
                    print("index out of range:",c)
                try:
                    gdai=g[i][h][n]
                except:
                    print("error",i,h,c,n)
                event=z_INI(settings.phikw,kappa,ksi,alpha,tindex,gdai,i,V,K)
                level_temp.append(event)
                nkj[event][tindex]+=1
                nk[event]+=1
                nigijk[i][gdai][event]+=1
                nigij[i][gdai]+=1
                nihk[i][h][event]+=1
                nih[i][event]+=1
                c+=1

            img_temp.append(level_temp)
        z.append(img_temp)
        
    np_ksi=np.asarray(ksi)
    print("numpy ksi shape ",np_ksi.shape)

    theta=[]
    for xtheta in range(0,N):
        image_theta=[]
        for ytheta in range(0,H):
            level_theta=[]
            for ztheta in range(0,K):
                level_theta.append(0)
            image_theta.append(level_theta)
        theta.append(image_theta)
    print("theta shape",np.asarray(theta).shape)
    '''print("consistency checking phi and n_kv")
    consflag=True
    for ik in range(0,K):
        for iv in range(0,V):
            if settings.phikw[ik][iv]==0:
                if nkj[ik][iv]!=0:
                    print("Error: not equal at",(ik,iv),"phi is",settings.phikw[ik][iv],'nks is',nkj[ik][iv])
                    consflag=False
    if consflag:
        print("Okay! consistency checking phi and n_kv")'''

    #print("\nnk")
    #print(nk)
    #print("\nndbdai")
    #print(ndbdai)
    #print("\nnda")
    #print(nda)
    settings.nkj=nkj
    settings.nk= nk
    settings.nigijk= nigijk
    settings.nigij=nigij
    settings.nihk= nihk
    settings.nih= nih
    settings.n_d_s_wcounts=n_d_s_wcounts
    settings.z=z
    settings.ksi=ksi
    settings.np_ksi=np_ksi
    settings.g=g
    settings.theta=theta
    settings.BETA=BETA
    settings.array=array
    settings.old_array=old_array


    total_time+=time.time()-start
    print("Initialization Process is completed!")
    print("\nTotal time taken:",total_time)
    
