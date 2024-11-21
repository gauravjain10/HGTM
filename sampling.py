import random
import numpy as np

import settings
import time
from  math import lgamma,log,exp
#from scipy.special import loggamma

def compute_prob_gdai(tau,iota,h1,Sd,nihk_i_h):

  summation_nihr=sum(nihk_i_h[h1:Sd])

  product_l_less_h=1
  l=0
  while l<h1:
    product_l_less_h*=( (iota[l]+sum(nihk_i_h[l+1:Sd])) / (tau[l]+iota[l]+sum(nihk_i_h[l:Sd])) )
    l+=1
  
  prob_gdai=( (tau[h1]+nihk_i_h[h1]) / (tau[h1]+iota[h1]+summation_nihr) ) * product_l_less_h
  return prob_gdai
  
def log_sum(log_a,log_b):
    if (log_a < log_b):
        vres = log_b + log(1 + exp(log_a - log_b))
    else:
        vres = log_a + log(1 + exp(log_b - log_a))

    return vres

  
def infer_phi(event,tindex): 
    if tindex in settings.pair_k_v_list:
        return
    val_nk_k=settings.nk[event]
    value_of_phikw_kv=settings.phikw[event][tindex]
    settings.phiEta_v[event] -= value_of_phikw_kv * settings.kappa
    settings.phi_k[tindex] -= value_of_phikw_kv
    
    nu_lambda=settings.nu*settings.lamb/ settings.V
    #nu_lambda_K_val=nu_lambda	+ settings.lamb + settings.K
    nu_lambda_K_val=settings.lamb + settings.K - settings.phi_k[tindex] - 1
    lamb_K_1_val=settings.lamb + settings.K - 1
    
    if (settings.nkj[event][tindex] > 0):
        settings.phikw[event][tindex] = 1
    else:
        try:
            ph1 = lgamma(settings.phiEta_v[event] + settings.kappa)  - lgamma(settings.phiEta_v[event] + settings.kappa + val_nk_k)

            ph1 += log((nu_lambda) + settings.phi_k[tindex]) - log(nu_lambda_K_val)

            if (settings.phiEta_v[event] > 0):
                ph2 = lgamma(settings.phiEta_v[event])- lgamma(settings.phiEta_v[event] + val_nk_k)
                ph2 += log(lamb_K_1_val - settings.phi_k[tindex]) - log(nu_lambda_K_val)
                lognorm = log_sum(ph1, ph2)    
            else:
                lognorm = ph1		

            rval = np.random.uniform()  

            if (log(rval) + lognorm <= ph1):
                settings.phikw[event][tindex] = 1
            else:
                settings.phikw[event][tindex] = 0
        except:
            settings.phikw[event][tindex] = 0
    
    value_of_phikw_kv=settings.phikw[event][tindex]
    settings.phiEta_v[event] += value_of_phikw_kv * settings.kappa
    settings.phi_k[tindex] += value_of_phikw_kv



def sampling(i,h,j,c,Sd):
    settings.nkj #K x V
    settings.nk #K x 1
    settings.nigijk #M x H x K
    settings.nigij  #M x H
    settings.nihk #M x H x K
    settings.nih # M x H
    settings.phikw
    settings.V
    settings.K
    settings.H
    settings.alpha
    settings.kappa
    settings.p
    settings.z
    settings.array
    settings.ksi
    settings.g
    settings.np_ksi
    settings.tau
    settings.iota
    settings.pg
    settings.pb
    settings.p_phikw
    settings.lamb
    #settings.Beta_sparse_thresh
    #settings.non_rare_visual_words_indexes
  
  ############################## Sampling g #################
    #bt=time.time()
    #gt0=time.time()
    zihj=settings.z[i][h][j]
    nihk_i_h=settings.nihk[i][h]
    
    event=settings.z[i][h][j]
    tindex=settings.array[c]
    settings.nkj[event][tindex]-=1
    settings.nk[event]-=1
    gihj=settings.g[i][h][j]
    settings.nigijk[i][gihj][event]-=1
    settings.nigij[i][gihj]-=1
    #ndbdaik[d][sent][event]-=1
    #ndbdai[d][sent]-=1
    settings.nihk[i][h][event]-=1
    settings.nih[i][event]-=1
    
     ##############################Sampling phikw#################
    #tic_phi=time.time()
    infer_phi(event,tindex)
    #toc_phi=time.time()
    ############################## phikw updated#######################
    #tic_g=time.time()
    #gt1=time.time()
    #bt_time_v[0]+=(gt1-gt0)
    if settings.H>1:
        for h1 in range(0,settings.H):
            #gt11=time.time()
            ksi_i_h=settings.ksi[i][h1]
            Ydjzdai=ksi_i_h[zihj]
            #gt2=time.time()
            sum_ksi_i_h=sum(ksi_i_h)
            #gt3=time.time()
            bdenominator=((sum_ksi_i_h*settings.alpha)+settings.nigij[i][h1])
            #gt4=time.time()
            if bdenominator==0:
                settings.pb[h1]=0
            else:
                #gt5=time.time()
                settings.pb[h1]=(((Ydjzdai*settings.alpha)+settings.nigijk[i][h1][zihj])/bdenominator)*compute_prob_gdai(settings.tau,settings.iota,h1,Sd,nihk_i_h)
                #gt6=time.time()
        #gt7=time.time()
        if settings.H>1:
            for h1 in range(1,settings.H):
                settings.pb[h1]+=settings.pb[h1-1]
                #gt1=time.time()
        #scaled sample because of unnormalized p[]
        #gt8=time.time()
        u1=random.random()*settings.pb[settings.H-1]

        for lev in range(0,settings.H):
            if settings.pb[lev]>u1:
                break
        settings.g[i][h][j]=lev
    else:
        lev=0
        #settings.g[i][h][j]=lev
    ################################ g updated#######################
    #toc_g=time.time()

    ##############################Sampling ksi#################
    # for sampling ksi h val is going to be the one which we found in sampling g which is lev

    tic_ksi=time.time()
    if  (settings.ksi_flag==False) and (settings.nigijk[i][lev][event])==0:
        ksi_i_lev=settings.ksi[i][lev]
        sum_r=sum(ksi_i_lev)
        r=i+1
        l=lev+1  
        rdotl=r*l
        m_d_K=settings.mew*settings.delta/settings.K
        rdotl_plus_m_d_K_plus_delta=rdotl+m_d_K+ settings.delta 
        summation_grl=np.sum(settings.np_ksi[0:l,0:r],axis=(0,1))

        for k in range(0,settings.K):
            sum_r_exclude_s_eq_k=sum_r-ksi_i_lev[k]
            summation_grlk=summation_grl[k]
            numerator=(sum_r_exclude_s_eq_k*settings.alpha)+settings.alpha
            denominator=numerator+settings.nigij[i][lev]
            prob_zd=np.exp(loggamma(numerator)-loggamma(denominator))
            prob_rdjk=((summation_grlk+m_d_K) / rdotl_plus_m_d_K_plus_delta)
            settings.pg[k]=prob_zd*prob_rdjk

      #cumulate multinomial parameters
        for k in range(1,settings.K):
            settings.pg[k]+=settings.pg[k-1]

      #scaled sample because of unnormalized p[]
        ur=random.random()*settings.pg[settings.K-1]

        for event1 in range(0,settings.K):
            if settings.pg[event1]>ur:
                break
        #gt11=time.time()
        settings.ksi[i][lev][event1]=1
        #gt12=time.time()
        settings.np_ksi[i][lev][event1]=1
    toc_ksi=time.time()
  ################################ ksi updated#######################

  ########################Sampling z############################
  #do multinomial sampling via cumulative method
    #tic_z=time.time()
    ksi_i_gih=settings.ksi[i][lev]
    #V_dot_eta=V*eta# pass this as parameter
    nigij_i_gih=settings.nigij[i][lev]
    nigijk_i_gih=settings.nigijk[i][lev]
    summation=sum(ksi_i_gih)
    summation_dot_alpha=summation*settings.alpha
    denominator_2=summation_dot_alpha+nigij_i_gih 
    for k in range(0,settings.K):
        if denominator_2==0:
            settings.p[k]=0
        else:
            sum_phikw=sum(settings.phikw[k])
            sum_phikw_dot_kappa=sum_phikw*settings.kappa
            phikw_dot_kappa=settings.phikw[k][tindex]*settings.kappa
            try:
                settings.p[k]=((phikw_dot_kappa + settings.nkj[k][tindex]) / ( sum_phikw_dot_kappa + settings.nk[k] ))*((ksi_i_gih[k]*settings.alpha)+ nigijk_i_gih[k]) / denominator_2
            except:
                settings.p[k]=0

  #cumulate multinomial parameters
    #zt10=time.time()
    for k in range(1,settings.K):
        settings.p[k]+=settings.p[k-1]
  
  #scaled sample because of unnormalized p[]
    u=random.random()*settings.p[settings.K-1]

    for newevent in range(0,settings.K):
        if settings.p[newevent]>u:
            break
    #toc_z=time.time()
    ######################got the topic##############################
    #print('phi time:',round(toc_phi-tic_phi,2),'g time:',round(toc_g-tic_g,2),'ksi time:',round(toc_ksi-tic_ksi,2),'z time:',round(toc_z-tic_z,2))
    settings.nkj[newevent][tindex]+=1
    settings.nk[newevent]+=1
    settings.nigijk[i][lev][newevent]+=1
    settings.nigij[i][lev]+=1
    settings.nihk[i][h][newevent]+=1
    settings.nih[i][newevent]+=1   


    return newevent




'''
clean phi code to understand
def infer_phi(): 
    nu_lambda=settings.nu*settings.lamb/ settings.V
    for k in  range(0,settings.K):
        for v in range(0,settings.V):
            value_of_phikw_kv=settings.phikw[k][v]
            settings.phiEta_v[k] -= value_of_phikw_kv * settings.kappa
            settings.phi_k[v] -= value_of_phikw_kv
            if v in settings.pair_k_v_list:
                continue
            #elif (settings.nkj[k][v] > 0) or (settings.rare_visual_words_indexes[k][v]==1):
            elif (settings.nkj[k][v] > 0):
                settings.phikw[k][v] = 1
            else:
                try:
                    ph1 = lgamma(settings.phiEta_v[k] + settings.kappa)  - lgamma(settings.phiEta_v[k] + settings.kappa + settings.nk[k])

                    ph1 += log((nu_lambda) + settings.phi_k[v]) - log((nu_lambda)	+ settings.lamb + settings.K)

                    if (settings.phiEta_v[k] > 0):
                        ph2 = lgamma(settings.phiEta_v[k])- lgamma(settings.phiEta_v[k] + settings.nk[k])
                        ph2 += log(settings.lamb + settings.K - settings.phi_k[v] - 1) - log((nu_lambda) + settings.lamb + settings.K)
                        lognorm = log_sum(ph1, ph2)    
                    else:
                        lognorm = ph1		

                    rval = np.random.uniform()  

                    if (log(rval) + lognorm <= ph1):
                        settings.phikw[k][v] = 1
                    else:
                        settings.phikw[k][v] = 0
                except:
                    settings.phikw[k][v] = 0
            
            value_of_phikw_kv=settings.phikw[k][v]
            settings.phiEta_v[k] += value_of_phikw_kv * settings.kappa
            settings.phi_k[v] += value_of_phikw_kv
'''