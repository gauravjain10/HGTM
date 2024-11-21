import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.metrics import roc_curve, auc,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys



#twords is no of top words
#for printing top words we need vocabulary,phi probabilities ,no of events K 
def topwords(twords,K,phi,voc,voc_dic):
    global jpath
    f = open(jpath+"output_events.txt", "w")
    probs_words=[]
    top_word_list=[]
    V=len(voc)
    if twords>V:
        twords=V
    for k in range(0,K):
        words_probs=[]
        for w in range(0,V):
            words_probs.append([w,phi[k][w]])
        words_probs.sort(key = lambda x: x[1],reverse=True)
        print("\n----------------Event ",k,"----------------------\n",file=f)
        temp_word=[]
        temp_prob=[]
        index=1
        for idx in words_probs[:twords]:
            for name, pixel in voc_dic.items():  
                if pixel == list(voc[idx[0]]):
                    color_name=name
            print("Word "+str(index)+"=",color_name,"  probability=",idx[1],file=f)
            #print("Word "+str(index)+"=",voc[idx[0]])
            temp_word.append(color_name)
            temp_prob.append(idx[1])
            index+=1
        top_word_list.append(temp_word)
        probs_words.append(temp_prob)
    f.close()
    return probs_words,top_word_list
#we are also returning two things top words for each event and their corresponding probabilities

def create_df(event_word_list,probs_words,V):
    df=pd.DataFrame(columns=['Event '+str(i) for i in range(1,len(event_word_list)+1)])
    for event in range(0,len(event_word_list)):
        #column=np.asarray(event_word_list[event])
        column=np.stack((event_word_list[event], probs_words[event]),axis=1)
        new_col=['nan' for i in range(V)]
        index=0
        for i in column:
            new_col[index]=i[0]+', '+str(i[1])
            index+=1
        df['Event '+str(event+1)]=new_col
    return df



def submain(no_of_words,M,home_path,res_path,eventsrange,levels,output_folder,burst_events_ind,tpname,threshold,voc_dic,delimeter,iteration_no):
    main_cf_matrix=[]
    #print(M)
    with open(res_path+'paths', 'rb') as b6:
        paths = pickle.load(b6)
    with open(res_path+'vocab', 'rb') as b7:
        voc = pickle.load(b7)
    with open(res_path+'image_category_dict', 'rb') as b7:
        image_category_dict = pickle.load(b7)
    for t_no in eventsrange:
        result_path=home_path
        with open(result_path+'BETA_'+str(iteration_no),'rb') as b1:
            phi=pickle.load(b1)
        with open(result_path+'theta_'+str(iteration_no),'rb') as b2:
            theta=pickle.load(b2)
        
        probs_words,top_word_list=topwords(no_of_words,t_no,phi,voc,voc_dic)       
        burst_event_index=burst_events_ind[str(t_no)]
        image_names=[]
        for i in range(len(paths)):
            image_names.append(paths[i].split(delimeter)[-1])

        
        df = pd.DataFrame(columns=['Image_no','Image_name','Burst_Probability','Predicted','Actual'])
        for image_index in range(len(image_names)):
            image_name=image_names[image_index]
            for j in range(levels):
                pred_category='nonburst'
                pred_prob_val=0
                theta_probs=theta[image_index][j]
                for theta_index in range(len(theta_probs)):
                    if theta_index in burst_event_index:
                        pred_prob_val+=theta_probs[theta_index]
                        
                if pred_prob_val>=threshold:
                    pred_category='burst'
                    break
                    
            if pred_category=='burst':
                pred=1
            else:
                pred=0
            row={'Image_no':image_index,'Image_name':image_name,'Burst_Probability':pred_prob_val,'Predicted':pred,'Actual':image_category_dict[image_name]}
            df=df.append(row,ignore_index=True)
            
        label_df=df[['Image_no','Image_name','Burst_Probability','Predicted']]
        label_df.to_csv(home_path+output_folder+'/label_prediction.csv',index=False)
        
        flog = open(home_path+output_folder+"/classification_report_"+str(threshold)+"_.txt", "w")
        y_actual=df['Actual']
        y_actual=y_actual.to_numpy(dtype ='int')
        y_pred=df['Predicted']
        y_pred=y_pred.to_numpy(dtype ='int')
        print("######################################################################")
        print("Input threshold for prediction",threshold)
        print("Input threshold for prediction",threshold,file=flog)
        print(classification_report(y_actual,y_pred,target_names=['NonBurst','Burst']))
        print(classification_report(y_actual,y_pred,target_names=['NonBurst','Burst']),file=flog)
        cf_matrix = confusion_matrix(y_actual, y_pred)
        np.set_printoptions(suppress=True)
        target_names = ['NonBurst','Burst']
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in target_names],columns = [i for i in target_names])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
        plt.tight_layout()
        plt.savefig(home_path+output_folder+'/burst_CF'+str(threshold)+'.png', dpi=300, bbox_inches='tight')
        
        yp=df['Burst_Probability']
        yp=yp.to_numpy(dtype ='float')
        ytruth=df['Actual']
        ytruth=ytruth.to_numpy(dtype ='float')
        
        fpr, tpr, thresholds =roc_curve(ytruth, yp)
        roc_auc = auc(fpr, tpr)
        print("######################################################################")
        print("Area under the ROC curve : %f" % roc_auc)
        print("Area under the ROC curve : %f" % roc_auc,file=flog)
        print("AUC score",roc_auc_score(ytruth, yp))
        print("AUC score",roc_auc_score(ytruth, yp),file=flog)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("optimal Threshold value is:", optimal_threshold)
        print("optimal Threshold value is:", optimal_threshold,file=flog)
        yprediction=[]
        for prob in yp:
            if prob>=optimal_threshold:
                yprediction.append(1)
            else:
                yprediction.append(0)
        print(classification_report(y_actual,np.array(yprediction),target_names=['NonBurst','Burst']))
        print(classification_report(y_actual,np.array(yprediction),target_names=['NonBurst','Burst']),file=flog)        
        print("Accuracy",accuracy_score(y_actual,np.array(yprediction)),file=flog)  
        flog.close()
        #print("Burst Precision:",precision_score(y_actual, y_pred))
        #print("Burst Recall:",recall_score(y_actual, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_actual, np.array(yprediction)).ravel()
        #print("NonBurst Precision:",(tn/(tn+fn)))
        #print("NonBurst Recall:",(tn/(tn+fp)))
        cf_matrix = confusion_matrix(y_actual, np.array(yprediction))
        np.set_printoptions(suppress=True)
        target_names = ['NonBurst','Burst']
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in target_names],columns = [i for i in target_names])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
        plt.tight_layout()
        plt.savefig(home_path+output_folder+'/burst_CF_optimal'+str(threshold)+'.png', dpi=300, bbox_inches='tight')
        
        main_cf_matrix.append(cf_matrix)
        with open(home_path+output_folder+"/df_K"+str(t_no)+'_J'+str(levels)+'_'+tpname+'_thresh_'+str(threshold),'wb') as f0:
            pickle.dump(df,f0)
        #print(len(df))
    with open(home_path+output_folder+"/main_cf_matrix_J"+str(levels)+'_'+tpname,'wb')as f1:
        pickle.dump(main_cf_matrix,f1)



def makedf(M,master,main_cf_matrix,events,levels,ind,threshold):
    
    for i in range(len(main_cf_matrix)):
        k=events[i]
        tn, fp, fn, tp = main_cf_matrix[i].ravel()
        acc=(tp+tn)/M
        if tp==0 and fp==0:
            bp=0
            br=0
        else:
            bp=(tp/(tp+fp))
            br=(tp/(tp+fn))
        if tn+fn==0:
            np=0
        else:
            np=(tn/(tn+fn))
        nr=(tn/(tn+fp))
        if bp+br==0:
            f1=0
        else:
            f1=( (2 *bp * br)/(bp + br) )
        row={'K':k,'H':levels,"Pred_Thres":threshold,'B_Pre':bp,'B_Rec':br,'Accuracy':acc,'F1_score':f1,'NB_Pre':np,'NB_Rec':nr,'True Pos':tp,'False Pos':fp,'True Neg':tn,'False Neg':fn,'burst_event_index':ind}
        #print(row)
        master=master.append(row,ignore_index=True)
        #print(masterdf)
    return master

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
    global jpath    
    jpath=sys.argv[1]     #####GMBM generated files in GMBM output folder
    hpath = sys.argv[2]   #####GMBM input folder
    iteration_no= int(sys.argv[3])
    no_of_events=int(sys.argv[4])
    no_of_words=int(sys.argv[5])
    filename=sys.argv[5]

    
    with open(hpath+'paths','rb') as f1:
        paths=pickle.load(f1)
        
    image_names=[]
    image_category_dict={}
    
    if '\\' in paths[0]:
        delimeter='\\'
    else:
        delimeter='/'
    for i in range(len(paths)):
        image_file_name=paths[i].split(delimeter)[-1]
        category=paths[i].split(delimeter)[-2]
        category=category.lower()
        if category=='burst':
            image_category_dict[image_file_name]=1
        else:
            image_category_dict[image_file_name]=0
        image_names.append(image_file_name)
    print("No of images:",len(image_category_dict))
    print("No of Burst images:",sum(image_category_dict.values()))
    
    with open(hpath+"image_category_dict",'wb')as f:
        pickle.dump(image_category_dict,f)
        
    voc_dic={'Blue1':[0,0,255],'Blue2':[  0,   0, 250],'Blue3':[  0,   0, 245],'Blue4':[  0,   0, 240],\
         'Red1':[255,0,0],'Red2':[250,   0,   0],'Red3':[245,   0,   0],'Red4':[240,   0,   0],\
         'Yellow1':[255, 255,   0],'Yellow2':[255, 250,   0],'Yellow3':[255, 245,   0],'Yellow4':[255, 240,   0],\
         'Orange1':[255, 165,   0],'Orange2':[255, 160,   0],'Orange3':[255, 155,   0],'Orange4':[255, 150,   0]}
    
    
    with open(jpath+'BETA_'+str(iteration_no),'rb') as f1:
        phi_event10=pickle.load(f1)
    with open(hpath+'vocab','rb') as f4:
        voc=pickle.load(f4)
    
    probs_words_10,top_word_list_10=topwords(no_of_words,no_of_events,phi_event10,voc,voc_dic)
    df_10=create_df(top_word_list_10,probs_words_10,len(voc_dic))
    df_10.to_excel(jpath+filename+"Topics"+str(no_of_events)+".xlsx",index=False)
    print(df_10)
    print()

    
    tpname=str(no_of_events)
    eventsrange=[no_of_events]
    levels=int(input("Enter no of levels H"))
    M=len(image_names)
    #threshold=np.exp(-500)
    print("Enter theshold value delta if not press 0 ,then threshold_list will be set as=[0.1,0.01,0.001,0.0001]\n")
    delta_threshold=float(input())
    if delta_threshold>0:
        threshold_list=[delta_threshold]
    else:
        #threshold_list=[0.1,0.01,0.001,0.0001]
        threshold_list=[0.1,0.01,0.001,0.0001,0.00001]
    output_folder='out'
    CHECK_FOLDER_master = os.path.isdir(jpath+output_folder)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER_master:
        os.mkdir(jpath+output_folder)
        print("created folder : ", jpath+output_folder)
    
    burst_events_ind={}
    masterdf=pd.DataFrame(columns=['K','H','Pred_Thres','B_Pre','B_Rec','Accuracy','F1_score','NB_Pre','NB_Rec','True Pos','False Pos','True Neg','False Neg','burst_event_index'])
    row_list=[]
    print("Enter Guidance rare event index")
    rare_event_index=int(input())
    for threshold in threshold_list:

        for i in [rare_event_index]:
            burst_events_ind[str(eventsrange[0])]=[i]
            submain(no_of_words,M,jpath,hpath,eventsrange,levels,output_folder,burst_events_ind,tpname,threshold,voc_dic,delimeter,iteration_no)
            cf_path=jpath+output_folder+"/main_cf_matrix_J"+str(levels)+'_'+tpname
            with open(cf_path,'rb')as c:
                main_cf_matrix=pickle.load(c)
            master=makedf(M,masterdf,main_cf_matrix,eventsrange,levels,i,threshold)
            row_list.append(master)
        print("Computation of threshold",threshold," is completed")
        print(master)
        print()
        
    resultdf=row_list[0]
    for row in range(1,len(row_list)):
        resultdf=pd.concat([resultdf,row_list[row]])
        
    resultdf.to_csv(jpath+filename+"result_df.csv",index=False)
    
if __name__ == "__main__":
    main()