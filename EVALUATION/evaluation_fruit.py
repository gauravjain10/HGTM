import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from sklearn.metrics import classification_report
import sys



#twords is no of top words
#for printing top words we need vocabulary,phi probabilities ,no of events K 
def topwords(twords,K,phi,voc,voc_dic):
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
        print("\n----------------Event ",k,"----------------------\n")
        temp_word=[]
        temp_prob=[]
        index=1
        for idx in words_probs[:twords]:
            for name, pixel in voc_dic.items():  
                if pixel == list(voc[idx[0]]):
                    color_name=name
            temp_word.append(color_name)
            temp_prob.append(idx[1])
            print(color_name,idx[1])
            index+=1
        top_word_list.append(temp_word)
        probs_words.append(temp_prob)
    return probs_words,top_word_list
#we are also returning two things top visual words for each event and their corresponding probabilities



def submain(iteration_no,M,home_path,res_path,eventsrange,H,voc_dic,non_interested_event_index):
    main_cf_matrix=[]
    #print(M)
    with open(res_path+'paths', 'rb') as b6:
        paths = pickle.load(b6)
    with open(res_path+'vocab', 'rb') as b7:
        voc = pickle.load(b7)
    with open(res_path+'labels_np', 'rb') as b7:
        labels_np = pickle.load(b7)
    with open(res_path+'image_names', 'rb') as b7:
        image_names = pickle.load(b7)
    for t_no in eventsrange:
        #print("No of events:",t_no)
        result_path=home_path
        with open(result_path+'BETA_'+str(iteration_no),'rb') as b1:
            phi=pickle.load(b1)
        with open(result_path+'theta_'+str(iteration_no),'rb') as b2:
            theta=pickle.load(b2)
            
        probs_words,top_word_list=topwords(10,t_no,phi,voc,voc_dic)        
        df = pd.DataFrame(columns=['Image_no','Image_name','fruit_col_Probability','Predicted','Actual'])
        for image_index in range(len(image_names)):
            image_name=image_names[image_index]
            for j in range(H):
                pred_prob_val=0
                theta_probs=theta[image_index][j][:non_interested_event_index]
                pred_class_index=np.argmax(theta_probs)
                pred_prob_val=theta_probs[pred_class_index]

            row={'Image_no':image_index,'Image_name':image_name,'fruit_col_Probability':pred_prob_val,'Predicted':pred_class_index,'Actual':labels_np[image_index]}
            df=df.append(row,ignore_index=True)
            
        label_df=df[['Image_no','Image_name','fruit_col_Probability','Predicted']]
        label_df.to_csv(home_path+'/label_prediction.csv',index=False)
        
        y_actual=df['Actual']
        y_actual=y_actual.to_numpy(dtype ='int')
        y_pred=df['Predicted']
        y_pred=y_pred.to_numpy(dtype ='int')
        print("Accuracy",accuracy_score(y_actual,y_pred))  
        #print("Burst Precision:",precision_score(y_actual, y_pred))
        #print("Burst Recall:",recall_score(y_actual, y_pred))
        flog = open(home_path+"/classification_report.txt", "w")
        target_names = ['Huckleberry', 'Apple Red', 'Banana','Watermelon']
        print(classification_report(y_actual,y_pred,target_names=target_names))
        print(classification_report(y_actual,y_pred,target_names=target_names),file=flog)
        flog.close()
        #print("NonBurst Precision:",(tn/(tn+fn)))
        #print("NonBurst Recall:",(tn/(tn+fp)))
        cf_matrix = confusion_matrix(y_actual, y_pred)
        print(cf_matrix)
    return y_pred,df,cf_matrix
        

def main():
    try:
        print("\nGMBM output path is ", sys.argv[1])
    except:
        print("GMBM output path not found")
        return 
    try:
        print("\n GMBM Input path is ",sys.argv[2])
    except:
        print("GMBM Input file path not found")
        return
        
    home_path=sys.argv[1]     #####stm generated files in stm output folder
    res_path = sys.argv[2]   #####resources preH in stm input folder
    iteration_no= int(sys.argv[3])
    no_of_events=int(sys.argv[4])
    filename=sys.argv[5]
    
    eventsrange=[5]
    H=1
    non_interested_event_index=4
    rare_case_flag=True


    voc_dic={'White':[255, 255, 255],'Blue':[ 0,  0, 255],'Red':[255,  0,  0],'Yellow':[255, 255,  0],'Green':[ 0, 255,  0]}

    with open(res_path+'paths','rb') as f1:
        paths=pickle.load(f1)
    M=len(paths)
    
    
    image_names=[]
    labels=[]
    for i in range(len(paths)):
        image_file_name=' '.join(paths[i].split('\\')[-2:])
        category=paths[i].split('\\')[-2]
        if category=='Huckleberry':
            GT=0
            labels.append(GT)
        elif category=='Apple Red':
            GT=1
            labels.append(GT)
        elif category=='Banana':
            GT=2
            labels.append(GT)
        elif category=='Watermelon':
            GT=3
            labels.append(GT)
        else:
            GT=4
            labels.append(GT)
        image_names.append(image_file_name)

    labels_np=np.array(labels)

    with open(res_path+"labels_np",'wb')as f:
        pickle.dump(labels_np,f)


    with open(res_path+"image_names",'wb')as f:
        pickle.dump(image_names,f)
    
    
    #4th event contains only white patches no need to consider
    ypred,df,cf_matrix=submain(iteration_no,M,home_path,res_path,eventsrange,H,voc_dic,non_interested_event_index)

    os.mkdir(home_path+'out')
    df.to_csv(home_path+"out/results_fruit_dataframe.csv",index=False)
    np.set_printoptions(suppress=True)
    target_names = ['Huckleberry', 'Apple Red', 'Banana','Watermelon']
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in target_names],columns = [i for i in target_names])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.savefig(home_path+'out/fruit_CF.png', dpi=300, bbox_inches='tight')
        
    

if __name__ == "__main__":
    main()