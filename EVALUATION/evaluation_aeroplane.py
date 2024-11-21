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



def submain(iteration_no,M,home_path,res_path,eventsrange,H,threshold,voc_dic,non_interested_event,rare_case_flag,rare_event_index,output_folder):
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
        df = pd.DataFrame(columns=['Image_no','Image_name','aeroplane_in_Sky_Probability','Predicted','Actual'])
        if rare_case_flag:
            for image_index in range(len(image_names)):
                image_name=image_names[image_index]     
                for j in range(H):
                    pred_prob_val=0
                    theta_probs=theta[image_index][j][:non_interested_event]
                    if theta_probs[rare_event_index]>=threshold:
                        pred_class_index=rare_event_index
                        pred_prob_val=theta_probs[rare_event_index]
                    else:
                        pred_class_index=np.argmax(theta_probs)
                        pred_prob_val=theta_probs[rare_event_index]

                row={'Image_no':image_index,'Image_name':image_name,'aeroplane_in_Sky_Probability':pred_prob_val,'Predicted':pred_class_index,'Actual':labels_np[image_index]}
                df=df.append(row,ignore_index=True)
        else:
            for image_index in range(len(image_names)):
                image_name=image_names[image_index]
                for j in range(H):
                    pred_prob_val=0
                    theta_probs=theta[image_index][j][:non_interested_event]
                    pred_class_index=np.argmax(theta_probs)
                    pred_prob_val=theta_probs[pred_class_index]

                row={'Image_no':image_index,'Image_name':image_name,'aeroplane_in_Sky_Probability':pred_prob_val,'Predicted':pred_class_index,'Actual':labels_np[image_index]}
                df=df.append(row,ignore_index=True)
        
        label_df=df[['Image_no','Image_name','aeroplane_in_Sky_Probability','Predicted']]
        label_df.to_csv(home_path+'/label_prediction.csv',index=False)
        np.set_printoptions(suppress=True)
        flog = open(home_path+"/classification_report_"+str(threshold)+"_.txt", "w")
        y_actual=df['Actual']
        y_actual=y_actual.to_numpy(dtype ='int')
        y_pred=df['Predicted']
        y_pred=y_pred.to_numpy(dtype ='int')
        print("######################################################################")
        print("Input threshold for prediction",threshold)
        print("Input threshold for prediction",threshold,file=flog)
        print(classification_report(y_actual,y_pred,target_names=['sky', 'aeroplane_in_sky']))
        print(classification_report(y_actual,y_pred,target_names=['sky', 'aeroplane_in_sky']),file=flog)
        print("Accuracy",accuracy_score(y_actual,y_pred),file=flog)
        cf_matrix = confusion_matrix(y_actual, y_pred)
        print(cf_matrix)
        np.set_printoptions(suppress=True)
        target_names = ['sky','aeroplane_in_sky']
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in target_names],columns = [i for i in target_names])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
        plt.tight_layout()
        plt.savefig(home_path+output_folder+'/aeroplane_CF'+str(threshold)+'.png', dpi=300, bbox_inches='tight')
        
        yp=df['aeroplane_in_Sky_Probability']
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
        print(classification_report(y_actual,np.array(yprediction),target_names=['sky', 'aeroplane_in_sky']))
        print(classification_report(y_actual,np.array(yprediction),target_names=['sky', 'aeroplane_in_sky']),file=flog)
        
        print("Accuracy",accuracy_score(y_actual,np.array(yprediction)),file=flog)
        cf_matrix = confusion_matrix(y_actual, np.array(yprediction))
        print(cf_matrix)
        print(cf_matrix,file=flog)
        target_names = ['sky','aeroplane_in_sky']
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in target_names],columns = [i for i in target_names])
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
        plt.tight_layout()
        plt.savefig(home_path+output_folder+'/aeroplane_CF'+str(optimal_threshold)+'.png', dpi=300, bbox_inches='tight')
        flog.close()
        
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
    print("Enter Guidance rare event index")
    rare_event_index=int(input())
    
    eventsrange=[2]
    H=1
    print("Enter theshold value delta")
    delta_threshold=float(input())
    non_interested_event_index=None
    rare_case_flag=True


    voc_dic={'White':[255, 255, 255],'Black':[ 0,  0, 0],'Grey':[187, 197, 199],'Sky_Blue':[135, 206, 235]}

    with open(res_path+'paths','rb') as f1:
        paths=pickle.load(f1)
        
    M=len(paths)
    image_names=[]
    labels=[]
    for i in range(len(paths)):
        image_file_name=' '.join(paths[i].split('\\')[-2:])
        category=paths[i].split('\\')[-2]
        if category=='aeroplane_in_sky':
            GT=1
            labels.append(GT)
        else:
            GT=0
            labels.append(GT)
        image_names.append(image_file_name)

    labels_np=np.array(labels)

    with open(res_path+"labels_np",'wb')as f:
        pickle.dump(labels_np,f)


    with open(res_path+"image_names",'wb')as f:
        pickle.dump(image_names,f)
    
    output_folder='out'
    CHECK_FOLDER_master = os.path.isdir(home_path+output_folder)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER_master:
        os.mkdir(home_path+output_folder)
        print("created folder : ", home_path+output_folder)
        
    #4th event contains only white patches no need to consider
    ypred,df,cf_matrix=submain(iteration_no,M,home_path,res_path,eventsrange,H,delta_threshold,voc_dic,non_interested_event_index,rare_case_flag,rare_event_index,output_folder)

    df.to_csv(home_path+"out/results_aeroplane_dataframe.csv",index=False)
    np.set_printoptions(suppress=True)

if __name__ == "__main__":
    main()



