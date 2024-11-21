import pickle
import os
from collections import Counter

#GMBM_INPUT_PATH='C:/Users/HP/Desktop/gaurav/final_GMBM_version/GMBM_server/input/synthetic_1200x800_B25_NB975/'
GMBM_INPUT_PATH='C:/Users/HP/Desktop/gaurav/final_GMBM_version/GMBM_server/input/org_burst_50/'
####################################################################
##Loading Data##
with open(GMBM_INPUT_PATH+'map_patch_to_id','rb')as f:
  map_patch_to_id=pickle.load(f)
  
old_file = os.path.join(GMBM_INPUT_PATH, "map_patch_to_id")
new_file = os.path.join(GMBM_INPUT_PATH, "map_patch_to_id_old")
os.rename(old_file, new_file)
  
with open(GMBM_INPUT_PATH+'images','rb')as f:
  images=pickle.load(f)
  
old_file = os.path.join(GMBM_INPUT_PATH, "images")
new_file = os.path.join(GMBM_INPUT_PATH, "images_old")
os.rename(old_file, new_file)

with open(GMBM_INPUT_PATH+'vocab','rb')as f:
  vocab=pickle.load(f)

  
print("Total images:",len(images))
print("Total words:",len(map_patch_to_id))
print("vocab size:",len(vocab))
####################################################################
imagelist=[]
map_p2id=[]
limiter=1000 #max limit of a particual word in each image we take,so to avoid extra high frequency of patch in same image.
word_count=0
for image_index in range(len(images)):
  imagetemp=[]
  dic_list_vocab={} 
  for i in range(len(vocab)):
    dic_list_vocab[i]=0 #0 assignment of each vocab index
  #print("Before limiter:",dic_list_vocab)
  #########################################################
  for patch_no in range(len(images[image_index])):
    index_value=map_patch_to_id[word_count]
    if dic_list_vocab[index_value]<1000:
      imagetemp.append(images[image_index][patch_no])
      map_p2id.append(index_value)
      dic_list_vocab[index_value]+=1
    word_count+=1
  ########################################################
  #print("After limiter:",dic_list_vocab)
  imagelist.append(imagetemp)
  print("Preprocessing for image no is completed:",image_index)

##################################  
col_count = Counter(map_p2id) #freq of each word
print(col_count)
flog = open(GMBM_INPUT_PATH+"/Counter_new.txt", "w")
print(col_count,file=flog)
flog.close()
#################################################################################
print("Total images:",len(imagelist))
print("Total words:",len(map_p2id))
print("vocab size:",len(vocab))
##Saving Data## 
with open(GMBM_INPUT_PATH+"images",'wb')as f2:
      pickle.dump(imagelist,f2)

with open(GMBM_INPUT_PATH+"map_patch_to_id",'wb')as f3:
      pickle.dump(map_p2id,f3)