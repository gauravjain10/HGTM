import os
import cv2
import sys
import time
import pickle
import random
import logging
import joblib
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt 
from skimage.util.shape import view_as_windows




#default is norm 2 which is euclidean distance computation
#for efficiency it is better to use the numpy function compare to scipy function
def euclid_distance(pixelA, pixelB):
    dist = np.linalg.norm(pixelA - pixelB)
    return dist

def image_resize(pathname_list,image_index):
  orig_img = cv2.imread(pathname_list[image_index])
  orig_img=orig_img[...,::-1]
  #newsize = (100, 100) 
  #orig_img = cv2.resize(orig_img, (newsize)) 
  return orig_img


def dominant_color_palette(colours,thresh,counts):
  wcounts1=0
  bcounts1=0
  rcounts1=0
  ycounts1=0
  gcounts1=0

  global White1
  global Blue1
  global Red1
  global Yellow1
  global Green1



  for ind in range(len(colours)):
    pixel_array=colours[ind]
    wdis1=euclid_distance(pixel_array,White1)
    bdis1=euclid_distance(pixel_array,Blue1)
    rdis1=euclid_distance(pixel_array,Red1)
    ydis1=euclid_distance(pixel_array,Yellow1)
    gdis1=euclid_distance(pixel_array,Green1)


    dist_list=[wdis1,bdis1,rdis1,ydis1,gdis1]
    #B1:B4=[0:4],R1:R4=[4:8],Y1:Y4=[8:12],O1:O4=[12:16]
    color_index=dist_list.index(min(dist_list))#minimum distance index find out

    if color_index==0:
      wcounts1+=counts[ind]
    elif color_index==1:
      bcounts1+=counts[ind]
    elif color_index==2:
      rcounts1+=counts[ind]

    elif color_index==3:
      ycounts1+=counts[ind]
      
    else:
      gcounts1+=counts[ind]


      
  #index style BRYOGBr
  num=bcounts1+rcounts1+ycounts1+gcounts1
  deno=sum([wcounts1,bcounts1,rcounts1,ycounts1,gcounts1])
  if num/deno>thresh:
    final_count_list=[0,bcounts1,rcounts1,ycounts1,gcounts1]
  else:
    final_count_list=[wcounts1,bcounts1,rcounts1,ycounts1,gcounts1]
  #print(final_count_list)
  
  color_palette_index=final_count_list.index(max(final_count_list))
  return color_palette_index,final_count_list


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
        
    dataset_path=sys.argv[1]
    input_path = sys.argv[2]
    #creating a list of folder names to make valid pathnames later
    
    folders = [f for f in os.listdir(dataset_path)]
    print("sub folders:",folders)

    #creating a  list to store list of all files from different folders
    folder_files = []
    for folder_name in folders:
        folder_path = os.path.join(dataset_path, folder_name)
        z=[]
        for f in os.listdir(folder_path):
          z.append(f)
        folder_files.append(z)



    #creating a list of pathnames of all the documents
    pathname_list = []
    for fo in range(len(folders)):
        print(folders[fo],len(folder_files[fo]))
        for fi in folder_files[fo]:
            pathname_list.append(os.path.join(dataset_path, os.path.join(folders[fo], fi)))


    print("List of all file names are stored in (paths)")
    with open(input_path+"paths",'wb') as f1:
      pickle.dump(pathname_list,f1)

    """# PreProcessing"""
    global White1
    global Blue1
    global Red1
    global Yellow1
    global Green1


    White1=np.asarray([255,255,255])
    Blue1=np.asarray([0,0,255])
    Red1=np.asarray([255,0,0])
    Yellow1=np.asarray([255,255,0])
    Green1=np.asarray([0,255,0]) 
    



    docs=[]
    array=[]
    map_patch_to_id=[]
    start=time.time()
    patch_index_row_col=[]
    for image_index in range(len(pathname_list)):
      docs_temp=[]
      temp_patch_index_row_col=[]
      patch_to_id=[]
      img=image_resize(pathname_list,image_index)
      patch_shape = (10,10,3)#H,W,channels
      #change step if patch_shape change width
      patches = view_as_windows(img, patch_shape,step=10)
      #BUFFER=[]
      
      thresh=0.2*patch_shape[0]*patch_shape[1]
      for row in range(patches.shape[0]):
        for col in range(patches.shape[1]):
          image_patch=patches[row][col][0]
          colours, counts = np.unique(image_patch.reshape(-1,3), axis=0, return_counts=1)  
          color_palette_index,final_count_list=dominant_color_palette(colours,thresh,counts)
          #if final_count_list not in BUFFER:
            #BUFFER.append(final_count_list)
          docs_temp.append(image_patch)
          array.append(color_palette_index) 
          patch_to_id.append(color_palette_index)
          temp_patch_index_row_col.append([row,col])
      docs.append(docs_temp)
      patch_index_row_col.append(temp_patch_index_row_col)
      map_patch_to_id.append(patch_to_id)
      print("Preprocessing for image no is completed:",image_index)
    print("Done")
    print("Total Time taken in seconds: ",time.time()-start)


    print("List of all image are stored in (images)")
    with open(input_path+"images",'wb')as f2:
      pickle.dump(docs,f2)
      
    print("List of all mapping b/w patches and vocab are stored in (map_patch_to_id)")
    with open(input_path+"map_patch_to_id",'wb')as f3:
      pickle.dump(array,f3)
    col_count = Counter(array) 
    print(col_count)


    with open(input_path+"map_patch_to_id_without_flatten",'wb')as f3:
      pickle.dump(map_patch_to_id,f3)


    print("Patch indexes(i,j) which gives position of an original image (patch_index_row_col)")
    with open(input_path+'patch_index_row_col','wb')as l1:
      pickle.dump(patch_index_row_col,l1)

    """## Vocabulary(unique words)"""
    print("Vocabulary stored in (vocab)")
    words=[White1,Blue1,Red1,Yellow1,Green1]
    with open(input_path+'vocab','wb')as l3:
      pickle.dump(words,l3)
    
    voc_index_list=['White1','Blue1','Red1','Yellow1','Green1']
    with open(input_path+'voc_index_list','wb')as l3:
      pickle.dump(voc_index_list,l3)




if __name__ == "__main__":
    main()



