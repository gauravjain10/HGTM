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
  bcounts1,bcounts2,bcounts3,bcounts4=0,0,0,0
  rcounts1,rcounts2,rcounts3,rcounts4=0,0,0,0
  ycounts1,ycounts2,ycounts3,ycounts4=0,0,0,0
  ocounts1,ocounts2,ocounts3,ocounts4=0,0,0,0

  global Blue1
  global Blue2
  global Blue3
  global Blue4

  global Red1
  global Red2
  global Red3
  global Red4

  global Yellow1
  global Yellow2
  global Yellow3
  global Yellow4

  global Orange1
  global Orange2
  global Orange3
  global Orange4

  for ind in range(len(colours)):
    pixel_array=colours[ind]

    bdis1=euclid_distance(pixel_array,Blue1)
    bdis2=euclid_distance(pixel_array,Blue2)
    bdis3=euclid_distance(pixel_array,Blue3)
    bdis4=euclid_distance(pixel_array,Blue4)

    rdis1=euclid_distance(pixel_array,Red1)
    rdis2=euclid_distance(pixel_array,Red2)
    rdis3=euclid_distance(pixel_array,Red3)
    rdis4=euclid_distance(pixel_array,Red4)

    ydis1=euclid_distance(pixel_array,Yellow1)
    ydis2=euclid_distance(pixel_array,Yellow2)
    ydis3=euclid_distance(pixel_array,Yellow3)
    ydis4=euclid_distance(pixel_array,Yellow4)

    odis1=euclid_distance(pixel_array,Orange1)
    odis2=euclid_distance(pixel_array,Orange2)
    odis3=euclid_distance(pixel_array,Orange3)
    odis4=euclid_distance(pixel_array,Orange4)

    dist_list=[bdis1,bdis2,bdis3,bdis4,rdis1,rdis2,rdis3,rdis4,ydis1,ydis2,ydis3,ydis4,odis1,odis2,odis3,odis4]
    #B1:B4=[0:4],R1:R4=[4:8],Y1:Y4=[8:12],O1:O4=[12:16]
    color_index=dist_list.index(min(dist_list))#minimum distance index find out

    if color_index==0:
      bcounts1+=counts[ind]
    elif color_index==1:
      bcounts2+=counts[ind]
    elif color_index==2:
      bcounts3+=counts[ind]
    elif color_index==3:
      bcounts4+=counts[ind]

    elif color_index==4:
      rcounts1+=counts[ind]
    elif color_index==5:
      rcounts2+=counts[ind]
    elif color_index==6:
      rcounts3+=counts[ind]
    elif color_index==7:
      rcounts4+=counts[ind]

    elif color_index==8:
      ycounts1+=counts[ind]
    elif color_index==9:
      ycounts2+=counts[ind]
    elif color_index==10:
      ycounts3+=counts[ind]
    elif color_index==11:
      ycounts4+=counts[ind]

    elif color_index==12:
      ocounts1+=counts[ind]
    elif color_index==13:
      ocounts2+=counts[ind]
    elif color_index==14:
      ocounts3+=counts[ind]
    else:
      ocounts4+=counts[ind]
  #index style BRYO
  final_count_list=[bcounts1,bcounts2,bcounts3,bcounts4,rcounts1,rcounts2,rcounts3,rcounts4,ycounts1,ycounts2,ycounts3,ycounts4,ocounts1,ocounts2,ocounts3,ocounts4]
  #print(final_count_list)
  numerator=rcounts1+rcounts2+rcounts3+rcounts4+ycounts1+ycounts2+ycounts3+ycounts4+ocounts1+ocounts2+ocounts3+ocounts4
  denominator=sum(final_count_list)
  if numerator/denominator>thresh:
    final_count_list=[0,0,0,0,rcounts1,rcounts2,rcounts3,rcounts4,ycounts1,ycounts2,ycounts3,ycounts4,ocounts1,ocounts2,ocounts3,ocounts4]
    color_palette_index=final_count_list.index(max(final_count_list))
  else:
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
    global Blue1
    global Blue2
    global Blue3
    global Blue4

    global Red1
    global Red2
    global Red3
    global Red4

    global Yellow1
    global Yellow2
    global Yellow3
    global Yellow4

    global Orange1
    global Orange2
    global Orange3
    global Orange4

    Blue1=np.asarray([0,0,255])
    Blue2=np.asarray([0,0,250])
    Blue3=np.asarray([0,0,245])
    Blue4=np.asarray([0,0,240])

    Red1=np.asarray([255,0,0])
    Red2=np.asarray([250,0,0])
    Red3=np.asarray([245,0,0])
    Red4=np.asarray([240,0,0])

    Yellow1=np.asarray([255,255,0])
    Yellow2=np.asarray([255,250,0])
    Yellow3=np.asarray([255,245,0])
    Yellow4=np.asarray([255,240,0])

    Orange1=np.asarray([255,165,0])
    Orange2=np.asarray([255,160,0])
    Orange3=np.asarray([255,155,0])
    Orange4=np.asarray([255,150,0])


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
      patch_shape = (20,5,3)#H,W,channels
      #change step if patch_shape change width
      patches = view_as_windows(img, patch_shape,step=5)
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
    words=[Blue1,Blue2,Blue3,Blue4,Red1,Red2,Red3,Red4,Yellow1,Yellow2,Yellow3,Yellow4,Orange1,Orange2,Orange3,Orange4]
    with open(input_path+'vocab','wb')as l3:
      pickle.dump(words,l3)
    
    voc_index_list=['Blue1','Blue2','Blue3','Blue4','Red1','Red2','Red3','Red4','Yellow1','Yellow2','Yellow3','Yellow4','Orange1','Orange2','Orange3','Orange4']    
    with open(input_path+"voc_index_list",'wb') as f:
      pickle.dump(voc_index_list,f)



if __name__ == "__main__":
    main()


