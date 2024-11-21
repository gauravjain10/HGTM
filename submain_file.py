from initialization import *
from gmbm_initialization import *
from estimate import *
from configparser import ConfigParser
import pickle
import numpy as np
import random
import os
import settings
from datetime import datetime
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

def str2bool(value):
  return value.lower() in ("yes", "true", "t", "1")
  
def input_text_clean(list_of_tokens):
    word_list = [word.lower() for word in list_of_tokens]    
    # lemmatize
    #lemma = WordNetLemmatizer()
    #word_list = [lemma.lemmatize(word) for word in word_list]  
    return word_list

def submain(config_file_path,eventno):
    #print("enter the values of alpha beta niters savestep")
    #l=list(input().split(" "))
    config = ConfigParser()
    config.read(config_file_path)
    sections = config.sections()
    print(f'Sections: {sections}')
    print("Fetching values\n")
    settings.alpha=float(config['hyperparameters']['alpha'])
    settings.kappa=float(config['hyperparameters']['kappa'])
    settings.lamb=float(config['hyperparameters']['lamb'])
    settings.nu=float(config['hyperparameters']['nu'])
    settings.mew=float(config['hyperparameters']['mew'])
    settings.delta=float(config['hyperparameters']['delta'])
    #settings.Beta_sparse_thresh=int(config['hyperparameters']['Beta_sparse_thresh'])
    settings.H=int(config['hyperparameters']['H'])
    
    settings.savestep=int(config['parameters']['savestep'])
    settings.niters=int(config['parameters']['niters'])
    settings.which_iteration_start=int(config['parameters']['which_iteration_start'])
    settings.image_dataset_flag=str2bool(config['boolean']['image_dataset_flag'])
    settings.checkpoint_load_flag=str2bool(config['boolean']['checkpoint_load_flag'])
    settings.main_folder_directory=str2bool(config['boolean']['main_folder_directory'])
    
    settings.single_rare_event_flag=str2bool(config['boolean']['single_rare_event_flag'])
    settings.multi_rare_event_flag=str2bool(config['boolean']['multi_rare_event_flag'])
    settings.all_1_for_nonrare_flag=str2bool(config['boolean']['all_1_for_nonrare_flag'])
    settings.ksi_flag=str2bool(config['boolean']['ksi_flag'])
    #settings.phikw_flag=str2bool(config['boolean']['phikw_flag'])
    settings.event_folder_flag=str2bool(config['boolean']['event_folder_flag'])
    
    settings.res_path=config['path']['res_path']
    settings.output_folder=config['path']['output_folder']
    settings.no_guidance_flag=str2bool(config['guidance']['no_guidance_flag'])
    settings.single_rare_event_index=int(config['guidance']['single_rare_event_index'])
    settings.multi_rare_event_indexes=list(map(int,str(config['guidance']['multi_rare_event_indexes']).split(',')))
    #settings.rare_visual_words_indexes=list(map(int,str(config['guidance']['rare_visual_words_indexes']).split(',')))
    #settings.non_rare_visual_words_indexes=list(map(int,str(config['guidance']['non_rare_visual_words_indexes']).split(',')))
    
    



    print("config file contains")
    print(f'alpha: {settings.alpha}')
    print(f'kappa: {settings.kappa}')
    print(f'lamb: {settings.lamb}')
    print(f'nu: {settings.nu}')
    print(f'mew: {settings.mew}')
    print(f'delta: {settings.delta}')
    #print(f'Beta_sparse_thresh: {settings.Beta_sparse_thresh}')
    print(f'H: {settings.H}')
    print(f'savestep: {settings.savestep}')
    print(f'niters: {settings.niters}')
    print(f'which_iteration_start: {settings.which_iteration_start}')
    print(f'image_dataset_flag: {settings.image_dataset_flag}')
    print(f'checkpoint_load_flag: {settings.checkpoint_load_flag}')
    print(f'main_folder_directory: {settings.main_folder_directory}')
    print(f'event_folder_flag: {settings.event_folder_flag}')
    print(f'ksi_flag: {settings.ksi_flag}')
    #print(f'phikw_flag: {settings.phikw_flag}')
    print(f'single_rare_event_flag: {settings.single_rare_event_flag}')
    print(f'multi_rare_event_flag: {settings.multi_rare_event_flag}')
    print(f'all_1_for_nonrare_flag: {settings.all_1_for_nonrare_flag}')
    print(f'res_path: {settings.res_path}')
    print(f'output_folder: {settings.output_folder}')
    print(f'res_path: {settings.res_path}')
    print(f'output_folder: {settings.output_folder}')
    print(f'no_guidance_flag: {settings.no_guidance_flag}')
    if settings.no_guidance_flag==False:
        print(f'single_rare_event_index: {settings.single_rare_event_index}')
        print(f'multi_rare_event_indexes: {settings.multi_rare_event_indexes}')
        #print(f'rare_visual_words_indexes: {settings.rare_visual_words_indexes}')
        #print(f'non_rare_visual_words_indexes: {settings.non_rare_visual_words_indexes}')
        
        settings.rare_words_for_each_event=[]
        for tt in range(eventno):
            settings.rare_words_for_each_event.append([])
            
        if settings.multi_rare_event_flag:   
            for each_t_ind in settings.multi_rare_event_indexes:
                settings.rare_words_for_each_event[each_t_ind]=list(config['guidance']['event_'+str(each_t_ind)].split(','))
                
        if settings.single_rare_event_flag:   
            rareeventindex=settings.single_rare_event_index    
            settings.rare_words_for_each_event[rareeventindex]=list(config['guidance']['event_'+str(rareeventindex)].split(','))
        print("Guidance matrix")
        for each in range(eventno):
            print("Event no",each)
            #print(settings.rare_words_for_each_event[each])
            if not settings.image_dataset_flag:
                #print("converting lowercase and lemmitization")
                settings.rare_words_for_each_event[each]=input_text_clean(settings.rare_words_for_each_event[each])
                try:
                    print(settings.rare_words_for_each_event[each])
                except:
                    print([ww.encode('utf-16') for ww in settings.rare_words_for_each_event[each]])
        if settings.image_dataset_flag:
            with open(settings.res_path+"voc_index_list",'rb') as f:
                voc_index_list=pickle.load(f)
            for icounter in range(len(settings.rare_words_for_each_event)):
                for jcounter in range(len(settings.rare_words_for_each_event[icounter])):
                    visualwordname=settings.rare_words_for_each_event[icounter][jcounter]
                    settings.rare_words_for_each_event[icounter][jcounter]=voc_index_list.index(visualwordname)
            print("mapping visual word name to pixel index conversion")
            for tup in settings.rare_words_for_each_event:
                print(tup)

        

    with open(settings.res_path+'images', 'rb') as d6:
        settings.images = pickle.load(d6)
    with open(settings.res_path+'vocab', 'rb') as d7:
        vocab = pickle.load(d7)
        
    settings.N=len(settings.images)
    vocabulary=len(vocab)

    
    settings.masterpath=settings.output_folder+"GMBM_V"+str(vocabulary)+"/"
    CHECK_FOLDER_master = os.path.isdir(settings.masterpath)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER_master:
        os.mkdir(settings.masterpath)
        print("created folder : ", settings.masterpath)
    else:
        print(settings.masterpath, "folder already exists.")
    with open(settings.res_path+'vocab', 'rb') as d1:
        settings.voc = pickle.load(d1)
    with open(settings.res_path+'map_patch_to_id', 'rb') as d2:
        settings.old_array = pickle.load(d2)
    fname_time=datetime.now().strftime('_%H_%M_%d_%m_%Y')
    if settings.checkpoint_load_flag:
        event_path=settings.masterpath+os.listdir(settings.masterpath)[0]+"/"
    else:
        event_path=settings.masterpath+"Events_"+str(eventno)+fname_time+"/"
    if not settings.event_folder_flag:
        os.mkdir(event_path)
    settings.result_path=event_path+'GMBM_generated_files/'
    if not settings.event_folder_flag:
        os.mkdir(settings.result_path)    
    settings.nevents=eventno
    print("No of Events set to:",settings.nevents," and vocab size:",vocabulary)
    initialization(settings.N,vocabulary,settings.nevents,settings.H)
    gmbm_initialization()
    estimate(event_path,settings.savestep,settings.niters)   
    return f'done {eventno}'
