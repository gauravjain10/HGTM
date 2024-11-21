#Steps to Run preprocessing script.

#1 In command line prompt change directory to /PREPROCESS/

#2 run python preprocessing.py dataset_path GMBM_input_path
		for example python preprocessing_burst.py burst_dataset/ GMBM_code/GMBM/input/burst/
					python preprocessing_aeroplane.py aeroplane_dataset/ GMBM_code/GMBM/input/aeroplane/
		Note: Try to give full path if file path not found error comes.
		
#3 Output files will be saved in output folder which will be used as inputs for GMBM script.