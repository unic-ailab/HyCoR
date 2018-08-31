information how to run the model
--------------------------------
#1: the archive must be downloaded and placed at the downloads folder
#2: the data folder must contain the datasets in excel format. The columns must be arranged in the folowing format:
	1st column : sentences
	3rd column : three classes labels if exist (for binary simply set the n_classes=2 in 'train_HyCoR.py' file
	5th column : five classes labels if exist
	6th column : six classes labels if exist
	#for greater number of classes place labels in the corresponding column.

#3: the "training_config.json" file includes all the hyperparameter for model training.
#4: open train_HyCoR.py as main file to train the model.