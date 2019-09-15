# HyCoR-ACM/SIGAPP
Code repo for A Hybrid Convolution to Reccurent Neural Network published in 34th ACM/SIGAPP Symposium on Applied Computing (SAC '19)

## Contributors

- [Pantelis Agathangelou](https://github.com/ailabunic-panagath)
- [Ioannis Katakis](https://github.com/iokat)

## Reference
When using the code-base please use the following reference to cite our work:
Pantelis Agathangelou and Ioannis Katakis. 2019. "A hybrid deep learning network for modelling opinionated content". In Proceedings of the 34th ACM/SIGAPP Symposium on Applied Computing (SAC '19). ACM, New York, NY, USA, 1051-1053. DOI: https://doi.org/10.1145/3297280.3297570

Information how to run the model
--------------------------------
#1: the code-base is set to run without path settings, if it is downloaded and placed at the downloads folder
#2: the data folder must contain the datasets in excel format. The columns must be arranged in the folowing format:
	-1st column : user's opinions,
	-3rd column : labels for three classes classification task (for binary simply set the n_classes=2 at the 'train_HyCoR.py' file,
	-5th column : labels for five classes or fine-grained user content, if exist,
	-6th column : labels for six classes labels if exist,
	-for text classification of many classes, simply place the labels at the corresponding column.
#3: the "training_config.json" file, it includes all the hyperparameters for the training of the model .
#4: The train_HyCoR.py is the main file to train the model.

## License
The framework is open-sourced under the Apache 2.0 License base. The codebase of the framework is maintained by the authors for academic research and is therefore provided "as is".
