# Randi
Randi is a comprehensive deep learning framework for classification, inference and segmentation of anomalous diffusion trajectories using recurrent neural networks. 

# Requirements 
Randi requires Python 3.6 (or higher), tensorflow 2.0 (or higher), scikit-learn, scipy, andi_datasets packages. 

# Data 
The data that is used by Randi for training neural networks is generated using [andi_datasets](https://github.com/AnDiChallenge/ANDI_datasets) package. Once the data is created, it will be saved in "data" folder in .mat format.    

# Training neural networks 
Classification task includes the identification of the underlying anomalous diffusion model for single trajectories. Entire workflow of RANDI from generating data to creating and training the neural network for this task is presented in classification_train_network.ipynb notebook file. 

Inference task includes the estimation of the anomalous diffusion exponent of single trajectories. Entire workflow of RANDI from generating data to creating and training the neural network for this task is presented in inference_train_network.ipynb notebook file. This can be done in 1D, 2D or 3D that is defined in the beginning of the file as "dimension". 

Segmentation tasks include estimating the transition point of an anomalous diffusion trajectory, where the anomalous diffusion model and the exponent changes randomly in a trajectory. This task also includes the classification and inference of both segments. Entire workflow of RANDI from generating data to creating and training the neural network for this task is presented in segmentation_train_network.ipynb notebook file

All neural networks can be trained for 1D, 2D or 3D trajectories that is defined in the beginning of each notebook file as "dimension". 

# Demonstration of the analysis
Randi trains multiple networks for trajectories of different length for each task. All the pre-trained-networks can be found in the "nets" folder. The notebook file using_the_nets.ipynb demonstrates how any trajectory with an arbitrary length is analyzed using a combination of the nearest Randi nets for all tasks. This files uses the pre-trained networks that can be changed to user-trained networks if needed.   

# Issues
If you have any problems executing the software, please do not hesitate to contact us! If you find any bugs, please report an issue and we will fix it. 

# Cite us! 
If you use Randi for your research, please cite us here: 

Argun, Aykut, Giovanni Volpe, and Stefano Bo. "Classification, inference and segmentation of anomalous diffusion with recurrent neural networks." arXiv preprint arXiv:2104.00553 (2021).

[Click for Bibtex item](https://scholar.googleusercontent.com/scholar.bib?q=info:jb8tncf58zcJ:scholar.google.com/&output=citation&scisdr=CgWvkRqMENLSsVx6s0o:AAGBfm0AAAAAYKN_q0ru8M0O9cA772CdyoEyw0ZPQSIS&scisig=AAGBfm0AAAAAYKN_q6YEMAHv7aekghaieYwfFmJeepLt&scisf=4&ct=citation&cd=-1&hl=en)
