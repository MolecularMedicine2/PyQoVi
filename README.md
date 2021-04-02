### PyQoVi - Quantification of Virus in images
Based on the Paper: Deep Transfer Learning approach for automatic recognition of drug toxicity and inhibition of SARS-CoV-2 <br>
Data are avaible at: https://uni-duesseldorf.sciebo.de/s/bJSSMnhCNEFUosN <br>
Implemented in Python using the PyTorch Framework<br>
We provide the code to be able to analyse different active substances automatically and quickly on the basis of the imaging. Due to the topicality of the epidemic, we provide a solution here. 
<br>
# Installation Guide (ATM Windows Only)
1.Step: Install git (https://git-scm.com/downloads) and git clone the repo. Or just download the zip folder.<br>
<img src="/images/CloneBySSH.PNG" alt="CloneBySSH" title="CloneBySSH" />
<img src="/images/CMD.PNG" alt="CMD" title="CMD" />
2.Step: Install Anaconda and create a Python interpreter (Python 3.8.5) <br>
(https://docs.anaconda.com/anaconda/install/)<br>
3.Step: Install Pycharm https://www.jetbrains.com/de-de/pycharm/ <br> and choose the anaconda interpreter from you installation before
<img src="/images/PythonInt.PNG" alt="PythonInt" title="PythonInterpreter auswaehlen" />
4.Step: install the packages from the requirement.txt by calling pip install -r requirements.txt in the python console <br>
<img src="/images/PIPR.PNG" alt="PIPR" title="PIPR" />
If Pytorch installation dont work go to: https://pytorch.org/get-started/locally/ . Make sure your NVidia GPU Drivers are updated<br>
<img src="/images/PyTorch.PNG" alt="PyTorch" title="PyTorchConfig" />
<br>
<br>
<br>
# How to Use Guide GUI (ATM Windows Only)
1.Step: Training: Use our GUI: run start_app.py. 
<img src="/images/start_app.PNG" alt="start_app" title="start_app" />
Training: Enter a name for the model (model_file_name). Select the path with the Training, Validation and Test Data.<br>
Inference: Inference: Enter a name for the resultfile (result_file_name). Load a trained model by selecting the path. (.pth). The choose the folder for the data, you want analyze.<br>
3.Step: You find the result as csv-file in result folder and the trained model in the saved_models folder. <br>
<br>
<br>
<br>
<br>
# How to Use Guide CMD (ATM Windows Only)
First adjust the paths in the parameter_holder.py <br>
-file_path_train='./data/cpetox', Path to you Data (with subfolders train, test, val) and for this subfolders for the different classes<br>
-folder_path='./data/inference', Path tho Folder with images you want classify <br>
-reload_path='./saved_models/train_model_resnet_19_02___22_15_36.pth', Path to the trained model <br>
<img src="/images/FolderStructure.PNG" alt="FolderStructure" title="FolderStructure" />
<br>
1.Step: Load the TrainingValidationTest dataset for train by using the train_public.py function (run in your IDE or cmd) or the download a pretrained model from.... <br>
<img src="/images/cmdTrain.PNG" alt="cmdTrain" title="cmdTrain" /> <br>
Add Link <br>
2.Step: Put your pictues into the inference folder and run infer_public.py with your IDE or in the cmd line (Hint use the right env python interpreter) <br>
<img src="/images/cmdInfer.PNG" alt="cmdInfer" title="cmdInfer" /> <br>
<img src="/images/ProjectFolder.PNG" alt="Projectfolder" title="Projectfolder" />
3.Step: You find the result as csv-file in result folder and the trained model in the saved_models folder. <br>
<img src="/images/csvTab.PNG" alt="csvTab" title="csvResultFile" />

# Error Handling:
Error -2 tempfile.tif: Cannot read TIFF header. conda install libtiff=4.1.0=h885aae3_4 -c conda-forge or  conda install -c anaconda libtiff<br>

# TBC / Todo
- Open Code for Hyperparametertuning

# Latest features (03/2021)
- train, inference

# Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people. It would be highly appreciated !

# Contributing to PyQosic
If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues. 

# License , citation and acknowledgements
Please advice the **LICENSE.md** file. For usage of third party libraries and repositories please advise the respective distributed terms. Please cite our paper, when using this code:

```
@Article{v13040610,
AUTHOR = {Werner, Julia and Kronberg, Raphael M. and Stachura, Pawel and Ostermann, Philipp N. and Müller, Lisa and Schaal, Heiner and Bhatia, Sanil and Kather, Jakob N. and Borkhardt, Arndt and Pandyra, Aleksandra A. and Lang, Karl S. and Lang, Philipp A.},
TITLE = {Deep Transfer Learning Approach for Automatic Recognition of Drug Toxicity and Inhibition of SARS-CoV-2},
JOURNAL = {Viruses},
VOLUME = {13},
YEAR = {2021},
NUMBER = {4},
ARTICLE-NUMBER = {610},
URL = {https://www.mdpi.com/1999-4915/13/4/610},
ISSN = {1999-4915},
ABSTRACT = {Severe acute respiratory syndrome coronavirus-2 (SARS-CoV-2) causes COVID-19 and is responsible for the ongoing pandemic. Screening of potential antiviral drugs against SARS-CoV-2 depend on in vitro experiments, which are based on the quantification of the virus titer. Here, we used virus-induced cytopathic effects (CPE) in brightfield microscopy of SARS-CoV-2-infected monolayers to quantify the virus titer. Images were classified using deep transfer learning (DTL) that fine-tune the last layers of a pre-trained Resnet18 (ImageNet). To exclude toxic concentrations of potential drugs, the network was expanded to include a toxic score (TOX) that detected cell death (CPETOXnet). With this analytic tool, the inhibitory effects of chloroquine, hydroxychloroquine, remdesivir, and emetine were validated. Taken together we developed a simple method and provided open access implementation to quantify SARS-CoV-2 titers and drug toxicity in experimental settings, which may be adaptable to assays with other viruses. The quantification of virus titers from brightfield images could accelerate the experimental approach for antiviral testing.},
DOI = {10.3390/v13040610}
}

```
## Acknowledgements
Implementation is based on the https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.

# Disclaimer
This progam/code can not be used as diagnostic tool.
