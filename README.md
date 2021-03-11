## GraphRepur: Drug repurposing against breast cancer by integrating of drug-exposure expression profiles and drug-drug links based on graph neural network

### Overview
This directory contains code necessary to run the GraphRepur algorithm.
GraphRepur can be used to predict new drugs for cancer. GraphRepur was built based on GraphSAGE and modified the loss function of GraphSAGE to handle the imbalance breast cancer data.

### Requirements
To install all dependencies, the version of Python must be <= 3.7. TensorFlow, numpy, scipy, sklearn, and networkx are required (but tensorflow must be<2, and networkx must be <=1.11). You can install all the required packages using the following command:

	$ pip install -r requirements.txt
	
To guarantee that you have the right package versions, you can use [docker](https://docs.docker.com/) to easily set up a virtual environment. See the Docker subsection below for more info.


### Files description
'data' directory
	Input files. Contain approved breast cancer drug, drugs' signature description and drug-drug links information.

'Breast cancer_degs.csv': The differentially expressed genes of breast cancer based on GSE26910. The preprocessing procedure included log2 transformation and quantile normalization. The corresponding log2 (fold change) was calculated which is a ratio between the disease and control expression levels. 

'drug-drug_links.csv': Drug-drug links information according to STITCH. All values were collected from STITCH. The scores for different kinds of types are listed in columns of the detailed scores file. According to the official explanation of the STITCH database, all scores in the files are the same as those on the website, multiplied by 1000 (to make it an integer instead of a floating-point number).

'drug-exposure gene expression profiles.csv': The drug-exposure gene expression profiles drugs obtained from LINCS. The data is LINCS level 5, which has taken into account the correlation of repeated data. On the basis of level 4, level 5 data has weighted the experimental results of the same experimental conditions according to Spearman correlation.

'label.csv': Approved breast cancer drug from the Food and Drug Administration (FDA). European Medicines Agency (EMA), Drug Efficacy Study Implementation, Mosby's Drug Consult approval documents.
	
'graphsage' directory
	Code of model directory. Contain all code for build GraphRepur.

'pre_trained_Repur' directory
	Contain the pretrained GraphRepur model for breast cancer drug repurposing prediction.
	
### Tutorial	
For prerdicting drug repurposing for breast cancer, run:

	$ python pre_trained_GraphRepur.py
	
The full dataset (described in the paper) is available on the 'data' directory. The detail information description is in ' Files description' mentioned above.

Run GraphRepur.py to prerdict drug repurposing for other cancers, the model should be retrained on new datase. Options are:

	-M: Aggregator. Full name = Model , default = 'gcn' 
	-k: Drug-drug links keywords. Full name = kw , default = 'combined_score'  
	-t: Differentially expressed genes threshold. Full name = threshold , default = 0.05 
	-L: Loss function. Full name = LOSS , default = 'focal' 
	-F: Loss parameter. Full name = focal_alpha , default = 0.75 
	-c: The checkpt file path. Full name = checkpt_file , default = 'pre_trained_Repur/mod.ckpt' 
	-p: The input data path. Full name = path , default = 'data/example_data_combined_score' , 
	-l: Learning rate. Full name = lr , default = 0.01 , 
	-s: Sampling number. Full name = sam , default = (25 , 10 , 0) 
	-D: Hidden units. Full name = Dim , default = (256 , 256)
	-b: Batch Size. Full name = bz , default = 64 
	-d: Dropout. Full name = drop , default = 0.2 


The hyperparameters and aggregator could be modified. In ordere to predict drug repurposing for new drug of user’s drugs. GraphRepur need the drugs’ drug-exposure gene expression profiles and links between drugs. The data structure of input needs to be as the form shown in Figure 2A of paper.
The input data include files:
1) Network file: A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
2) Node names file: A json-stored dictionary mapping the graph node names to consecutive integers.
3) Node labels file: A json-stored dictionary mapping the graph node names to labels.
4) Node features file: A numpy-stored array of node features; ordering given by Node name file. 

### Acknowledgements
The original version of this code base was originally forked from https://github.com/williamleif/GraphSAGE, and we owe many thanks to William L. Hamilton for making his code available.
