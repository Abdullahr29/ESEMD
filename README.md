# Python implementation of Enhanced SEMD for extraction of eigenmotions and eigenactions
### Main Author: Abdullah Rehman
### Co-Author: Illimar Assak
### Date: June 2022

*Please don't hesitate to contact me for any questions/details through email: abdullahyrehman49@gmail.com*

More details about the implementation with diagrams and parameters are in the thesis folder. Here my thesis, final presentation, and overleaf zip folder can be found.

## Project organization
`input_data` - various sets of input data used
`output_data` - various sets of dictionaries produced and pickled for further processing
`Thesis` - Folder with thesis and presentation
`Dictionary_Testing` - Jupyter notebook where extracted dictionaries, sklearn algorithms, PCA and more were tested and plotted
`eigenaction_testing` - Jupyter notebook where extracted dictionaries from experts and novices were tested and plotted
`ESEMD.py` - Main algorithm python file
`mvnx_video_better.m + mvnx_video_better_original.m` - 3D data plotters (incomplete)
`nonlinear_dimreduc` - Jupyter notebook where nonlinear dimensionality reduction methods were tested
`README.md` - A highly informative document
`requirements.txt` - Requirements file


### Installation
Simply create an environment and run `pip install -r requirements.txt`

#### Main script: `ESEMD.py`
To start an experiment run `python ESEMD.py` followed by a list of arguments.
For example if you want to run an experiment with parameters of window size 6, output directory experiment_1/w06, data size 50000, the command will look like this: `python run_SED.py --window_size 6 --output_name experiment_1/w06 --data_size 50000`
More info about the arguments in the 'Arguments' section below

However, input and output files need to be manually changed within the ESEMD.py file if necessary. 

### Hyperparameters:

`ws` - window size of ROI, 8 found to be optimal, for eigenactions, their dimensionality will often be as much as the window size due to algorithm
`prqt & srqt` - reconstruction thresholds for extracted PCs, decreasing them will mean more PCs are considered good and will result in a larger dictionary
`st` - reconstruction threshold for eigenmotion/eigenaction, increasing this means a higher percentage of the data will need to be accurately reconstructed, resulting in a smaller dictionary
`data_size` - training data size
`dimension_flag` - set to: True for eigenmotions, False for eigenactions

Eigenmotion Hyperparameters: 
`ws` - 8
`prqt` - 0.75 
`srqt` - 0.65 
`st` - 0.01
`dimension_flag` - True

Eigenaction Hyperparameters: 
`ws` - 8
`prqt` - 0.95 
`srqt` - 0.85 
`st` - 0.02
`dimension_flag` - False

Interference Reduction Stage has many values which can be parametrised and further explored, however they currently work well