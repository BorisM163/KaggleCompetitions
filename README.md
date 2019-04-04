
Hello!

Below you can find a outline of how to reproduce our solution for the reducing-commercial-aviation-fatalities competition.
If you run into any trouble with the setup/code or have any questions please contact us at  Lustigy@post.bgu.ac.il  or Borismos@post.bgu.ac.il

# ARCHIVE CONTENTS
README.md  : this readme file
directory_structure.txt :  readout of the directory tree at the top level of the archive
requirements.txt : python packages
SETTINGS.json :   file allow the code to run
train.py             :  python code to rebuild models from scratch
predcit.py           : python  code to generate predictions from model binaries


# HARDWARE: 
(The following specs were used to create the original solution)
Centos-release-7-3.1611.el7.centos.x86_64
12 x Intel(R) Xeon(R) CPU E7- 2860  @ 2.27GHz
No GPU
24GiB memory (DIMM DRAM EDO)



# SOFTWARE :
Python 2.7.5
(python packages are detailed separately in `requirements.txt`, 
use the following shell command to install it: `pip install -r requirements.txt`
# DATA SETUP:
#(assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
#below are the shell commands used in each step, as run from the top level directory

    mkdir -p data
    cd data/
    kaggle competitions download -c reducing-commercial-aviation-fatalities -f train.csv
    kaggle competitions download -c reducing-commercial-aviation-fatalities -f test.csv

# Model Build & Prediction
 

 1. First make sure that the following files stored in the same path: 

> 	 - man_data.py
> 	 - SETTINGS.json
> 	 - train.py
> 	 - predict.py

 2. Second make sure that the following all the folder exists,  if not create using the shell: 
	 - models `mkdir -p models`
	 - submissions `mkdir -p submissions`
## Traing
shell command to run each the model traing is below:

    ./train.py

## Predict
shell command to run each the model traing is below:

    ./predict.py
