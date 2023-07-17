# HLM

The 2Dclass2vec is the method for using word2vec to convert the 2D class into vectors. The filament embedding can be converted to vector by averaging the 2D class vector into filment vectors. These filament vectors can be used for further processing. 
![alt text](https://github.com/smallelephant9516/HLM/blob/master/figure1_new.png)

## Dependencies:

python3

pytorch >= 1.7

numpy

umap-learn

scikit-learn

### HLM-BERT
TO use the HLM-Bert, an additional hugging face environment need to be install from here: https://huggingface.co/docs/transformers/installation

## Usage

runing the jupyter note embedding.ipynb to get the embedding vector of each filament and separate them into different clusters using HLM_word2vec method, also, try to run HLM bert method, you can run the jupyter notebook BERT.ipynb

Or run:

     $ python filament_embedding.py --o External/job443/ --in_parts Class2D/job087/run_it025_data.star and use -h to check the possible command

# web app
  
