# HLM

The HLM is the method for using the language models to convert a helical filament into vectors. These filament vectors can be used for further processing. 
![alt text](https://github.com/smallelephant9516/HLM/blob/master/HLM.png)

## Dependencies:

python3

pytorch >= 1.7

numpy

umap-learn

scikit-learn

### HLM-BERT
TO use the HLM-Bert, an additional hugging face environment needs to be installed from here: https://huggingface.co/docs/transformers/installation

## Usage

running the jupyter note embedding.ipynb to get the embedding vector of each filament and separate them into different clusters using HLM_word2vec method, also, try to run HLM bert method, you can run the jupyter notebook BERT.ipynb

Or run:

     $ python HLM.py --o your_output_directory --in_parts your_star_file
and use -h to check the possible command

  
