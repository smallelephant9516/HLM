# HLM

The 2Dclass2vec is the method for using word2vec to convert the 2D class into vectors. The filament embedding can be converted to vector by averaging the 2D class vector into filment vectors. These filament vectors can be used for further processing. 

## Dependencies:

python3

pytorch >= 1.7

numpy

umap-learn

scikit-learn

### HLM-BERT
TO use the HLM-Bert, an additional hugging face environment need to be install from here: https://huggingface.co/docs/transformers/installation

## usage

runing the jupyter note embedding.ipynb to get the embedding vector of each filament and separate them into different clusters using HLM_word2vec method, also, try to run HLM bert method, you can run the jupyter notebook BERT.ipynb

Or run:

     $ python filament_embedding.py --o External/job443/ --in_parts Class2D/job087/run_it025_data.star
   
Here is the possible arguments

  -i , --in_parts IN_PARTS
                        path to the particle.star meta files
                        
  --o O                 folder to save the particle file
  
  --datatype DATATYPE   0 is relion 3.1, 1 is relion 3, 2 is cryosparc
  
  --model MODEL         model to calculate the 2D class embedding
  
  -v, --verbose         Increaes verbosity
  
  --j                  number of thread
  

word2vec model parameter and filament averaging:

  --window        the window size to consider the neighboring segments
  
  --negative    number of the negative sample
  
  --emb_size    the size of word embedding
  
  --batch          the batch size
  
  --min_gain    min gain of the loss
  
  --loss            loss function
  
  --optim          loss function
  
  --avg_method 
                        the method to do filament average, 0 means simple
                        average, 1 means weight average
                        
dimension reduction and cluster diameter:

  --n_neighbors 
                        n_neighboers for the umap
                        
  --min_dist    min_dis in umap
  
  --pca_dim      PCA dimension
  
