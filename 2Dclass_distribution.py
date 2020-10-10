import sys, os
import numpy as np
import pandas as pd
import EMdata

#data path
file_path='F:/script/class2vec/real_star_file/10340_case3_400.star'
datatype=0 #0 is relion 3.1, 1 is relion 3, 2 is cryosparc

file_name=os.path.basename(file_path)
output_path=os.path.dirname(file_path)+'/'+os.path.splitext(file_name)[0]
if os.path.isdir(output_path) is False:
    os.mkdir(output_path)

if datatype<2:
    file_info=EMdata.read_relion(file_path)
    if datatype==0:
        #read data (relion3.1)
        dataset=file_info.getRdata_31()
        optics=file_info.extractoptic()
    else:
        #read relion 3.0
        dataset=file_info.getRdata()
    metadata=dataset[0]
    print(metadata)
    data=dataset[1]
    print(data[0])
    corpus_information=EMdata.process_helical(dataset).extarct_helical()
else:
    #read cryosparc
    dataset=np.load(file_path)
    corpus_information=EMdata.process_cryosparc_helical(dataset).extract_helical()
corpus_dic=corpus_information[0]
corpus=list(corpus_dic.values())


class2D_label=[]
for i in range(len(corpus)):
    class2D_label.extend(corpus[i])
data_line=0
positive_label=[]
for i in range(len(data)):
    positive_label.append(data[data_line][7][18:21])
    #positive_label.append(data[data_line][0][67:70])
    data_line+=1 

class2D_label=list(map(int,class2D_label))
#positive_label=list(map(int,positive_label))
d={'class2D':class2D_label,'positive':positive_label}
df=pd.DataFrame(data=d)

all_2Dclass=df['class2D'].unique()
all_2Dclass.sort()
all_positive=df['positive'].unique()
all_positive.sort()

distribution=[]
for i in range(len(all_2Dclass)):
    class_index=all_2Dclass[i]
    temp=list(df['class2D'])
    class_number=temp.count(class_index)
    lst=[]
    for j in range(len(all_positive)):
        label_index=all_positive[j]
        lst.append(len(df[(df['class2D']==class_index) & (df['positive']==label_index)])/class_number)
    distribution.append(lst)
distribution=np.array(distribution)

import matplotlib.pyplot as plt
import matplotlib


height=np.zeros(len(distribution))

fig,ax=plt.subplots(2,figsize=(15,10))
cmap = matplotlib.cm.get_cmap('RdYlGn')
for i in range(len(all_positive)):
    lst=distribution[:,i]
    ax[0].bar(all_2Dclass,lst,bottom=height,label=i+1,color=cmap(i/2.5))
    height=height+lst
ax[0].plot(all_2Dclass,height+0.1-0.6)
ax[0].legend()
ax[0].set_xticks(all_2Dclass)
type12_distribution=distribution[:,0]
ax[1].hist(type12_distribution,bins=10,range=(0,1))
plt.savefig(output_path+'/'+os.path.splitext(file_name)[0]+'_2Dclass_label.png')
plt.show()