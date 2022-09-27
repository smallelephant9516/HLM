# streamlit 
import streamlit as st
import os
import argparse
# arrange data
import numpy as np
import pandas as pd
from utils import EMdata
#plotting
import matplotlib.pyplot as plt
import mrcfile
import plotly.express as px
from streamlit_plotly_events import plotly_events
import cv2
# clustering pacage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description='argument')

parser.add_argument('--dm_path', action='store', default='/net/jiang/home/li3221/scratch/practice-filament/10230-tau/External/job444/umap_2D.npy',
                    help="path for the dimension reduction result")
parser.add_argument('--meta_path', action='store', default='/net/jiang/home/li3221/scratch/practice-filament/10230-tau/Class2D/job087/run_it025_data.star',
                    help='the path for the 2Dclass meta file')
args = parser.parse_args()


st.set_page_config(page_title="2Dclass2vec", layout="wide")
def main(args):
    with st.sidebar:    # sidebar at the left of the screen
        # dimension reduction tab
        dm_path = st.text_input('dimension reduction result', value=args.dm_path) #F:\Github\2Dclass2vec\data\umap_2D.npy
        if dm_path is not None:
            filament_umap_2D=np.load(dm_path)
        N2D = st.checkbox('not from 2D classification')
        # 2D class result tab
        meta_path = st.text_input('2D class meta file', value=args.meta_path) # 'F:\Github/2Dclass2vec/data/run_it025_data.star'
        root_dir='/'.join(meta_path.split('/')[:-3])
        file_name=os.path.basename(meta_path)
        output_path=os.path.dirname(meta_path)+'/'+os.path.splitext(file_name)[0]
        if os.path.isdir(output_path) is False:
            os.mkdir(output_path)
        dataframe, helix_name, positive_label, optics= getdata(meta_path)
        #print('/'.join('-'.join(helix_name[0].split('-')[:-1]).split('/')[2:]))
        # number of clusters
        filament_cluster_number = st.number_input('number of cluster', value=3, min_value=1, max_value=20, format="%d")
        filament_pd=filament_basic(filament_umap_2D,helix_name,filament_cluster_number,positive_label)
        #elbow test
        st.line_chart(calculate_elbow(filament_umap_2D))
        true_positive_label = st.checkbox('show true label')
        #seperation
        separate=st.button('Separate!')
        if separate:
            pt(dataframe, filament_cluster_number, filament_pd['label'], optics, separate, output_path, file_name)
            separate=False
            print('finish seperation')

    st.title("visualize the 2Dclass2vec result")
    col1,_,col2 = st.beta_columns((5,0.1,5))
    # left side column, clustering result
    with col1:
        if true_positive_label:
            fig = px.scatter(filament_pd, x='umap1', y='umap2', color='true_label', width=800, height=800,opacity=0.5)
        else:
            fig = px.scatter(filament_pd, x='umap1', y='umap2', color='label',  width=800, height=800)
        selected_points = plotly_events(fig, click_event=True, hover_event=False)
    # right column, select filament
    with col2:
        # Writes a component similar to st.write()
        if selected_points != []:
            x = selected_points[0]['x']
            y = selected_points[0]['y']
            st.write('umap xy is:', round(x,6), round(y,6))
            index = filament_pd.index[(filament_pd['umap1'] == x) & (filament_pd['umap2'] == y)].tolist()
            row = filament_pd.iloc[index[0]]
            st.write('the filament is', row)
            def_select=dataframe[(dataframe['_rlnHelicalTubeID']==row['FiD']) & (dataframe['_rlnImageName_noseq']==row['MiD'])]
            def_select_xy=def_select[['_rlnCoordinateX','_rlnCoordinateY']]
            st.dataframe(def_select_xy)
            startx,endx,starty,endy = filament_start_end(def_select)
            st.write ('start position ',startx,starty)
            st.write ('end position ',endx,endy)
            micro_path=root_dir+'/'+'/'.join(row['MiD'].split('/')[2:])[:-1]
            with mrcfile.open(micro_path, permissive=True) as mrc:
                micrograph_array = mrc.data[:]
            micrograph_array=micrograph_array
            micrograph_array = cv2.normalize(micrograph_array, None, 0, 1, cv2.NORM_MINMAX)
            micrograph_array=low_pass_filter(micrograph_array,10)
            fig, ax = plt.subplots()
            ax.imshow(micrograph_array, cmap='gray')
            ax.plot([endx,startx],[endy,starty],c='red',alpha=0.2)
            print('finish')
            st.pyplot(fig)
        else:
            st.write('No select')

@st.cache(persist=True, show_spinner=True)
def cluster(filament_umap_2D, filament_cluster_number):
    umap_predict=KMeans(n_clusters=filament_cluster_number).fit_predict(filament_umap_2D)
    return umap_predict

@st.cache(persist=True, show_spinner=True)
def filament_basic(filament_umap_2D,helix_name,filament_cluster_number,positive_label):
    filament_ID=[]
    mic_ID=[]
    for i in range(len(helix_name)):
        mic_ID.append('-'.join(helix_name[i].split('-')[:-1]))
        filament_ID.append(helix_name[i].split('-')[-1])
    umap_predict=cluster(filament_umap_2D, filament_cluster_number)
    filament_pd=pd.DataFrame({'umap1':filament_umap_2D[:len(helix_name),0],'umap2':filament_umap_2D[:len(helix_name),1],
                              'FiD':filament_ID,'MiD':mic_ID, 'label': umap_predict[:len(helix_name)].astype('str'),'true_label': positive_label})
    return filament_pd

@st.cache(persist=True, show_spinner=True)
def calculate_elbow(filament_umap_2D):
    res = []
    n_cluster = range(1,20)
    for n in n_cluster:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(filament_umap_2D)
        res.append(np.average(np.min(cdist(filament_umap_2D, kmeans.cluster_centers_, 'euclidean'), axis=1)))
    return res

@st.cache(persist=True, show_spinner=True, ttl=1)
def pt(dataframe, filament_cluster_number,umap_predict, optics, separate, output_path, file_name):
    if separate=True:
        metadata=list(dataframe.columns)
        data=dataframe.values
        optics=optics
        for i in range(filament_cluster_number):
            locals()['cluster'+str(i)]=[]
            locals()['clusterID'+str(i)]=[]
        for i in range(len(corpus)):
            labels=umap_predict[i]
            locals()['clusterID'+str(labels)].append(i)
            lst=corpus[i]
            for j in range(len(lst)):
                dataline=lst[j][-1]
                locals()['cluster'+str(labels)].append(data[dataline])
        for i in range(filament_cluster_number):
            cluster_name='cluster'+str(i)
            data_cluster=locals()[cluster_name]
            if datatype==0:
                output=EMdata.output_star(output_path+'/'+file_name,i,data_cluster,metadata)
                output.opticgroup(optics)
                output.writecluster()
            elif datatype==1:
                output=EMdata.output_star(output_path+'/'+file_name,i,data_cluster,metadata)
                output.writemetadata()
                output.writecluster()
    print(separate)
    return 0

@st.cache(persist=True, show_spinner=True)
def getdata(meta_path,datatype=0,N2D=False):
    if datatype<2:
        file_info=EMdata.read_relion(meta_path)
        if datatype==0:
            #read data (relion3.1)
            dataset=file_info.getRdata_31()
            optics=file_info.extractoptic()
        else:
            #read relion 3.0
            dataset=file_info.getRdata()
        metadata=dataset[0]
        data=dataset[1]
        print(metadata)
        if N2D:
            label=np.load('/net/jiang/home/li3221/scratch/Github/Unsupervised-Classification/results/10230_485_ctf/custom_single/pretext/classes_KM.npy')
            corpus_information=EMdata.process_helical(dataset).extarct_helical(label)
        else:
            corpus_information=EMdata.process_helical(dataset).extarct_helical_select()
    
    dataframe=pd.DataFrame(data=data,columns=metadata)
    dataframe['_rlnImageName_noseq'] = [x[7:] for x in dataframe['_rlnImageName']]
    corpus_dic,helix_name=corpus_information

    positive_label = []
    for i in range(len(helix_name)):
        # positive_label.append(helix_name[i][11:14])
        # simulate experiment
        # positive_label.append(helix_name[i][63:68])
        positive_label.append(helix_name[i][11:14])

    return dataframe, helix_name, positive_label, optics

@st.cache(persist=True, show_spinner=True)
def filament_start_end(data_select):
    data_select_x = np.array(data_select['_rlnCoordinateX']).astype('float')
    data_select_y = np.array(data_select['_rlnCoordinateY']).astype('float')
    minx,maxx,miny,maxy=(np.min(data_select_x),np.max(data_select_x),np.min(data_select_y),np.max(data_select_y))
    if np.argmin(data_select_x)==np.argmin(data_select_y):
        startx, endx, starty, endy = minx, maxx, miny, maxy
    else:
        startx, endx, starty, endy = maxx, minx, miny, maxy
    return startx,endx,starty,endy

#@st.cache(persist=True, show_spinner=True)
#def show_micrograph(position,micrograph_path):
#    with mrcfile.open(micrograph_path, permissive=True) as mrc:
#        micrograph_array=mrc.data[:]
#    print(micrograph_array[0,0])
#    fig, ax = plt.subplots()
#    ax.imshow(micrograph_array,cmap='gray')
#    print('finish')
#    return fig, ax
@st.cache(persist=True, show_spinner=True)
def low_pass_filter(img,angstrom):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2)
    mask_length=int(max(crow,ccol)/2*(1/angstrom))
    mask = np.zeros((rows, cols,2), np.uint8)
    mask[crow - mask_length:crow + mask_length, ccol - mask_length:ccol + mask_length] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back

@st.cache(persist=True, show_spinner=True)
def separate_cluster():
    return 0

@st.cache(persist=True, show_spinner=True)
def write_cluster(filament_cluster_number):
    return 0

main(args)
