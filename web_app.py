# streamlit 
import streamlit as st
import os
# arrange data
import numpy as np
import pandas as pd
from utils import EMdata
#plotting
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_plotly_events import plotly_events
# clustering pacage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

st.beta_set_page_config(page_title="2Dclass2vec", layout="wide")
def main():
    with st.sidebar:    # sidebar at the left of the screen
        # dimension reduction tab
        dm_path = st.text_input('dimension reduction result', value='F:\Github/2Dclass2vec/data/umap_2D.npy') #F:\Github\2Dclass2vec\data\umap_2D.npy
        if dm_path is not None:
            filament_umap_2D=np.load(dm_path)
        # 2D class result tab
        meta_path = st.text_input('2D class meta file', value='F:\Github/2Dclass2vec/data/run_it025_data.star') # 'F:\Github/2Dclass2vec/data/run_it025_data.star'
        file_name=os.path.basename(meta_path)
        output_path=os.path.dirname(meta_path)+'/'+os.path.splitext(file_name)[0]
        if os.path.isdir(output_path) is False:
            os.mkdir(output_path)
        dataframe, helix_name = getdata(meta_path)   
        #print('/'.join('-'.join(helix_name[0].split('-')[:-1]).split('/')[2:]))
        #print(dataframe['_rlnMicrographName'])
        # number of clusters
        filament_cluster_number = st.number_input('number of cluster', value=3, min_value=1, max_value=20, format="%d")
        filament_pd=filament_basic(filament_umap_2D,helix_name,filament_cluster_number)
        #elbow test
        st.line_chart(calculate_elbow(filament_umap_2D))
        #seperation
        separate=st.button('Separate!')
        if separate:
            pt(separate)



    st.title("visualize the 2Dclass2vec result")
    col1,_,col2 = st.beta_columns((5,0.1,5))
    #with col1d: # left side column, clustering result
    with col1:
        fig = px.scatter(filament_pd,x='umap1',y='umap2',color='label',width=800, height=800)
        selected_points = plotly_events(fig, click_event=True, hover_event=False)

    with col2:
        # Writes a component similar to st.write()
        if selected_points != []:
            index = selected_points[0]['pointIndex']
            st.write('the index is', index)
            row=filament_pd.iloc[int(index)]
            st.write('the filament is', row)
            def_select=dataframe[(dataframe['_rlnHelicalTubeID']==row['FiD']) & (dataframe['_rlnImageName_noseq']==row['MiD'])]
            def_select_xy=def_select[['_rlnCoordinateX','_rlnCoordinateY']]
            st.dataframe(def_select_xy)
            startx,endx,starty,endy = filament_start_end(def_select)
            st.write ('start position ',startx,starty)
            st.write ('end position ',endx,endy)
        else:
            st.write('No select')

@st.cache(persist=True, show_spinner=True)
def cluster(filament_umap_2D, filament_cluster_number):
    umap_predict=KMeans(n_clusters=filament_cluster_number).fit_predict(filament_umap_2D)
    return umap_predict

@st.cache(persist=True, show_spinner=True)
def filament_basic(filament_umap_2D,helix_name,filament_cluster_number):
    filament_ID=[]
    mic_ID=[]
    for i in range(len(helix_name)):
        mic_ID.append('-'.join(helix_name[i].split('-')[:-1]))
        filament_ID.append(helix_name[i].split('-')[-1])
    umap_predict=cluster(filament_umap_2D, filament_cluster_number)
    filament_pd=pd.DataFrame({'umap1':filament_umap_2D[:len(helix_name),0],'umap2':filament_umap_2D[:len(helix_name),1],'FiD':filament_ID,'MiD':mic_ID, 'label': umap_predict[:len(helix_name)].astype('str')})
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
def pt(separate):
    print(separate)

@st.cache(persist=True, show_spinner=True)
def getdata(meta_path,version=0):
    if version<2:
        file_info=EMdata.read_relion(meta_path)
        if version==0:
            #read data (relion3.1)
            dataset=file_info.getRdata_31()
            optics=file_info.extractoptic()
        else:
            #read relion 3.0
            dataset=file_info.getRdata()
        metadata=dataset[0]
        data=dataset[1]
        print(metadata)
        corpus_information=EMdata.process_helical(dataset).extarct_helical_select()
    
    dataframe=pd.DataFrame(data=data,columns=metadata)
    dataframe['_rlnImageName_noseq'] = [x[7:] for x in dataframe['_rlnImageName']]
    corpus_dic,helix_name=corpus_information
    return dataframe, helix_name

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


@st.cache(persist=True, show_spinner=True)
def separate_cluster():
    return 0

@st.cache(persist=True, show_spinner=True)
def write_cluster(filament_cluster_number):

    return 0
main()