import numpy as np
import os
import sys
import time
import pandas as pd
import gemmi

start_time= time.time()

class read_data_df():
    def __init__(self,file):
        self.file = file
    
    def star2dataframe(self):
        starFile=self.file
        from gemmi import cif
        star = cif.read_file(starFile)
        if len(star) == 2:
            optics = cif.Document()
            optics.add_copied_block(star[0])
            del star[0]
            js = optics.as_json(True)  # True -> preserve case
            optics = pd.read_json(js).T
            d = {c.strip('_'): optics[c].values[0] for c in optics}
            optics = pd.DataFrame(d)
        else:
            optics = None
        js = star.as_json(True)  # True -> preserve case
        data = pd.read_json(js).T
        d = {c.strip('_'): data[c].values[0] for c in data}
        data = pd.DataFrame(d)
        
        assert("rlnImageName" in data)
        tmp = data["rlnImageName"].str.split("@", expand=True)
        indices, filenames = tmp.iloc[:,0], tmp.iloc[:, -1]
        indices = indices.astype(int)-1
        data["pid"] = indices
        data["filename"] = filenames
    
        if optics is not None:
            og_names = set(optics["rlnOpticsGroup"].unique())
            for gn, g in data.groupby("rlnOpticsGroup", sort=False):
                if gn not in og_names:
                    print(f"ERROR: optic group {gn} not available ({sorted(og_names)})")
                    sys.exit(-1)
                ptcl_indices = g.index
                og_index = optics["rlnOpticsGroup"] == gn
                if "rlnPixelSize" in optics:
                    data.loc[ptcl_indices, "apix"] = optics.loc[og_index, "rlnPixelSize"].astype(float).iloc[0]
        if "rlnPixelSize" in data:
            data.loc[:, "apix"] = data["rlnPixelSize"]
        if "rlnClassNumber" in data:
            data.loc[:, "class"] = data["rlnClassNumber"]
        if "rlnHelicalTubeID" in data:
            data.loc[:, "helicaltube"] = data["rlnHelicalTubeID"].astype(int)-1
        if "rlnAnglePsiPrior" in data:
            data.loc[:, "phi0"] = data["rlnAnglePsiPrior"].astype(float).round(3) - 90.0
    
        return data

class read_relion():
    def __init__(self, file):
        self.file = file

    def getRdata(self):
        Rvar = []  # read the variables metadata
        Rdata = []  # read the data

        for star_line in open(self.file).readlines():
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue

            else:
                Rdata.append(star_line.split())

        return Rvar, Rdata

    def extractoptic(self):
        optics=[]
        for star_line in open(self.file).readlines()[0:19]:
            optics.append(star_line.split())
        return optics

    def getRdata_31(self):
        Rvar = []  # read the variables metadata
        Rdata = []  # read the data

        for star_line in open(self.file).readlines()[20:]:
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue

            else:
                Rdata.append(star_line.split())

        return Rvar, Rdata
class process_helical_df():
    def __init__(self,dataframe):
        self.df=dataframe
    def extract_helical_select(self):
        dataframe=self.df
        filament_data=dataframe.groupby(['filename','helicaltube'])
        filament_index=list(filament_data.groups.keys())
        helicaldic={}
        helicalnum = []
        dtype=[('class2D',int),('place',int),('index',int)]
        for i in range(len(filament_index)):
            name='-'.join(map(str, filament_index[i]))
            helicaldic[name]=[]
            helicalnum=helicalnum+[name]
        print('The filament number are: ',len(helicalnum))
        print('The number of particles are:',len(dataframe))
        for i in range(len(dataframe)):
            particle=dataframe.iloc[i]
            ID=str(particle['filename']) + '-' + str(particle['helicaltube'])
            helicaldic[ID]=helicaldic[ID]+[(particle['class'],particle['rlnImageName'][0:6],i)]
            if i%100000==0:
                end_time=time.time()
                passed_time=(end_time-start_time)/60
                print(i,'%s mins' % passed_time)
        for i in range(len(helicalnum)):
            lst=np.array(helicaldic[helicalnum[i]],dtype=dtype)
            helicaldic[helicalnum[i]]=np.sort(lst,order='place')
        print('finish converting')
        for i in range(10):
            print(helicaldic[helicalnum[i]])
        return helicaldic, filament_index

#the data is read_relion(sys.argv[1]).getRdata()
class process_helical():
    def __init__(self, dataset, classnumber=50):
        self.metadata=dataset[0]
        self.data=dataset[1]
        self.classnumber=classnumber
    def extarct_helical(self,label=None):
        data=self.data
        M = self.metadata.index('_rlnImageName')
        H = self.metadata.index('_rlnHelicalTubeID')
        if label is None:
            C = self.metadata.index('_rlnClassNumber')
        print('finish reading')
        # extract helical parameters
        helicaldic = {}
        helicalnum = []
        count = -1
        label_id=0
        for particle in data:
            ID = particle[M][7:] + '-' + str(particle[H])
            if ID in helicalnum:
                n = str(count)
                lst = helicaldic[n]
                if label is not None:
                    lst.append(label[label_id])
                    label_id +=1
                else: 
                    lst.append(particle[C])
                helicaldic[n] = lst
            else:
                helicalnum.append(ID)
                n = str(helicalnum.index(ID))
                count += 1
                if label is not None:
                    helicaldic[n]=[label[label_id]]
                    label_id +=1
                else:
                    helicaldic[n] = [particle[C]]
        print('finish converting')
        for i in range(10):
            print(helicaldic[str(i)])
        return helicaldic, helicalnum
    def extarct_helical_select(self):
        data=self.data
        M = self.metadata.index('_rlnImageName')
        H = self.metadata.index('_rlnHelicalTubeID')
        C = self.metadata.index('_rlnClassNumber')
        print('finish reading')
        # extract helical parameters
        helicaldic = {}
        helicalnum = []
        count = -1
        dtype=[('class2D',int),('place',int),('index',int)]
        print('number of particles',len(data))
        for i, particle in enumerate(data):
            if i%100000==0:
                end_time=time.time()
                passed_time=(end_time-start_time)/60
                print(i,'%s mins' % passed_time)
            ID = particle[M][7:] + '-' + str(particle[H])
            if ID in helicalnum:
                n = str(helicalnum.index(ID))
                helicaldic[n]=helicaldic[n]+[(particle[C],particle[M][0:6],i)]
            else:
                helicalnum=helicalnum+[ID]
                n = str(helicalnum.index(ID))
                count += 1
                helicaldic[n] = [(particle[C],particle[M][0:6],i)]
        for i in range(len(helicaldic)):
            lst=np.array(helicaldic[str(i)],dtype=dtype)
            helicaldic[str(i)]=np.sort(lst,order='place')
        print('finish converting')
        #for i in range(5):
        #    print(helicaldic[str(i)])
        return helicaldic, helicalnum
    def extarct_helical_select_fast(self):
        data=self.data
        M = self.metadata.index('_rlnMicrographName')
        H = self.metadata.index('_rlnHelicalTubeID')
        C = self.metadata.index('_rlnClassNumber')
        print('finish reading')
        dataframe=pd.DataFrame(data=data,columns=self.metadata)
        print('finish dataframe')
        groupby_filament=dataframe.groupby(['_rlnMicrographName','_rlnHelicalTubeID'])
        print(groupby_filament.count())
        # extract helical parameters
        print('finish converting')
        #for i in range(5):
        #    print(helicaldic[str(i)])
        return groupby_filament
class process_cryosparc_helical():
    def __init__(self,data):
        self.data=data
    def extract_helical(self):
        data=self.data
        helicaldic = {}
        helicalnum = []
        count = -1
        for particle in data:
            ID = str(os.path.basename(particle[1]))
            if ID in helicalnum:
                n = str(count)
                lst = helicaldic[n]
                lst.append(particle[-2])
                helicaldic[n] = lst
            else:
                helicalnum.append(ID)
                n = str(helicalnum.index(ID))
                count += 1
                helicaldic[n] = [particle[-2]]
        print('finish converting')
        for i in range(10):
            print(helicaldic[str(i)])
        return helicaldic, helicalnum
    
class output_simple_helical():
    def __init__(self, file, data):
        self.data=process_helical(data).extarct_helical()
        self.name=os.path.splitext(file)[0]
    def export(self):
        helicalnum=self.data[1]
        helicaldic=self.data[0]
        with open(self.name+".txt", "a") as f:
            for i in range(len(helicalnum)):
                lst = helicaldic[str(i)]
                for j in range(len(lst)):
                    if j == len(lst) - 1:
                        f.write(lst[j] + '\n')
                    else:
                        f.write(lst[j] + ' ')



class output_star():
    def __init__(self,file,cluster_n,data,metadata):
        self.cluster_n=cluster_n
        self.data=data
        self.metadata=metadata
        self.name=os.path.splitext(file)[0]+"_"+str(cluster_n)+".star"

    def writemetadata(self):
        filename = self.name
        with open(filename, "a") as file:
            file.writelines("%s\n" % "          ")
            file.writelines("%s\n" % "data_")
            file.writelines("%s\n" % "           ")
            file.writelines("%s\n" % "loop_")

            i=0
            for item in self.metadata:
                i+=1
                # fullstr = ' '.join([str(elem) for elem in item ])
                file.writelines("%s %s\n" % (item,'#{}'.format(i)))

    def writecluster(self):
        filename = self.name
        with open(filename, "a") as file:
            for item in self.data:
                full_line='  '.join([str(elem) for elem in item])
                file.writelines("%s\n" % full_line)

    def opticgroup(self,optictitle):
        filename = self.name
        with open(filename,"w") as file:
            for item in optictitle:
                full_line = '  '.join([str(elem) for elem in item])
                file.writelines("%s\n" % full_line)

        with open(filename, "a") as file:
            file.writelines("%s\n" % "          ")
            file.writelines("%s\n" % "data_particles")
            file.writelines("%s\n" % "           ")
            file.writelines("%s\n" % "loop_")

            i=0
            for item in self.metadata:
                i+=1
                # fullstr = ' '.join([str(elem) for elem in item ])
                file.writelines("%s %s\n" % (item,'#{}'.format(i)))



