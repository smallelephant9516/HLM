import numpy as np
import os
import sys

class read_relion(object):
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

        for star_line in open(self.file).readlines()[19:]:
            if star_line.find("_rln") != -1:
                var = star_line.split()
                Rvar.append(var[0])
            #    Rvar_len = Rvar_len+1
            elif star_line.find("data_") != -1 or star_line.find("loop_") != -1 or len(star_line.strip()) == 0:
                continue

            else:
                Rdata.append(star_line.split())

        return Rvar, Rdata


#the data is read_relion(sys.argv[1]).getRdata()
class process_helical():
    def __init__(self, dataset, classnumber=50):
        self.metadata=dataset[0]
        self.data=dataset[1]
        self.classnumber=classnumber
    def extarct_helical(self):
        data=self.data
        M = self.metadata.index('_rlnImageName')
        H = self.metadata.index('_rlnHelicalTubeID')
        C = self.metadata.index('_rlnClassNumber')
        print('finish reading')
        # extract helical parameters
        helicaldic = {}
        helicalnum = []
        count = -1
        for particle in data:
            ID = particle[M][7:] + '-' + str(particle[H])
            if ID in helicalnum:
                n = str(count)
                lst = helicaldic[n]
                lst.append(particle[C])
                helicaldic[n] = lst
            else:
                helicalnum.append(ID)
                n = str(helicalnum.index(ID))
                count += 1
                helicaldic[n] = [particle[C]]
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
            file.writelines("%s\n" % "data_particle")
            file.writelines("%s\n" % "           ")
            file.writelines("%s\n" % "loop_")

            i=0
            for item in self.metadata:
                i+=1
                # fullstr = ' '.join([str(elem) for elem in item ])
                file.writelines("%s %s\n" % (item,'#{}'.format(i)))



