import EMdata
import numpy as np


def import_metafile(file_path, datatype):
    if datatype < 2:
        file_info = EMdata.read_relion(file_path)
        if datatype == 0:
            # read data (relion3.1)
            dataset = file_info.getRdata_31()
            optics = file_info.extractoptic()
        else:
            # read relion 3.0
            dataset = file_info.getRdata()
        metadata = dataset[0]
        print(metadata)
        data = dataset[1]
        print(data[0])
        corpus_information = EMdata.process_helical(dataset).extarct_helical_select()
    else:
        # read cryosparc
        dataset = np.load(file_path)
        corpus_information = EMdata.process_cryosparc_helical(dataset).extract_helical()
    corpus_dic, helix_name = corpus_information
    corpus = list(corpus_dic.values())
    corpus_backup = corpus[:]

    corpus_ignore = []
    for i in range(len(corpus)):
        corpus_row = []
        lst = corpus[i]
        count = lst[0][1]
        for j in range(len(lst)):
            particle = lst[j]
            if count == int(particle[1]):
                corpus_row.append(particle[0])
                count += 1
            else:
                while 1:
                    if count == int(lst[j][1]):
                        corpus_row.append(particle[0])
                        count += 1
                        break
                    corpus_row += [0]
                    count += 1
        corpus_ignore.append(corpus_row)

    corpus_length_histogram = []
    for i in range(len(corpus_ignore)):
        corpus_length_histogram.append(len(corpus_ignore[i]))
    max_length = max(corpus_length_histogram)
    return corpus_ignore, max_length