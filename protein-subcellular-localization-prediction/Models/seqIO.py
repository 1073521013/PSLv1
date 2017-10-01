# -*- coding: utf-8 -*-

from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from time import time
from scipy.stats import expon,randint
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn import preprocessing

raw_data_chlo = SeqIO.parse("../datasets/dataset0/chloroplast.fasta", "fasta");
training_chlo = [(seq_record, "chloroplast") for seq_record in raw_data_chlo]

raw_data_cyto = SeqIO.parse("../datasets/dataset0/cytoplasmic.fasta", "fasta");
training_cyto = [(seq_record, "cytoplasmic") for seq_record in raw_data_cyto]

raw_data_ER = SeqIO.parse("../datasets/dataset0/ER.fasta", "fasta");
training_ER = [(seq_record, "ER") for seq_record in raw_data_ER]

raw_data_extra = SeqIO.parse("../datasets/dataset0/extracellular.fasta", "fasta");
training_extra = [(seq_record, "extracellular") for seq_record in raw_data_extra]

raw_data_Golgi = SeqIO.parse("../datasets/dataset0/Golgi.fasta", "fasta");
training_Golgi = [(seq_record, "Golgi") for seq_record in raw_data_Golgi]

raw_data_lyso = SeqIO.parse("../datasets/dataset0/lysosomal.fasta", "fasta");
training_lyso = [(seq_record, "lysosomal") for seq_record in raw_data_lyso]

raw_data_mito= SeqIO.parse("../datasets/dataset0/mitochondrial.fasta", "fasta");
training_mito = [(seq_record, "mitochondrial") for seq_record in raw_data_mito]

raw_data_nucl = SeqIO.parse("../datasets/dataset0/nuclear.fasta", "fasta");
training_nucl = [(seq_record, "nuclear") for seq_record in raw_data_nucl]

raw_data_pero = SeqIO.parse("../datasets/dataset0/peroxisomal.fasta", "fasta");
training_pero = [(seq_record, "peroxisomal") for seq_record in raw_data_pero]

raw_data_plas = SeqIO.parse("../datasets/dataset0/plasma_membrane.fasta", "fasta");
training_plas = [(seq_record, "plasma_membrane") for seq_record in raw_data_plas]

raw_data_vacu = SeqIO.parse("../datasets/dataset0/vacuolar.fasta", "fasta");
training_vacu = [(seq_record, "vacuolar") for seq_record in raw_data_vacu]
# test data
blind_tests = SeqIO.parse("../datasets/blind.fasta", "fasta");
testing_blind = [(seq_record, "blind") for seq_record in blind_tests];
training = [];
training.extend(training_chlo);
training.extend(training_cyto);
training.extend(training_ER);
training.extend(training_extra);
training.extend(training_Golgi);
training.extend(training_lyso);
training.extend(training_mito);
training.extend(training_nucl);
training.extend(training_pero);
training.extend(training_plas);
training.extend(training_vacu);

def bio_feat(record):
    clean_seq = str(MutableSeq(record.seq)).replace("X", "")
    clean_seq = clean_seq.replace("U", "C")
    clean_seq = clean_seq.replace("B", "N")
    clean_seq = clean_seq.replace('Z', 'Q')
    clean_seq = MutableSeq(clean_seq).toseq()

    ### features
    seq_length = len(str(clean_seq))
    analysed_seq = ProteinAnalysis(str(clean_seq))
    molecular_weight = analysed_seq.molecular_weight()
    amino_percent = analysed_seq.get_amino_acids_percent().values()
    isoelectric_points = analysed_seq.isoelectric_point()
    count = analysed_seq.count_amino_acids().values()
    # aromaticity = analysed_seq.aromaticity()
    instability_index = analysed_seq.instability_index()
    # hydrophobicity = analysed_seq.protein_scale(ProtParamData.kd, 5, 0.4)
    secondary_structure_fraction = analysed_seq.secondary_structure_fraction()
    return np.array([seq_length, molecular_weight, isoelectric_points, instability_index] + list(secondary_structure_fraction) + list(count) + list(amino_percent))

features = ([bio_feat(record) for record, _ in training])
labels = ([label for _, label in training])
## cross validation
# normalize features before SVM
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=0)

## SVM
parameters={'gamma':list(np.arange(0,0.1,0.01)),'C':list(range(1,20))}
clf=GridSearchCV(SVC(probability=True),parameters)
clf.fit(X_train, y_train)
print(clf.best_estimator_)
print("score svm: ", clf.score(X_test, y_test))
#RandomizedSearchCV类实现了在参数空间上进行随机搜索的机制
#def report(results,n_top=3):
#    for i in range(1,n_top+1):
#        candidates=np.flatnonzero(results['rank_test_score']==i)
#        for candidate in candidates:
#            print("Model with rank:{0}".format(i))
#            print("Mean validation score:{0:.3f}(std:{1:.3f})".format(
#                results['mean_test_score'][candidate],
#                results['std_test_score'][candidate]))
#            print("Parameters:{0}".format(results['params'][candidate]))
#            print("")
#
##设置想要优化的超参数以及他们的取值分布
##param_dist={'C':expon(scale=10),'gamma':expon(scale=.1)}
#param_dist={'C':randint(1,30),'gamma':expon(0.01)}
##开启超参数空间的随机搜索
#n_iter_search=20
#random_search=RandomizedSearchCV(SVC(),param_distributions=param_dist,n_iter=n_iter_search)
#start=time()
#random_search.fit(X_train,y_train)
#print("RandomizedSearchCV took %.3f seconds for %d candidates"
#      "parameter settings."%((time()-start),n_iter_search))
#print(param_dist)
#report(random_search.cv_results_)
