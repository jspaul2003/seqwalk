import numpy as np
from multiprocessing.pool import ThreadPool
import tqdm
from itertools import compress
import pickle
import nupack as nu
from analysis import *
from seqwalk import design
RT = nu.Model(material="dna", celsius=30)
ncores = 32

library=design.max_size(12, 6, alphabet="ACGT")
nu_mat = np.load("nu_mat.npy")
threshold = 0.95
choices = np.diag(nu_mat) > threshold/2
pick = list(compress(library, choices))
with open("pick1", "wb") as fp:   #Pickling
    pickle.dump(pick, fp)

print("ANALYSIS ON PICK #1 BEGINNING\n")
nu_mat2 = nupack_matrix_mp(pick, conc=1e-8, RCfree=True)
np.save("nu_mat2", nu_mat2)
print(nu_mat2)


