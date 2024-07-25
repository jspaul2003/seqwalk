
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import tqdm
import nupack as nu
from seqwalk import design
from analysis import *
ncores = cpu_count()
RT = nu.Model(material="dna", celsius=20)

# run my design job
library = design.max_size(12, 7, alphabet="ACGT", RCfree=True)
print("LENGTH TO ANALYZE: " + str(len(library)))
nu_mat = nupack_matrix_mp(library, conc=1e-8, RCfree=True)
np.save("nu_mat12.7", nu_mat)
print(nu_mat)


