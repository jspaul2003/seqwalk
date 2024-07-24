import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import tqdm
import nupack as nu
from seqwalk import design
from analysis import *
ncores = cpu_count()
RT = nu.Model(material="dna", celsius=30)

# run my design job
library = design.max_size(14, 7, alphabet="ACGT")
print("LENGTH TO ANALYZE: " + str(len(library)))
nu_mat = nupack_matrix_mp(library, conc=1e-8, RCfree=True)
np.save("nu_mat3", nu_mat)
print(nu_mat)
