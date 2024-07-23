import numpy as np
from multiprocessing.pool import ThreadPool
import tqdm
'''
wget “http://46.101.39.40/nupack_public/nupack-4.0.1.11.zip”
unzip -q ‘nupack-4.0.1.11.zip’
pip install -U nupack -f nupack-4.0.1.11/package
'''
import nupack as nu
from seqwalk import design
ncores = 32

RT = nu.Model(material="dna", celsius=30)

def hamming(seq1, seq2):
    """
    compute hamming distance of two sequences

    Args:
        seq1: string
        seq2: string

    Returns:
        int
            hamming distance
    """
    assert (len(seq1) == len(seq2)), "Sequences must have equal length"
    return sum(seq1[i] != seq2[i] for i in range(len(seq1)))

def hamming_matrix(library):
    """
    matrix where element i, j is H distance between seq i and seq j

    Args:
        library: list of strings of equal length

    Returns:
        NxN numpy array : hdists
            hamming distance "heatmap"
    """

    hdists = np.zeros((len(library), len(library)))
    for i in range(len(library)):
        for j in range(i+1, len(library)):
            d = hamming(library[i], library[j])
            hdists[i, j] = d
            hdists[j, i] = d
    return hdists

def np_crosstalk(seq1, seq2, model=RT, conc=1e-6, RCfree=False):
    """
    compute thermodynamic binding probability

    Args:
        seq1: string (DNA seq)
        seq2: string (DNA seq)
        model: nupack conditions (default RT)
        conc: molar concentrations of strands (default 1e-6)
        RCfree: Bool, True if library is to be RC free

    Returns:
        float
            equilibrium concentration of on target binding
    """
    A = nu.Strand(seq1, name='A')
    B = nu.Strand(seq2, name='B')
    ct1 = nu.Complex([A, ~A], name="c1")
    tRC = nu.Tube(strands={A: conc, ~B: conc, ~A: conc},
                 name='t1', complexes=nu.SetSpec(max_size=2))
    tRCfree = nu.Tube(strands={A: conc,~A: conc, B: conc,  ~B: conc},
                 name='tRC', complexes=nu.SetSpec(max_size=2))
    tube_results = nu.tube_analysis(tubes=[[tRC, tRCfree][RCfree]], model=model)
    print(tube_results)
    print("BRUH")
    if RCfree:
        return [tube_results.tubes[tRCfree].complex_concentrations[c] for c in [ct1]]
    return [tube_results.tubes[tRC].complex_concentrations[c] for c in [ct1]]

def nupack_matrix(library, model=RT, conc=1e-6, RCfree=False):
    """
    matrix where element i, j is binding prob between seq i and seq j

    Args:
        library: list of seqs
        model: nupack conditions (default RT)
        conc: molar concentrations of strands (default 1e-6)
        RCfree: Bool, True if library is to be RC free

    Returns:
        NxN numpy array : np_probs
            binding probability heatmap
    """

    np_probs = np.zeros((len(library), len(library)))
    print("PART 1/2")
    for i in tqdm.tqdm(range(len(library))):
        for j in range(i+1, len(library)):
            p =  1 - np_crosstalk(library[i], library[j], model, conc, RCfree)[0]/conc
            np_probs[i, j] += p
            np_probs[j, i] += p
		
    print("\n PART 2/2")
    for i in tqdm.tqdm(range(len(library))):
        np_probs[i, i] += np_crosstalk(library[i], library[i], model, conc, RCfree)[0]/conc
    return np_probs

def ij_np_mat_mp_helper(library, model, conc, RCfree, i, j, np_probs):
    assert(np_probs[i,j] == 0 and np_probs[j,i] == 0) 
    # safety check in case i misunderstood original code, avoid race conditions
    if i != j:
        p =  1 - np_crosstalk(library[i], library[j], model, conc, RCfree)[0]/conc
    else:
        p = np_crosstalk(library[i], library[i], model, conc, RCfree)[0]/conc
    np_probs[i, j] += p
    np_probs[j, i] += p

def nupack_matrix_mp(library, model=RT, conc=1e-6, RCfree=False):
    """
    matrix where element i, j is binding prob between seq i and seq j

    Args:
        library: list of seqs
        model: nupack conditions (default RT)
        conc: molar concentrations of strands (default 1e-6)
        RCfree: Bool, True if library is to be RC free

    Returns:
        NxN numpy array : np_probs
            binding probability heatmap
    """

    np_probs = np.zeros((len(library), len(library)))
    print("BEGINNING")
    # create the thread pool
    n = len(library)
    with ThreadPool(ncores) as pool:
        with tqdm.tqdm(total= n*(n+1)/2) as pbar:
            for i in range(n):
                for j in range(i, n):
                    # issue task
                    _ = pool.apply_async(ij_np_mat_mp_helper, args=(library, model, conc, RCfree, i, j, np_probs), callback=lambda x: pbar.update(1))	
        # close the pool
        pool.close()
        # wait for tasks to complete
        pool.join()

    return np_probs


# run my design job
library = design.max_size(12, 6, alphabet="ACGT")
nu_mat = nupack_matrix(library, conc=1e-8, RCfree=True)
np.save("nu_mat", nu_mat)
print(nu_mat)
