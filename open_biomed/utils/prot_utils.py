import numpy as np
import math

def get_normalized_ctd(proteins):
    from PyBioMed.PyProtein import CTD
    ctds = []
    for prot in proteins:
        ctds.append(np.array(list(CTD.CalculateCTD(prot).values())))
    ctds = np.array(ctds)
    for i in range(ctds.shape[1]):
        mean = np.mean(ctds[:, i])
        var = np.var(ctds[:, i])
        ctds[:, i] = (ctds[:, i] - mean) / math.sqrt(var)
    for i in range(ctds.shape[0]):
        ctds[i] /= np.linalg.norm(ctds[i])
    return ctds