import numpy as np
from tqdm import tqdm

def to_clu_sparse(data):
    s = "%d %d %d" % (data.shape[0], data.shape[1], np.sum(data))
    s_row = [""] * data.shape[0]
    non_zero_row, non_zero_col = np.where(data > 0)
    for i in tqdm(range(len(non_zero_row))):
        s_row[non_zero_row[i]] += " %d %f" % (non_zero_col[i] + 1, data[non_zero_row[i], non_zero_col[i]])
    return "\n".join([s] + s_row)