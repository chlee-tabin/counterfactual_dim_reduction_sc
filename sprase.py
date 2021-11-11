import numpy as np
import scipy.sparse as sparse

def get_residual_variance_sparse(mtx, batch_label, block_size=100):
    # compute n_bio = \sum_{i \in b} y_bij, length is G
    n_bio = mtx.sum(axis=0)

    # compute n_boj = \sum_j y_{bij}, mtx_batch_sum_op: matrix where (i,b) = I(i in b)
    mtx_batch_sum_op = sparse.csr_matrix((np.ones(mtx.shape[1]), batch_label, np.arange(mtx.shape[1]+1)))
    n_boj = mtx @ mtx_batch_sum_op

    # loop across blocks
    resid_vars = np.zeros(mtx.shape[0])
    blocks = np.append(np.arange(int(mtx.shape[0] / block_size)) * block_size, mtx.shape[0])
    for i in range(int(mtx.shape[0]/block_size)):
        idx_start, idx_end = blocks[i], blocks[i+1]

        # calculate mean and variance
        mu = (np.array(n_bio)[0][None,:] * n_boj[idx_start:idx_end,:].toarray()[:,batch_label]) / n_bio.sum()
        var = mu * (1 - n_boj[idx_start:idx_end,:].toarray()[:,batch_label]/ n_bio.sum()) 

        # calculate residual and residual variance
        with np.errstate(divide='ignore'):
            resid = (mtx[idx_start:idx_end,:].toarray() - mu)/np.sqrt(var)
        np.nan_to_num(resid, nan=0, copy=False)

        resid_vars[idx_start:idx_end] = resid.var(axis=1)    
    
    return resid_vars
