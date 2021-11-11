import numpy as np
import scipy.sparse as sparse

def get_residual_variance_sparse(mtx, batch_label, block_size=100):
    # default input format is coo
    # convert sparse matrix to csr format
    mtx = mtx.tocsr()
    
    # compute n_bio = \sum_{i \in b} y_bij, length is G
    n_bio = np.array(mtx.sum(axis=0))[0]

    # compute n_boj = \sum_j y_{bij}, mtx_batch_sum_op: matrix where (i,b) = I(i in b)
    mtx_batch_sum_op = sparse.csr_matrix((np.ones(mtx.shape[1]), batch_label, np.arange(mtx.shape[1]+1)))
    n_boj = mtx.dot(mtx_batch_sum_op).toarray()
    
    # compute n_boo
    n_boo = n_boj.sum(axis=0)

    # loop across blocks
    resid_vars = np.zeros(mtx.shape[0])
    blocks = np.append(np.arange(int(mtx.shape[0] / block_size)) * block_size, mtx.shape[0])
    for i in range(int(mtx.shape[0]/block_size)):
        idx_start, idx_end = blocks[i], blocks[i+1]

        # mean, variance
        mu = n_bio[None,:] * n_boj[idx_start:idx_end, batch_label] / n_boo[batch_label]
        var = mu * (1 - n_boj[idx_start:idx_end, batch_label] / n_boo[batch_label]) 

        # residual
        with np.errstate(divide='ignore'):
            resid = (mtx[idx_start:idx_end,:].toarray() - mu)/np.sqrt(var)
        np.nan_to_num(resid, nan=0, copy=False)

        resid_vars[idx_start:idx_end] = resid.var(axis=1)    
    
    return resid_vars

def pca_conditional_residual_sparse(mtx, batch_label):
    mtx_csr = mtx.tocsr()
    r_r_T = np.zeros((mtx_csr.shape[0], mtx_csr.shape[0]))
    # loop for each batch
    for b in np.unique(batch_label):
        mtx_active = mtx_csr[:, batch_label == b]
        mtx_coo = mtx_active.tocoo()

        # compute n_bio, n_boj, n_boo
        n_bio = np.array(mtx_active.sum(axis=0))[0]
        n_boj = np.array(mtx_active.sum(axis=1)).flatten()
        n_boo = n_boj.sum()

        # compute Y Y^T
        var = n_boj[mtx_coo.row] * n_bio[mtx_coo.col] / n_boo * (1 - n_bio[mtx_coo.col] / n_boo)
        mtx_div_sigma = sparse.csr_matrix((mtx_active.data / np.sqrt(var), mtx_active.indices, mtx_active.indptr))
        mtx_mtx_T = mtx_div_sigma @ mtx_div_sigma.T

        # oompute Y mu^T
        p_bio = n_bio / n_boo
        mtx_mu_T = mtx_div_sigma.dot(np.sqrt(p_bio * (1-p_bio)))[:,None] @ np.sqrt(n_boj)[None,:]

        # compute mu mu^T
        mu_mu_T = (p_bio / (1 - p_bio)).sum() * np.sqrt(n_boj[:,None] @ n_boj[None,:])

        r_r_T += mtx_mtx_T - 2 * mtx_mu_T + mu_mu_T
    
    return r_r_T
