def conditional_pearson_residuals_pandas(Y, batch_labels):
    """
    Y: numpy matrix rows: cells, columns:genes
    batch_labels: python list of batch labels
    Current implementation is not optimal due to use of pandas
    """
    df_mtx = pd.DataFrame(Y)
    
    # calculate sum_{i \in b} Y_{bij}
    df_sum_y_bij = df_mtx.groupby(batch_label).sum()
    
    # calculate \pi_{bij}
    df_pi_bij_nonormed = df_mtx_used.sum(axis=1)
    df_pi_bij = df_pi_bij_nonormed.values / df_pi_bij_nonormed.groupby(batch_label).sum().loc[batch_label].values
    
    # calculate \mu_{bij} and \sigma^2_{bij}
    mu = df_sum_y_bij.loc[batch_label,:].values * df_pi_bij[:,None]
    sigma2 = mu * (1-df_pi_bij[:,None])
    
    return (Y-mu)/np.sqrt(sigma2)
