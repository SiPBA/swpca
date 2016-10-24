# module swpca

import numpy as np
from scipy.linalg import svd
from scipy.stats import f_oneway

def swpca(dataset, catvar, k=0.05, trset=False):
    N = dataset.shape[0]
    if not trset:
        trset=np.ones(N)
        training = False
    else:
        training = True

    # Per subject mean substraction (doesn't depend on training set)
    subjMean = dataset.mean(axis=1) 
    X = (dataset.transpose()-subjMean).transpose()
    del dataset    
    
    # Standardize the data (depends on training set)
    meanTr = X[trset==1,:].mean(axis=0)
    X = X - meanTr
    varTr = X[trset==1,:].var(axis=0)
    X = X/varTr
    if training:
        XNTEST = X[trset!=1,:]
    XNTRAIN = X[trset==1,:]
    del X 
        
    # singular value decomposition factorises your data matrix such that:
    # 
    #   M = U*S*V.T     (where '*' is matrix multiplication)
    # 
    # * U and V are the singular matrices, containing orthogonal vectors of
    #   unit length in their rows and columns respectively.
    #
    # * S is a diagonal matrix containing the singular values of M - these 
    #   values squared divided by the number of observations will give the 
    #   variance explained by each PC.
    #
    # * if M is considered to be an (observations, features) matrix, the PCs
    #   themselves would correspond to the rows of S^(1/2)*V.T. if M is 
    #   (features, observations) then the PCs would be the columns of
    #   U*S^(1/2).
    #
    # * since U and V both contain orthonormal vectors, U*V.T is equivalent 
    #   to a whitened version of M.
    U, s, Wt = svd(XNTRAIN, full_matrices=False)
    W = Wt.T
    
    # sort the PCs by descending order of the singular values (i.e. by the
    # proportion of total variance they explain)
    ind = np.argsort(s)[::-1]
    U = U[:, ind]
    s = s[ind]
    W = W[:, ind]
    
    STr = XNTRAIN.dot(W)
    if training:
        STe = XNTEST.dot(W)
        
    # we apply One-Way-ANOVA 
    # The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
    # 
    #     The samples are independent.
    #     Each sample is from a normally distributed population.
    #     The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
    labs = np.unique(catvar)
    F,p_val = f_oneway(STr[catvar[trset==1]==labs[0],:], STr[catvar[trset==1]==labs[1],:])
    
    # Compute the weightings
    weights = 1-np.exp(-p_val/k)
    
    # Reconstruct the signals. 
    A = np.linalg.pinv(W)
    weightMat = np.diag(weights)
    if training:
        XTRhat = STr.dot(weightMat).dot(A) 
        XTEhat = STe.dot(weightMat).dot(A)
        Xhat = np.zeros([N,XTRhat.shape[1]])
        Xhat[trset==1,:] = XTRhat
        Xhat[trset!=1,:] = XTEhat
        del XTRhat,XTEhat
    else:
        Xhat = STr.dot(weightMat).dot(A) 
        
    Xhat = Xhat*varTr
    Xhat = Xhat+meanTr
    Xhat = (Xhat.T+subjMean).T
    
    return Xhat, weights, A

