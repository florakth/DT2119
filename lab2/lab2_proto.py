import numpy as np
from lab2_tools import *

def concatTwoHMMs(hmm1, hmm2):
    """ Concatenates 2 HMM models

    Args:
       hmm1, hmm2: two dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be different for each)

    Output
       dictionary with the same keys as the input but concatenated models:
          startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models
   
    Example:
       twoHMMs = concatHMMs(phoneHMMs['sil'], phoneHMMs['ow'])

    See also: the concatenating_hmms.pdf document in the lab package
    """
    con_hmm = {}
    
    M = hmm1['transmat' ].shape[0]
    N = hmm2['transmat' ].shape[0]
    K = M + N - 1
    
    # concatenate start probability
    sp_1 = hmm1['startprob' ][:(M-1)]
    sp_2 = hmm1['startprob' ][(M-1)] * hmm2['startprob']
    con_hmm['startprob' ] = np.concatenate((sp_1,sp_2))
   # print(con_hmm['startprob' ])

    # concatenate transition matrix
    con_hmm['transmat' ] = np.zeros([K,K])
    con_hmm['transmat' ][:M,:M] = hmm1['transmat']
    con_hmm['transmat' ][M-1:,M-1:] = hmm2['transmat']
    con_hmm['transmat' ][:M-1,M-1:] = np.outer(hmm1['transmat'][:M-1,-1],hmm2['startprob'])
    con_hmm['transmat' ][-1,-1] = 1

    # concatenate means abd covariances
    con_hmm['means' ] = np.vstack((hmm1['means' ], hmm2['means' ]))

    con_hmm['covars' ] = np.vstack((hmm1['covars' ], hmm2['covars' ]))

    # concatenate names
    con_hmm['name' ] = hmm1['name' ] + ',' + hmm2['name']

    return con_hmm
    

    
    
# this is already implemented, but based on concat2HMMs() above
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: dictionary of models indexed by model name. 
       hmmmodels[name] is a dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models:
         startprob: K+1 array with priori probability of state
          transmat: (K+1)x(K+1) transition matrix
             means: KxD array of mean vectors
            covars: KxD array of variances

    K is the sum of the number of emitting states from the input models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    concat = hmmmodels[namelist[0]]
    for idx in range(1,len(namelist)):
        concat = concatTwoHMMs(concat, hmmmodels[namelist[idx]])
    return concat


def obsloglik(X,means,cov):
    
     return log_multivariate_normal_density_diag(X, means, cov)
    

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """
    result = 0
    log_weights = np.log(weights)
    for lik in log_emlik:
        w_lik = logsumexp(log_weights+lik)
        result += w_lik
      
    return result
   

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    
    forward_prob = np.zeros((N,M))
    
    forward_prob[0, :] = log_startprob[:-1] + log_emlik[0, :]

    for i in range(1, N):
        for j in range(M):
            forward_prob[i, j] = logsumexp(forward_prob[i-1, :] + log_transmat[:-1, j]) + log_emlik[i, j]
    
    return forward_prob


def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """
    N, M = log_emlik.shape
    backward_prob = np.zeros((N,M))
    
    for n in range(N-2, -1, -1):
        for j in range(M):
            backward_prob[n,j] = logsumexp(log_transmat[j,:-1]
                                           + log_emlik[n+1,:]
                                           + backward_prob[n+1,:])

    return backward_prob

def viterbi(log_emlik, log_startprob, log_transmat, forceFinalState=True):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j
        forceFinalState: if True, start backtracking from the final state in
                  the model, instead of the best state at the last time step

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """
    N, M = log_emlik.shape
    
    V = np.zeros((N,M))
    B = np.zeros((N,M))

    # Initialization
    V[0, :] = log_startprob[:-1] + log_emlik[0, :]
    B[0, :] = 0

    # Induction
    for i in range(1, N):
        for j in range(M):
            V[i, j] = np.max(V[i-1, :] + log_transmat[:-1, j]) + log_emlik[i, j]
            B[i, j] = np.argmax(V[i-1, :] + log_transmat[:-1, j])

    # Termination
    best = np.max(V[-1, :])
    sN = np.argmax(B[-1, :])

    # Backtracking
    st = np.zeros(N)
    st[-1] = sN
    for i in reversed(range(N-1)):
        st[i] = B[i+1, int(st[i+1])]


    viterbi_loglik = best
    viterbi_path = st

    return viterbi_loglik, viterbi_path

    

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """
    log_gamma = log_alpha + log_beta - logsumexp(log_alpha[-1,:])
    
    return log_gamma

def updateMeanAndVar(X, log_gamma, varianceFloor=5.0):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
         varianceFloor: minimum allowed variance scalar
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state

         
    """
    N, D = X.shape
    M = log_gamma.shape[1]
    
    means  = np.zeros((M, D))
    covars = np.zeros((M, D))

    for i in range(M):
        means[i, :] = np.dot(X.T, np.exp(log_gamma[:, i])) / np.sum(np.exp(log_gamma[:, i]))

        C = X.T - means[i,:].reshape((D, 1))

        res = 0
        for j in range(N):
            res = res + np.exp(log_gamma[j, i]) * np.outer(C[:, j], C[:, j])

        covars[i, :] = np.diag(res) / np.sum(np.exp(log_gamma[:, i]))

    covars[covars < varianceFloor] = varianceFloor

    return means, covars
