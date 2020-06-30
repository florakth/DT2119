import numpy as np
from lab3_tools import *
from lab2_tools import *
from lab2_proto import *
from tqdm.notebook import tqdm

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    w2ph = []
    if addSilence:
        w2ph.append('sil')
        
    for word in wordList:
        w2ph.extend(pronDict[word])
        if addShortPause:
            w2ph.extend(['sp'])
    
    if addSilence:
        w2ph.append('sil')

    return w2ph

def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    phoneme_index = [ ]
    utteranceHMM = concatHMMs(phoneHMMs, phoneTrans)
    
    stateTrans = [phone + '_' + str(stateid) for phone in phoneTrans \
                                 for stateid in range(phoneHMMs[phone]['means'].shape[0])]
    
    obsloglik = log_multivariate_normal_density_diag(lmfcc,
                                    utteranceHMM['means'],
                                    utteranceHMM['covars'])


    v_loglik, v_path = viterbi(obsloglik,
                               np.log(utteranceHMM['startprob']),
                               np.log(utteranceHMM['transmat']),
                               forceFinalState=True)
    
    for i in v_path:
        phoneme_index.append(stateTrans[int(i)])
        #phoneme_index = phoneme_index + [stateTrans[int(i)]]
        
    return phoneme_index


def get_features(data,dynamic=True):
    '''
    Get fetures from the .npz data, if dynamic=True, get the dynamic features
    Dynamic features: stack 7 MFCC or filterbank features symmetrically distributed around the current time step. 
    At time n, stack the features at times [n − 3, n − 2, n − 1, n, n + 1, n + 2, n + 3]). 
    At the beginning and end of each utterance, use mirrored feature vectors in place of the missing vectors.
    '''
    D_lmfcc = data[0]['lmfcc'].shape[1]
    D_mspec = data[0]['mspec'].shape[1]
    N = sum([len(x['targets']) for x in data])
    
    if dynamic:
        mfcc_features = np.zeros((N,D_lmfcc*7))
        mspec_features = np.zeros((N,D_mspec*7))
    else:
        mfcc_features = np.zeros((N,D_lmfcc))
        mspec_features = np.zeros((N,D_mspec))
    
    targets = []
    
    k = 0
    for x in tqdm(data): 
        n_frames, dim = x['lmfcc'].shape

        ## for each timestep
        for i in range(n_frames):
            if dynamic:
                if i< 3 or i >= n_frames-3:
                    mfcc_features[k,:] = np.hstack(np.pad(x['lmfcc'], pad_width=((3, 3), (0, 0)), mode='reflect')[i:i+7,:])
                    mspec_features[k,:] = np.hstack(np.pad(x['mspec'], pad_width=((3, 3), (0, 0)), mode='reflect')[i:i+7,:])
                else:
                    mfcc_features[k,:] = np.hstack(x['lmfcc'][i-3:i+4,:])
                    mspec_features[k,:] = np.hstack(x['mspec'][i-3:i+4,:])
            else:
                mfcc_features[k,:] = x['lmfcc'][i,:]
                mspec_features[k,:] = x['mspec'][i,:]
            
            k +=1
   
        targets = targets + x['targets']
    
    return mfcc_features, mspec_features, targets


def hmmLoop(hmmmodels, namelist=None):
    """ Combines HMM models in a loop

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to combine, if None,
                 all the models in hmmmodels are used

    D is the dimension of the feature vectors
    M is the number of emitting states in each HMM model (could be
      different in each model)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models
       stateMap: map between states in combinedhmm and states in the
                 input models.

    Examples:
       phoneLoop = hmmLoop(phoneHMMs)
       wordLoop = hmmLoop(wordHMMs, ['o', 'z', '1', '2', '3'])
    """
