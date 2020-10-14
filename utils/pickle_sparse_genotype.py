import sys
import json
import bz2
from scipy.sparse import lil_matrix, csc_matrix, coo_matrix, vstack, save_npz
import os
import numpy as np
from tqdm import tqdm

# import argparse


def convert_to_sparse_matrix(corpora_id, pop, firsthalf = True):
    fname = os.path.join("../corpora", "%s_%s_genotype"%(corpora_id, pop))
    print(fname)
    fname_npy = fname + ".npy"
    fname_json = fname + ".json"
    genotype = None
    if os.path.exists(fname_npy):
        genotype = np.load(fname_npy)
    elif os.path.exists(fname_json):
        with open(fname_json, 'rb') as f:
            genotype = np.asarray(json.load(f), dtype = np.uint8)
    elif os.path.exists(fname_json + ".bz2"):
        with bz2.open(fname_json + ".bz2", 'rt', encoding="ascii") as f:
            genotype = np.asarray(json.load(f), dtype=np.uint8)
    if genotype is None:
        raise OSError("Neither %s, %s, or %s exist" %(fname_npy, fname_json, fname_json+".bz2")) 

    #make the transpose of the matrix
    lis = []
    rows = range(genotype.shape[0])#range(genotype.shape[0]//2) if firstHalf else range(genotype.shape[0]//2, genotype.shape[0])
    for r in tqdm(rows):
        row = genotype[r:r+1, :]
        inds = np.where(row != 0)
        data = row[inds]
        sparse_r = coo_matrix((data, inds), shape = row.shape)
        lis.append(sparse_r)
    
    print("v-stacking")
    sparse_genotype = vstack(lis)
    print("converting to csc_matrix")
    csc_sparse_genotype = csc_matrix(sparse_genotype)
    return csc_sparse_genotype





if __name__=="__main__":
    corpora_id = sys.argv[1]
    pop = sys.argv[2]
    # firstHalf = sys.argv[3]
    # suffix = 1 if firstHalf else 2
    csc_sparse_genotype = convert_to_sparse_matrix(corpora_id, pop)
    fname = os.path.join("../corpora", "%s_%s_sparse_genotype"%(corpora_id, pop))
    save_npz(fname, csc_sparse_genotype)    
    