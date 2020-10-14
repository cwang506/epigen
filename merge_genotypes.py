import json
import bz2
import os
import numpy as np

def merge_corpora(list_of_corpus_ids, list_of_pops, final_corpus_id, append='SNPS'):
    pop = "MIX" if len(set(list_of_pops)) == 1 else list_of_pops[0]
    genotype = None
    snps = []
    mafs = None
    cum_mafs = []
    axis = 0 if append == 'SNPS' else 1
    num_snps = 0
    num_inds = 0

    print("Merging genotype corpora")
    confirm_paths_exist(list_of_corpus_ids, list_of_pops)

    #merge the corpora
    genotype_list = []
    # snps_list = []
    for pos in range(len(list_of_corpus_ids)):
        print("loading corpus: %s" %list_of_corpus_ids[pos])
        corpus_id = list_of_corpus_ids[pos]
        pop = list_of_pops[pos]
        fname = os.path.join("corpora", str(corpus_id) + "_" + pop + "_genotype")
        fname_json = fname + ".json"
        fname_npy = fname+".npy"
        fname_genotype_bz2 = fname_json + ".bz2"
        genotype_to_be_added = None
        if os.path.exists(fname_npy):
            print("using pickle")
            genotype_to_be_added = np.load(fname_npy)
        elif os.path.exists(fname_json):
            with open(fname_json, 'rt') as jsonfile:
                genotype_to_be_added = np.asarray(json.load(jsonfile), dtype = np.uint8)
        elif os.path.exists(fname_genotype_bz2):
            with bz2.open(fname_genotype_bz2, "rt", encoding="ascii") as zipfile:
                genotype_to_be_added = np.asarray(json.load(zipfile), dtype = np.uint8)
        if genotype_to_be_added is not None:
            genotype_list.append(genotype_to_be_added)

        fname_snp = os.path.join("corpora", str(corpus_id) + "_" + pop + "_snps.json")
        fname_snp_bz2 = fname_snp + ".bz2"
        snp_to_be_added = None
        if os.path.exists(fname_snp):
            with open(fname_snp, "rt") as jsonfile:
                snp_to_be_added = json.load(jsonfile)
        elif os.path.exists(fname_snp_bz2):
            with bz2.open(fname_snp_bz2, "rt", encoding = "ascii") as zipfile:
                snp_to_be_added = json.load(zipfile)
        if snp_to_be_added is not None:
            snps += snp_to_be_added
    print("concatenating genotype")
    genotype = np.concatenate(genotype_list, axis=axis)

    num_snps = float(np.shape(genotype)[0])
    num_inds = float(np.shape(genotype)[1])

    print("computing mafs")
    counter = 1
    mafs = np.apply_along_axis(np.sum, 1, np.sign(genotype)) / num_inds
    print("computing cumulative maf distributions")
    sorted_mafs = sorted(mafs.tolist())
    current_maf = sorted_mafs[0]
    for pos in range(1, len(sorted_mafs)):
        if sorted_mafs[pos] != current_maf:
            cum_mafs.append([current_maf, counter])
            current_maf = sorted_mafs[pos]
        counter += 1
    cum_mafs.append([current_maf, counter])

    print("serializing the genotype corpus") #pickle genotype
    # with open("corpora_manual/" + str(final_corpus_id) + "_" + pop + "_genotype.json", "wt", encoding="ascii") as jsonfile:
    #     json.dump(genotype.tolist(), jsonfile)
    np.save("corpora_manual/" + str(final_corpus_id)+"_"+pop+"_genotype", genotype, allow_pickle=True)
    with open("corpora_manual/" + str(final_corpus_id) + "_" + pop  + "_snps.json", "wt", encoding="ascii") as jsonfile:
        json.dump(snps, jsonfile)
    with open("corpora_manual/" + str(final_corpus_id) + "_" + pop + "_mafs.json", "wt", encoding="ascii") as jsonfile:
        json.dump(mafs.tolist(), jsonfile)
    with open("corpora_manual/" + str(final_corpus_id) + "_" + pop + "_cum_mafs.json", "wt", encoding="ascii") as jsonfile:
        json.dump(cum_mafs, jsonfile)

def confirm_paths_exist(list_of_corpus_ids, list_of_pops):
    assert(len(list_of_corpus_ids) == len(list_of_pops))
    for i in range(len(list_of_corpus_ids)):
        corpus_id = list_of_corpus_ids[i]
        pop = list_of_pops[i]
        fname_genotype = os.path.join("corpora", str(corpus_id) + "_" + pop + "_genotype.json")
        fname_genotype_bz2 = fname_genotype + ".bz2"
        if not os.path.exists(fname_genotype):
            if not os.path.exists(fname_genotype_bz2):
                raise OSError("Neither %s or %s exist" %(fname_genotype, fname_genotype_bz2))

        fname_snps = os.path.join("corpora", str(corpus_id) + "_"+pop+"_snps.json")
        fnames_snps_bz2 = fname_snps + ".bz2"
        if not os.path.exists(fname_snps):
            if not os.path.exists(fnames_snps_bz2):
                raise OSError("Neither %s or %s exist" %(fname_snps, fnames_snps_bz2))

if __name__ == "__main__":
    merge_corpora([i for i in range(1, 23)], ['CEU' for i in range(22)], 122, 'SNPS')