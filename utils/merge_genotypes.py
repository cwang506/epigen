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


def confirm_paths_exist(list_of_corpus_ids, list_of_pops):
    assert(len(list_of_corpus_ids) == len(list_of_pops))
    for i in range(len(list_of_corpus_ids)):
        corpus_id = list_of_corpus_ids[i]
        pop = list_of_pops[i]
        
