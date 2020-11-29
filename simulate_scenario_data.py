import json
import os
import numpy as np
#MAF
from simulate_data import run_script_args
maf_num_inds = 3000
maf_num_snps = 10000
MAF_RANGE_DICT_TRAIN = {
    "MAF_09_1": { "disease_snps": [137485, 61462, 480102, 180185, 1218037, 1222974, 74797, 30467, 345736, 787243],
                 "num_inds":maf_num_inds,
                 "num_snps":maf_num_snps,
                 "pop": "ASW",
                 "corpus_id": 122,
                 "sim_id": 0},
    "MAF_08_09": {"disease_snps":[577094, 1089932, 308554, 1239568, 649686, 1047885, 509363, 1120976, 1273018, 130779],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_07_08": {"disease_snps": [920960, 390669, 1276901, 470863, 481804, 1195753, 769111, 776837, 767922, 55195],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_06_07": {"disease_snps":[731740, 27196, 121517, 179463, 304997, 267007, 1096286, 932489, 141273, 429524],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_05_06": {"disease_snps":[621607, 1300018, 265297, 57486, 949827, 1133870, 123113, 1124858, 658047, 574148],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},      
    "MAF_04_05": {"disease_snps": [1049357, 325975, 1295561, 166341, 28517, 1335989, 1240756, 757148, 1385993, 347987],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_03_04": {"disease_snps":[557827, 1278820, 118596, 304982, 482978, 358449, 1194868, 464556, 998785, 387109],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_02_03": {"disease_snps":[566997, 1031046, 912113, 754045, 956062, 1284571, 302670, 1273640, 1116138, 24976],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_01_02": {"disease_snps":[1047084, 1107340, 507229, 1342591, 882482, 1271943, 1082946, 1102430, 1174389, 1046501],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_0_01": { "disease_snps":[491985, 1137867, 1217080, 950458, 1157083, 313380, 628425, 92487, 962278, 1274553],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0}                             
}

MAF_RANGE_DICT_TEST = {
    "MAF_09_1": {"disease_snps": [137485, 61462, 480102, 180185, 1218037, 1222974, 74797, 30467, 345736, 787243],
                 "num_inds":maf_num_inds,
                 "num_snps":maf_num_snps,
                 "pop": "CEU",
                 "corpus_id": 122,
                 "sim_id": 0},
    "MAF_08_09": {"disease_snps":[577094, 1089932, 308554, 1239568, 649686, 1047885, 509363, 1120976, 1273018, 130779],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_07_08": {"disease_snps": [920960, 390669, 1276901, 470863, 481804, 1195753, 769111, 776837, 767922, 55195],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_06_07": {"disease_snps":[731740, 27196, 121517, 179463, 304997, 267007, 1096286, 932489, 141273, 429524],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_05_06": {"disease_snps":[621607, 1300018, 265297, 57486, 949827, 1133870, 123113, 1124858, 658047, 574148],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},      
    "MAF_04_05": {"disease_snps": [1049357, 325975, 1295561, 166341, 28517, 1335989, 1240756, 757148, 1385993, 347987],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_03_04": {"disease_snps":[557827, 1278820, 118596, 304982, 482978, 358449, 1194868, 464556, 998785, 387109],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_02_03": {"disease_snps":[566997, 1031046, 912113, 754045, 956062, 1284571, 302670, 1273640, 1116138, 24976],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_01_02": {"disease_snps":[1047084, 1107340, 507229, 1342591, 882482, 1271943, 1082946, 1102430, 1174389, 1046501],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_0_01": { "disease_snps":[491985, 1137867, 1217080, 950458, 1157083, 313380, 628425, 92487, 962278, 1274553],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0}                             
}

on_server = False

def get_simulated_data_fname(sim_id, corpus_id, pop, num_inds, num_snps, disease_snps):
    #disease snps come from terminal 
    prefix = "sim/" if not on_server else "/home/cwang506/epigen_data/"
    return prefix+ str(sim_id) + "_" + str(corpus_id) + "_" + pop + "_" +\
            str(num_inds) + "_inds_" + str(num_snps) + "_snps_" + "_".join([str(i) for i in disease_snps])\
            + "_disease_snps"+".json"

def get_corpora_index_from_snps_id(pop, corpus_id):
    #make snps to index mapping
    fname = os.path.join("corpora", '%s_%s_snps.json' %(corpus_id, pop))
    if os.path.exists(fname):
        with open(fname, "r") as f:
            snps_json = json.load(f)
    elif os.path.exists(fname + ".bz2"):
        with open(fname, "rt", encoding="ascii") as f:
            snps_json = json.load(f)
    else:
        raise OSError("Neither %s or %s exist" %(fname, fname + ".bz2"))
    corpora_snps_mapping = {}
    for i, snps_id in enumerate(snps_json):
        corpora_snps_mapping[snps_id[0]] = i
    return corpora_snps_mapping

def get_maf_dict(pop, corpora_id):
    #return snp id to its maf
    fname = '%s_%s_snps.json' %(corpora_id, pop)
    maf_dict = {}
    with open(os.path.join( "corpora", fname), "r") as f:
        snps_json = json.load(f)
    maf_fname = "%s_%s_mafs.json" %(corpora_id, pop)
    with open(os.path.join("corpora", maf_fname), "r") as f:
        mafs_json = json.load(f)
    for i, snps_id in enumerate(snps_json):
        maf_dict[snps_id[0]] = mafs_json[i]
    return maf_dict

def simulate_maf_data():
    num_disease_snps = 10
    first_key = None
    first_disease_snps = None
    for i, key in enumerate(MAF_RANGE_DICT_TRAIN):
        print("generating training data for %s"%key)
        maf_dic = MAF_RANGE_DICT_TRAIN[key]
        # snp_to_maf_training = get_maf_dict(maf_dic['pop'], maf_dic['corpus_id'])
        # # #get the maf min and max from the key name
        # maf_min = int(key.split("_")[1])/10.0
        # maf_max = int(key.split("_")[2])/10.0 if key.split("_")[2]!= "1" else 1
        # print(maf_min, maf_max)
        # possible_snps = [snp_id for snp_id in snp_to_maf_training.keys() \
        #             if snp_to_maf_training[snp_id]>=maf_min and snp_to_maf_training[snp_id]<=maf_max]
        # corpora_snps_mapping = get_corpora_index_from_snps_id(maf_dic['pop'], maf_dic['corpus_id'])
        # possible_snps = [(corpora_snps_mapping[snp_id]) for snp_id in possible_snps]

        if i == 0:
            first_key = key
            #generate training disease snps for the first key
            #generate the total set of disease snps + non-disease snps
            disease_snps = MAF_RANGE_DICT_TEST[key]['disease_snps'] #np.random.choice(possible_snps, num_disease_snps).tolist()
            first_disease_snps = disease_snps.copy()
            print(disease_snps) 
            continue
            run_script_args(maf_dic['pop'],maf_dic['corpus_id'], [maf_dic['sim_id']], "models/param_model_train_simple.xml", maf_dic['num_snps'], maf_dic['num_inds'],
                        disease_snps)
                        
            print("generating test data")
            filename = get_simulated_data_fname(maf_dic['sim_id'], maf_dic['corpus_id'], maf_dic['pop'], maf_dic['num_inds'], maf_dic['num_snps'], disease_snps)
            print(filename)
            with open(filename) as f:
                epigen_json = json.load(f)
            maf_dic_test = MAF_RANGE_DICT_TEST[key]

            corpora_snps_mapping_other_set = get_corpora_index_from_snps_id(maf_dic_test['pop'],maf_dic_test['corpus_id'])
            snps_in_filename = [] 

            #make the snps the same from train to test
            for snps_id in epigen_json['disease_snps']:
                snp = epigen_json['snps'][snps_id]
                index = corpora_snps_mapping_other_set[snp[0]]
                snps_in_filename.append(index)

            for i, snps_id in enumerate(epigen_json['snps']):
                if i in epigen_json['disease_snps']:
                    continue
                index = corpora_snps_mapping_other_set[snps_id[0]]
                snps_in_filename.append(index) #need to map back to CEU data todo
            print(len(snps_in_filename))
            run_script_args(maf_dic_test['pop'],maf_dic_test['corpus_id'], [maf_dic_test['sim_id']], "models/param_model_test_simple.xml", maf_dic_test['num_snps'], maf_dic_test['num_inds'],
                        snps_in_filename) 
            
        else:
            #other 
            if key!= 'MAF_0_01':
                continue
            first_maf_dic = MAF_RANGE_DICT_TRAIN[first_key]
            filename = get_simulated_data_fname(first_maf_dic['sim_id'], first_maf_dic['corpus_id'], 
                        first_maf_dic['pop'], first_maf_dic['num_inds'], first_maf_dic['num_snps'], first_disease_snps)
            # print(filename)
            with open(filename) as f:
                epigen_json = json.load(f)
            corpora_snps_mapping = get_corpora_index_from_snps_id(maf_dic['pop'],maf_dic['corpus_id'])
            snps_in_filename = [] 
            # for snps_id in epigen_json['disease_snps']:
            #     snp = epigen_json['snps'][snps_id]
            #     index = corpora_snps_mapping_other_set[snp[0]]
            #     snps_in_filename.append(index)
            for i, snps_id in enumerate(epigen_json['snps']):
                if i in epigen_json['disease_snps']:
                    continue
                index = corpora_snps_mapping[snps_id[0]]
                snps_in_filename.append(index) 
            #get the disease snps, but have to get rid of the possibilities in snps_in_filename
            # possible_snps_filtered = [k for k in possible_snps if k not in snps_in_filename]
            disease_snps = maf_dic['disease_snps']#np.random.choice(possible_snps_filtered, num_disease_snps).tolist()
            print(disease_snps)
            snps_in_filename = disease_snps + snps_in_filename
            run_script_args(maf_dic['pop'],maf_dic['corpus_id'], [maf_dic['sim_id']], "models/param_model_test_simple.xml", maf_dic['num_snps'], maf_dic['num_inds'],
                        snps_in_filename)
            filename = get_simulated_data_fname(maf_dic['sim_id'], maf_dic['corpus_id'], maf_dic['pop'], maf_dic['num_inds'], maf_dic['num_snps'],disease_snps)
            print(filename)
            with open(filename) as f:
                epigen_json = json.load(f)
            maf_dic_test = MAF_RANGE_DICT_TEST[key]
            corpora_snps_mapping_other_set = get_corpora_index_from_snps_id(maf_dic_test['pop'],maf_dic_test['corpus_id'])
            snps_in_filename = [] 
            for snps_id in epigen_json['disease_snps']:
                snp = epigen_json['snps'][snps_id]
                index = corpora_snps_mapping_other_set[snp[0]]
                snps_in_filename.append(index)
            print(len(epigen_json['snps']))
            print(len(epigen_json['disease_snps']))

            for i, snps_id in enumerate(epigen_json['snps']):
                if i in epigen_json['disease_snps']:
                    continue
                print(i)
                index = corpora_snps_mapping_other_set[snps_id[0]]
                snps_in_filename.append(index) #need to map back to CEU data todo
            print(len(snps_in_filename))
                
            run_script_args(maf_dic_test['pop'],maf_dic_test['corpus_id'], [maf_dic_test['sim_id']], "models/param_model_test_simple.xml", maf_dic_test['num_snps'], maf_dic_test['num_inds'],
                        snps_in_filename) 

if __name__ == '__main__':
    simulate_maf_data()

