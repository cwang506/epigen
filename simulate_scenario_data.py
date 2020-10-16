#MAF
from simulate_data import run_script_args
MAF_RANGE_DICT_TRAIN = {
    "MAF_09_1": {"disease_snps":[1162907, 808021, 59983, 89931, 22601, 377757, 305546, 524297, 1253274, 177],
                 "num_inds":10000,
                 "num_snps":1387466,
                 "pop": "ASW",
                 "corpus_id": 122,
                 "sim_id": 0},
    "MAF_08_09": {"disease_snps": [578139, 449581, 818454, 1084344, 611200, 148185, 81076, 746175, 1335937, 909773],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_07_08": {"disease_snps": [1308312, 362535, 1177479, 746994, 4618, 305675, 401990, 554524, 490023, 580056],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_06_07": {"disease_snps": [1259010, 928778, 952667, 1150292, 1254978, 116966, 738663, 408789, 1019740, 496455],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_05_06": {"disease_snps": [1123610, 353184, 867535, 1118003, 1218052, 338208, 67507, 715245, 891074, 441402],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},      
    "MAF_04_05": {"disease_snps": [1285619, 1343349, 124719, 892889, 259888, 1002197, 757111, 921863, 91881, 1108291],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_03_04": {"disease_snps": [669353, 1210402, 435035, 481911, 389848, 654246, 200661, 1328036, 919819, 441492],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_02_03": {"disease_snps": [522186, 1347005, 1024901, 160912, 1049729, 962595, 862339, 434846, 878125, 609244],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_01_02": {"disease_snps": [989080, 946293, 889675, 1165480, 1363809, 1266715, 256110, 409596, 1146700, 1376950],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_0_01": {"disease_snps": [621273, 19662, 1353288, 878803, 469038, 983270, 86493, 828637, 1230899, 78816],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0}                             
}

MAF_RANGE_DICT_TEST = {
    "MAF_09_1": {"disease_snps":[1162907, 808021, 59983, 89931, 22601, 377757, 305546, 524297, 1253274, 177],
                 "num_inds":10000,
                 "num_snps":1387466,
                 "pop": "CEU",
                 "corpus_id": 122,
                 "sim_id": 0},
    "MAF_08_09": {"disease_snps": [578139, 449581, 818454, 1084344, 611200, 148185, 81076, 746175, 1335937, 909773],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_07_08": {"disease_snps": [1308312, 362535, 1177479, 746994, 4618, 305675, 401990, 554524, 490023, 580056],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_06_07": {"disease_snps": [1259010, 928778, 952667, 1150292, 1254978, 116966, 738663, 408789, 1019740, 496455],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_05_06": {"disease_snps": [1123610, 353184, 867535, 1118003, 1218052, 338208, 67507, 715245, 891074, 441402],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},      
    "MAF_04_05": {"disease_snps": [1285619, 1343349, 124719, 892889, 259888, 1002197, 757111, 921863, 91881, 1108291],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_03_04": {"disease_snps": [669353, 1210402, 435035, 481911, 389848, 654246, 200661, 1328036, 919819, 441492],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_02_03": {"disease_snps": [522186, 1347005, 1024901, 160912, 1049729, 962595, 862339, 434846, 878125, 609244],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},    
    "MAF_01_02": {"disease_snps": [989080, 946293, 889675, 1165480, 1363809, 1266715, 256110, 409596, 1146700, 1376950],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_0_01": {"disease_snps": [621273, 19662, 1353288, 878803, 469038, 983270, 86493, 828637, 1230899, 78816],
                  "num_inds":10000,
                  "num_snps":1387466,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0}                             
}
def load_maf_data():
    for key in MAF_RANGE_DICT_TRAIN:
        maf_dic = MAF_RANGE_DICT_TRAIN[key]
        print(maf_dic)
        run_script_args(maf_dic['pop'],maf_dic['corpus_id'], [maf_dic['sim_id']], "models/param_model_train.xml", maf_dic['num_snps'], maf_dic['num_inds'],
                        maf_dic['disease_snps'])
        
        maf_dic_test = MAF_RANGE_DICT_TEST[key]
        run_script_args(maf_dic_test['pop'],maf_dic_test['corpus_id'], [maf_dic_test['sim_id']], "models/param_model_train.xml", maf_dic_test['num_snps'], maf_dic_test['num_inds'],
                        maf_dic_test['disease_snps']) #use train model, since we're using all snps


if __name__ == '__main__':
    load_maf_data()

