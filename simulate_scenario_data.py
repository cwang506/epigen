import json
import os
import numpy as np
#MAF
from simulate_data import run_script_args
maf_num_inds = 10000
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
                  "sim_id": 0},
    "MAF_0_01_DIFF_LD": { "disease_snps": [1369845, 821991, 1356799, 1375434, 561370, 1049417, 1122798, 854187, 1012842, 953409],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_01_02_DIFF_LD": { "disease_snps": [1229681, 161677, 624523, 1050027, 467866, 728707, 462476, 1082441, 931427, 520533],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_02_03_DIFF_LD": { "disease_snps": [1369266, 370781, 377012, 964634, 415376, 736583, 292255, 561028, 949696, 232461],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_03_04_DIFF_LD": { "disease_snps": [645739, 1295698, 700734, 1276134, 275783, 419022, 168917, 95581, 152440, 935533],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_04_05_DIFF_LD": { "disease_snps": [929212, 723206, 33126, 985260, 167218, 456494, 1352517, 1314368, 512583, 1192105],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_05_06_DIFF_LD": { "disease_snps": [965361, 522230, 697864, 855235, 66819, 1173980, 1202403, 1335101, 896895, 838011],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},   
    "MAF_06_07_DIFF_LD": { "disease_snps": [1373769, 775958, 797117, 692500, 103419, 96587, 1080026, 287607, 1361242, 301167],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_07_08_DIFF_LD": { "disease_snps": [1254302, 640804, 150950, 255416, 1311832, 556160, 293898, 689363, 1144096, 100214],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0}, 
    "MAF_08_09_DIFF_LD": { "disease_snps": [1376596, 610197, 141928, 966161, 612162, 1345454, 348550, 948191, 706110, 594813],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_09_1_DIFF_LD": { "disease_snps": [377757, 1089050, 64245, 533142, 24313, 744568, 45785, 345736, 19139, 832659],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_0_01_SAME_LD": { "disease_snps": [5004, 5009, 5017, 5018, 5022, 5024, 5025, 5046, 5053, 5056],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_01_02_SAME_LD": { "disease_snps": [5012, 5031, 5037, 5055, 5057, 5065, 5066, 5069, 5087, 5088],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_02_03_SAME_LD": { "disease_snps": [4995, 5015, 5020, 5026, 5030, 5032, 5038, 5041, 5044, 5063],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_03_04_SAME_LD": { "disease_snps": [5019, 5021, 5023, 5027, 5028, 5029, 5048, 5049, 5050, 5051],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_04_05_SAME_LD": { "disease_snps": [4996, 4997, 5000, 5001, 5016, 5045, 5105, 5116, 5120, 5122],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_05_06_SAME_LD": { "disease_snps": [4999, 5002, 5003, 5008, 5033, 5035, 5036, 5073, 5083, 5094],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},   
    "MAF_06_07_SAME_LD": { "disease_snps": [5010, 5011, 5013, 5014, 5034, 5039, 5042, 5047, 5059, 5061],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_07_08_SAME_LD": { "disease_snps": [4998, 5005, 5006, 5040, 5071, 5072, 5100, 5117, 5145, 5159],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0}, 
    "MAF_08_09_SAME_LD": { "disease_snps": [5007, 5043, 5115, 5658, 6301, 6783, 6791, 7017, 7425, 7477],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "ASW",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_09_1_SAME_LD": { "disease_snps": [14243, 16390, 18397, 19139, 19217, 21385, 21510, 22601, 23599, 24313],
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
                  "sim_id": 0},
    "MAF_0_01_DIFF_LD": { "disease_snps": [1369845, 821991, 1356799, 1375434, 561370, 1049417, 1122798, 854187, 1012842, 953409],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_01_02_DIFF_LD": { "disease_snps": [1229681, 161677, 624523, 1050027, 467866, 728707, 462476, 1082441, 931427, 520533],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_02_03_DIFF_LD": { "disease_snps": [1369266, 370781, 377012, 964634, 415376, 736583, 292255, 561028, 949696, 232461],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_03_04_DIFF_LD": { "disease_snps": [645739, 1295698, 700734, 1276134, 275783, 419022, 168917, 95581, 152440, 935533],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_04_05_DIFF_LD": { "disease_snps": [929212, 723206, 33126, 985260, 167218, 456494, 1352517, 1314368, 512583, 1192105],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_05_06_DIFF_LD": { "disease_snps": [965361, 522230, 697864, 855235, 66819, 1173980, 1202403, 1335101, 896895, 838011],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},   
    "MAF_06_07_DIFF_LD": { "disease_snps": [1373769, 775958, 797117, 692500, 103419, 96587, 1080026, 287607, 1361242, 301167],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_07_08_DIFF_LD": { "disease_snps": [1254302, 640804, 150950, 255416, 1311832, 556160, 293898, 689363, 1144096, 100214],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0}, 
    "MAF_08_09_DIFF_LD": { "disease_snps": [1376596, 610197, 141928, 966161, 612162, 1345454, 348550, 948191, 706110, 594813],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_09_1_DIFF_LD": { "disease_snps": [377757, 1089050, 64245, 533142, 24313, 744568, 45785, 345736, 19139, 832659],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_0_01_SAME_LD": { "disease_snps": [5004, 5009, 5017, 5018, 5022, 5024, 5025, 5046, 5053, 5056],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_01_02_SAME_LD": { "disease_snps": [5012, 5031, 5037, 5055, 5057, 5065, 5066, 5069, 5087, 5088],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_02_03_SAME_LD": { "disease_snps": [4995, 5015, 5020, 5026, 5030, 5032, 5038, 5041, 5044, 5063],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_03_04_SAME_LD": { "disease_snps": [5019, 5021, 5023, 5027, 5028, 5029, 5048, 5049, 5050, 5051],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_04_05_SAME_LD": { "disease_snps": [4996, 4997, 5000, 5001, 5016, 5045, 5105, 5116, 5120, 5122],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_05_06_SAME_LD": { "disease_snps": [4999, 5002, 5003, 5008, 5033, 5035, 5036, 5073, 5083, 5094],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},   
    "MAF_06_07_SAME_LD": { "disease_snps": [5010, 5011, 5013, 5014, 5034, 5039, 5042, 5047, 5059, 5061],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},  
    "MAF_07_08_SAME_LD": { "disease_snps": [4998, 5005, 5006, 5040, 5071, 5072, 5100, 5117, 5145, 5159],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0}, 
    "MAF_08_09_SAME_LD": { "disease_snps": [5007, 5043, 5115, 5658, 6301, 6783, 6791, 7017, 7425, 7477],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0},
    "MAF_09_1_SAME_LD": { "disease_snps": [14243, 16390, 18397, 19139, 19217, 21385, 21510, 22601, 23599, 24313],
                  "num_inds":maf_num_inds,
                  "num_snps":maf_num_snps,
                  "pop": "CEU",
                  "corpus_id": 122,
                  "sim_id": 0}                          
}

on_server = True

def get_simulated_data_fname(sim_id, corpus_id, pop, num_inds, num_snps, disease_snps, categorical = False):
    #disease snps come from terminal 
    prefix = "sim/" if not on_server else "/nobackup1c/users/cwang506/epigen_data2/"
    return prefix+ str(sim_id) + "_" + str(corpus_id) + "_" + pop + "_" +\
            str(num_inds) + "_inds_" + str(num_snps) + "_snps_" + "_".join([str(i) for i in disease_snps])\
            + "_disease_snps"+ ("_categorical" if categorical else"") + ".json"

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

def simulate_maf_data_normal():
    num_disease_snps = 10
    first_key = None
    first_disease_snps = None
    categorical = True
    #get the set of correlated SNPs, all from the same choromosome
    set_of_all_disease_snps = []
    for i in MAF_RANGE_DICT_TRAIN:
        set_of_all_disease_snps += MAF_RANGE_DICT_TRAIN[i]
    
    other_snps_list = [i for i in range(maf_num_snps - num_disease_snps + len(set_of_all_disease_snps))] #gets the snps that are in close proximity to each other
    for i in set_of_all_disease_snps:
        if i in other_snps_list:
            other_snps_list.remove(i)
    
    # other_snps_list = other_snps_list[:maf_num_snps - num_disease_snps] #final list of other SNPs

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
            disease_snps = MAF_RANGE_DICT_TRAIN[key]['disease_snps'] #np.random.choice(possible_snps, num_disease_snps).tolist()
            first_disease_snps = disease_snps.copy()
            print(disease_snps)
            continue
            run_script_args(maf_dic['pop'],maf_dic['corpus_id'], [maf_dic['sim_id']], "models/param_model_train_simple.xml", maf_dic['num_snps'], maf_dic['num_inds'],
                        disease_snps)
                        
            print("generating test data")
            filename = get_simulated_data_fname(maf_dic['sim_id'], maf_dic['corpus_id'], maf_dic['pop'], maf_dic['num_inds'], maf_dic['num_snps'], disease_snps, categorical)
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
            first_maf_dic = MAF_RANGE_DICT_TRAIN[first_key]

            filename = get_simulated_data_fname(first_maf_dic['sim_id'], first_maf_dic['corpus_id'], 
                        first_maf_dic['pop'], first_maf_dic['num_inds'], first_maf_dic['num_snps'], first_disease_snps, categorical)
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
            if len(set(snps_in_filename)) != maf_num_snps:
                raise RuntimeError("disease snp happens to be in set of other snps")
            run_script_args(maf_dic['pop'],maf_dic['corpus_id'], [maf_dic['sim_id']], "models/param_model_test_simple.xml", maf_dic['num_snps'], maf_dic['num_inds'],
                        snps_in_filename)
            filename = get_simulated_data_fname(maf_dic['sim_id'], maf_dic['corpus_id'], maf_dic['pop'], maf_dic['num_inds'], maf_dic['num_snps'],disease_snps, categorical)
            print(filename)
            with open(filename) as f:
                epigen_json = json.load(f)
            maf_dic_test = MAF_RANGE_DICT_TEST[key]
            corpora_snps_mapping_other_set = get_corpora_index_from_snps_id(maf_dic_test['pop'],maf_dic_test['corpus_id'])
            snps_in_filename1 = snps_in_filename
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
            if len(set(snps_in_filename)) != maf_num_snps:
                raise RuntimeError("disease snp happens to be in set of other snps")
            run_script_args(maf_dic_test['pop'],maf_dic_test['corpus_id'], [maf_dic_test['sim_id']], "models/param_model_test_simple.xml", maf_dic_test['num_snps'], maf_dic_test['num_inds'],
                        snps_in_filename) 


def simulate_maf_data_different_ld_block():
    num_disease_snps = 10
    num_other_snps = 9990
    num_inds = 10000
    categorical = True
    train_pop = 'ASW'
    train_corpus_id = 122
    test_pop = 'CEU'
    test_corpus_id = 122
    for key in MAF_RANGE_DICT_TRAIN:
        if "DIFF_LD" not in key:
            continue
        print(key)
        maf_dic = MAF_RANGE_DICT_TRAIN[key]
        snp_to_maf_training = get_maf_dict(train_pop, train_corpus_id)
        # print(maf_min, maf_max)
        # possible_snps = [snp_id for snp_id in snp_to_maf_training.keys() \
        #             if snp_to_maf_training[snp_id]>=maf_min and snp_to_maf_training[snp_id]<=maf_max]
        # corpora_snps_mapping = get_corpora_index_from_snps_id(train_pop, train_corpus_id)
        # possible_snps = [(corpora_snps_mapping[snp_id]) for snp_id in possible_snps]
        disease_snps = maf_dic['disease_snps']
        other_snps = []
        for disease_snp in disease_snps: #range(num_disease_snps):
            # disease_snp = int(np.random.choice(possible_snps, 1)) 
            threshold = int(num_other_snps/2/num_disease_snps +0.5)
            # while disease_snp < threshold:
            #     disease_snp = np.random.choice(possible_snps, 1)
            # #pick other snps that are close to this snp
            # disease_snps.append(disease_snp)
            other_snps += list(range(disease_snp - threshold, disease_snp + threshold +1))
            other_snps.remove(disease_snp)
        
        print(disease_snps)
        # disease_snps = disease_snps + other_snps
        run_script_args(train_pop,train_corpus_id, [0], "models/param_model_test_simple.xml", num_disease_snps+num_other_snps, num_inds,
                        (disease_snps+other_snps)[:num_disease_snps+num_other_snps])
                    
        print("generating test data")
        filename = get_simulated_data_fname(0, train_corpus_id, train_pop, num_inds, num_disease_snps + num_other_snps, disease_snps, categorical)
        print(filename)
        with open(filename) as f:
            epigen_json = json.load(f)

        corpora_snps_mapping_other_set = get_corpora_index_from_snps_id(test_pop,test_corpus_id)
        snps_in_filename = [] 

        #make the snps the same from train to test
        for snps_id in epigen_json['disease_snps']:
            snp = epigen_json['snps'][snps_id]
            index = corpora_snps_mapping_other_set[snp[0]]
            snps_in_filename.append(index)
        print(len(snps_in_filename))

        for i, snps_id in enumerate(epigen_json['snps']):
            if i in epigen_json['disease_snps']:
                continue
            index = corpora_snps_mapping_other_set[snps_id[0]]
            snps_in_filename.append(index) #need to map back to CEU data todo
        print(len(snps_in_filename))
        run_script_args(test_pop,test_corpus_id, [0], "models/param_model_test_simple.xml", len(snps_in_filename), num_inds,
                    snps_in_filename) 
        

def simulate_scenario_data_same_ld_block():
    num_disease_snps = 10
    num_other_snps = 9990
    num_inds = 10000
    categorical = True
    train_pop = 'ASW'
    train_corpus_id = 122
    test_pop = 'CEU'
    test_corpus_id = 122
    for maf_min in np.linspace(0, 0.9, 10):
        maf_max = maf_min + 0.1
        # if "DIFF_LD" not in key:
        #     continue
        # print(key)
        # maf_dic = MAF_RANGE_DICT_TRAIN[key]
        snp_to_maf_training = get_maf_dict(train_pop, train_corpus_id)
        print(maf_min, maf_max)
        possible_snps = [snp_id for snp_id in snp_to_maf_training.keys() \
                    if snp_to_maf_training[snp_id]>=maf_min and snp_to_maf_training[snp_id]<=maf_max]
        corpora_snps_mapping = get_corpora_index_from_snps_id(train_pop, train_corpus_id)
        possible_snps = [(corpora_snps_mapping[snp_id]) for snp_id in possible_snps]
        possible_snps.sort() #get the first 10
        threshold = int(num_other_snps/2)
        for index in range(len(possible_snps)):
            if possible_snps[index]<threshold:
                continue
            break
        disease_snps = possible_snps[index:index+10]
        other_snps = list(range(disease_snps[0]-threshold, disease_snps[0])) + list(range(disease_snps[-1], disease_snps[-1]+threshold))
        print(len(other_snps))
        print(disease_snps)
        # disease_snps = disease_snps + other_snps
        run_script_args(train_pop,train_corpus_id, [0], "models/param_model_test_simple.xml", num_disease_snps+num_other_snps, num_inds,
                        (disease_snps+other_snps)[:num_disease_snps+num_other_snps])
                    
        print("generating test data")
        filename = get_simulated_data_fname(0, train_corpus_id, train_pop, num_inds, num_disease_snps + num_other_snps, disease_snps, categorical)
        print(filename)
        with open(filename) as f:
            epigen_json = json.load(f)

        corpora_snps_mapping_other_set = get_corpora_index_from_snps_id(test_pop,test_corpus_id)
        snps_in_filename = [] 

        #make the snps the same from train to test
        for snps_id in epigen_json['disease_snps']:
            snp = epigen_json['snps'][snps_id]
            index = corpora_snps_mapping_other_set[snp[0]]
            snps_in_filename.append(index)
        print(len(snps_in_filename))

        # for i, snps_id in enumerate(epigen_json['snps']):
        #     if i in epigen_json['disease_snps']:
        #         continue
        #     index = corpora_snps_mapping_other_set[snps_id[0]]
        #     snps_in_filename.append(index) #need to map back to CEU data todo
        # print(len(snps_in_filename))
        run_script_args(test_pop,test_corpus_id, [0], "models/param_model_test_simple.xml", len(snps_in_filename), num_inds,
                    snps_in_filename) 


if __name__ == '__main__':
    simulate_scenario_data_same_ld_block()

