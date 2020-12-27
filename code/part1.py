from mid_interface import *
from POS_tagger import *
np.set_printoptions(suppress=True, precision=2)

DATA_DIR = "Dataset"
load = True

if load:
    saving_dict = load_dict("save_dir/checkpoint.pickle")

    part1 = saving_dict["model"]
    test_sent_tag = saving_dict["test_sent_tag"]
    test_texts = saving_dict["test_texts"]

    # results = saving_dict["results"]

    results = part1.POS_tag_file("Dataset/input.txt")
    result_file = open("save_dir/results-out.txt", "w")
    result_file.write(str(results[:100]))

    # cal_accuracy_conf(test_sent_tag, results, part1.tags_vocab)

else:
    part1 = Viterbi(DATA_DIR + "/POStrutf.txt")
    test_sent_tag, test_texts = Viterbi.text_tokenize(DATA_DIR + "/POSteutf.txt")
    results = part1.test(test_texts)

    saving_dict = {
        "model": part1,
        "test_sent_tag": test_sent_tag,
        "test_texts": test_texts,
        "results": results
    }

    pickle.dump(saving_dict, open("save_dir/checkpoint.pickle", 'wb'))

