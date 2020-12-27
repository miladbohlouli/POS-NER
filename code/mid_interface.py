import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

def load_dict(path):
    print("**************************************************************\n"
          "Attempting to load he last checkpoint of the model \n"
          "**************************************************************")
    print("**************************************************************\n"
          "Last checkpoint loaded \n"
          "**************************************************************")
    return pickle.load(open(path, 'rb'))


def cal_accuracy_conf(ground_truth, predicted, tags_vocab):
    len_sentences = len(predicted)
    num_tags = len(tags_vocab)
    conf_matrix = np.zeros((num_tags, num_tags))
    tags_dict = {k: v for v, k in enumerate(tags_vocab)}

    total_accuracy = 0
    for i, sent in enumerate(predicted):
        len_sent = len(sent)
        temp_acc = 0
        for j in range(len_sent):
            if sent[j] in ground_truth[i] :
                temp_acc += 1

        temp_acc /= len_sent
        total_accuracy += temp_acc

    temp_set = set()
    for i in range(len(predicted)):
        len_true = len(ground_truth[i])
        len_pred = len(predicted[i])
        for true_sent in range(len_true):
            for pred_sent in range(len_pred):
                if ground_truth[i][true_sent][1] in tags_vocab:
                    if ground_truth[i][true_sent] == predicted[i][pred_sent]:
                        conf_matrix[tags_dict[ground_truth[i][true_sent][1]], tags_dict[ground_truth[i][true_sent][1]]] += 1

                    elif ground_truth[i][true_sent][0] == predicted[i][pred_sent][0]:
                        conf_matrix[tags_dict[predicted[i][pred_sent][1]], tags_dict[ground_truth[i][true_sent][1]]] += 1
                        # print(type(predicted[i][pred_sent][1]))

                        if predicted[i][pred_sent][1] == "N" and ground_truth[i][true_sent][1] == "ADV":
                            temp_set.add(predicted[i][pred_sent][0])

    total_accuracy /= len_sentences
    print(temp_set)
    print("**************************************************************\n"
          "The calculated accuracy:%.2f%% \n"
          "**************************************************************" % (total_accuracy * 100))
    print("**************************************************************\n"
          "Confusion matrix:\n"
          "**************************************************************" )

    file = open("save_dir/normal_results.txt", 'w')
    file.write(str(conf_matrix / np.reshape(np.sum(conf_matrix, axis=1), (-1, 1))))
    file.write(str(tags_dict))
    file.close()

    return conf_matrix, total_accuracy


def cal_acc_confusion(ground_truth, predicted):
    ground_truth_temp = []
    predicted_temp = []
    for i in range(len(ground_truth)):
        if ground_truth[i] != 'O':
            ground_truth_temp.append(ground_truth[i])
            predicted_temp.append(predicted[i])

    accuracy = accuracy_score(ground_truth_temp, predicted_temp)
    precision = precision_score(ground_truth_temp, predicted_temp, average="macro")
    recall = recall_score(ground_truth_temp, predicted_temp, average="macro")
    conf_matrix = confusion_matrix(ground_truth_temp, predicted_temp)

    return accuracy, precision, recall, conf_matrix
