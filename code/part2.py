from NER import *
from mid_interface import *
import pickle
load = False

np.set_printoptions(suppress=True, precision = 2)

if load:
    pass


else:
    # jar = "ner_model/stanford-ner.jar"
    # model = 'ner_model/20qk_my-ner-model-eng.ser.gz'
    #
    # x_test, y_test = text_tokenize("Dataset/NERte.txt")
    # x_test, y_test = x_test[0:10], y_test[0:10]
    # num_sent = len(x_test)
    #
    # y_pred = []
    # classes = set()
    # for i in range(len(x_test)):
    #     if i % 100 == 0:
    #         print("Applying the NER (%d%%)" % (int(i / len(x_test) * 100)))
    #
    #     for j in range(len(x_test[i])):
    #         classes.add(y_test[i][j])
    #     result = NER_tagger(x_test[i], model, jar)
    #     y_pred.append([a[1] for a in result])
    #
    # whole_predictions = []
    # whole_y_test = []
    # for i in range(len(x_test)):
    #     whole_predictions.extend(y_pred[i])
    #     whole_y_test.extend(y_test[i])
    #
    # accuracy, precision, recall, conf_matrix = cal_acc_confusion(whole_y_test, whole_predictions)
    #
    # print("***************metrics***************\n"
    #       "Accuracy: %.2f\n"
    #       "precision: %.2f\n"
    #       "recall: %.2f\n"
    #       "*************************************\n" % (accuracy, precision, recall))
    #
    # print("********************conf matrix***********************\n")
    # print(conf_matrix)
    # print("******************************************************\n")

    jar = "ner_model/stanford-ner.jar"
    model = 'ner_model/custom-ner-model.ser.gz'

    x_test, y_test = text_tokenize1("Dataset/NERte.txt")
    num_sent = len(x_test)

    classes = np.unique(y_test)
    results = NER_tagger(x_test, model, jar)

    result_y = [a[1] for a in results]

    accuracy, precision, recall, conf_matrix = cal_acc_confusion(y_test, result_y)

    print("***************metrics***************\n"
          "Accuracy: %.2f\n"
          "precision: %.2f\n"
          "recall: %.2f\n"
          "*************************************\n" % (accuracy, precision, recall))

    conf_matrix = conf_matrix / np.reshape(np.sum(conf_matrix, axis=1), (-1, 1))

    print("********************conf matrix***********************\n")
    print(conf_matrix)
    print("******************************************************\n")

    file = open("save_dir/NER_results.txt", "w")
    file.write(str(conf_matrix) + "\n")
    classes_dict = {i: k for i, k in enumerate(classes)}
    file.write(str(classes_dict))
    file.close()
