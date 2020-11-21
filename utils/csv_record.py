
import csv
import copy

train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader

triggertest_fileHeader = ["model", "trigger_name", "epoch", "average_loss", "accuracy", "correct_data",
                          "total_data"]
poisontriggertest_result = []  # triggertest_fileHeader

posion_test_result = []  # train_fileHeader
posion_posiontest_result = []  # train_fileHeader
scale_result=[]

def save_result_csv(epoch, is_posion,folder_path):
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()

    if is_posion:
        test_csvFile = open(f'{folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

        test_csvFile = open(f'{folder_path}/poisontriggertest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(triggertest_fileHeader)
        test_writer.writerows(poisontriggertest_result)
        test_csvFile.close()



