import csv
import numpy as np

def prob(x, mean, sd):
    if(sd != 0):
        return (1 / (np.sqrt(2 * np.pi) * sd)) * np.exp(-((x - mean) ** 2) / (2 * sd ** 2))
    else: 
        if(x == mean):
            return 1
        else:
            return 0
def classify_nb(training_data, training_class, testing_data, testing_class):
    attributeN = len(training_data[0])

    mean = []
    sd = []
    
    for i in range(attributeN):
        columnYes = []
        columnNo = []
        for j in range(len(training_data)):
            row = training_data[j]
            if(training_class[j] == "yes"):
                columnYes.append(row[i])
            else:
                columnNo.append(row[i])
        mean.append([np.mean(columnYes), np.mean(columnNo)])
        sd.append([np.std(columnYes), np.std(columnNo)])

    correct = 0
    total = len(testing_data)
    for j in range(len(testing_data)):
        row = testing_data[j]
        yesProb = 1
        noProb = 1
        for i in range(attributeN):
            yesProb = yesProb * prob(row[i], mean[i][0], sd[i][0])
            noProb = noProb * prob(row[i], mean[i][1], sd[i][1])
        yes = 0
        no = 0
        for i in range(len(training_class)):
            if(training_class[i] == "yes"):
                yes = yes + 1
            else:
                no = no + 1
        
        yesProb = yesProb * (yes/(yes + no))
        noProb = noProb * (no/(yes + no))

        if(yesProb > noProb):
            if(testing_class[j] == "yes"):
                correct += 1
        else:
            if(testing_class[j] == "no"):
                correct += 1
    return (correct/total)

def ten_fold_validation(filename):
    data_by_fold = []
    current_fold_data = []
    class_by_fold = []
    current_fold_class = []

    with open(filename) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("fold"):
                if current_fold_data:
                    data_by_fold.append(current_fold_data)
                    class_by_fold.append(current_fold_class)
                    current_fold_data = []
                    current_fold_class = []
            else:
                parts = line.split(",")
                feature = list(map(float, parts[:-1]))
                current_fold_data.append(feature)
                current_fold_class.append(parts[-1])

        # Add the last fold's data
        if current_fold_data:
            data_by_fold.append(current_fold_data)
            class_by_fold.append(current_fold_class)


        res = 0
        for i in range(len(data_by_fold)):
            training_data = []
            training_class = []
            testing_data = []
            testing_class = []
            for j in range(len(data_by_fold)):
                if(i == j):
                    testing_class = testing_class + class_by_fold[j]
                    testing_data = testing_data + data_by_fold[j]

                else:
                    training_class = training_class + class_by_fold[j]
                    training_data = training_data + data_by_fold[j]

            res += classify_nb(training_data, training_class, testing_data, testing_class)
    return res/10

if __name__ == '__main__':
    print(ten_fold_validation("pima-folds-numeric.csv"))
        
