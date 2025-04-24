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
def classify_nb(training_filename, testing_filename):
    # Load the CSV file
    training_data = []
    training_class = []
    testing_data = []
    
    with open(training_filename, newline='') as training:
        reader = csv.reader(training)
        for row in reader:
            numeric_part = [float(x) for x in row[:-1]]
            training_data.append(numeric_part)
            training_class.append(row[-1])
    
    with open(testing_filename, newline='') as testing:
        reader = csv.reader(testing)
        for row in reader:
            numeric_part = [float(x) for x in row]
            testing_data.append(numeric_part)

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

    res = []
    cnt = 0
    for row in testing_data:
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
            res.append("yes")
        else:
            res.append("no")
    return res
        

    
    
    
