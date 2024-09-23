# Note: To switch to store all instances, change storeAll to True in the crossValidation function. To switch back to storing only the k-nearest neighbors, change storeAll to False.
# Note: To use the iris dataset, run the 'k-NN Iris Dataset.py' file

import numpy as np
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Read the data
def readData(filePath):
    data = []
    with open(filePath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            name = parts[3]
            data.append((label, (x, y), name))
    return data

# kNN algorithm
def knn(trainData, testInstance, k):
    distances = [] # Store the distances between the test instance and each training instance
    for label, (x, y), _ in trainData:
        dist = np.sqrt((x - testInstance[0])**2 + (y - testInstance[1])**2) # Euclidean distance
        distances.append((dist, label))
    distances.sort(key=lambda x: x[0]) # Sort by distance in ascending order
    kNearestLabels = [label for _, label in distances[:k]]
    majorityLabel = Counter(kNearestLabels).most_common(1)[0][0]
    return majorityLabel

# Train the model with k-nearest neighbors
def trainModel(trainData, k, storeAll):
    if storeAll:
        return trainData
    else:
        storage = [] # Store the k-nearest neighbors of each class
        classCounts = Counter([label for label, _, _ in trainData])
        classStoredCounts = Counter()
        
        for label, instance, name in trainData:
            if classStoredCounts[label] < k:
                storage.append((label, instance, name)) # Store the first k instances of each class
                classStoredCounts[label] += 1
            else:
                predictedLabel = knn(storage, instance, k)
                if predictedLabel != label: # Replace the instance with the new one if the predicted label is different
                    storage.append((label, instance, name))

        return storage

# Evaluate the model
def evaluateModel(trainData, data, k):
    yTrue = [label for label, _, _ in data] # Labels in the test data
    yPred = [knn(trainData, instance, k) for _, instance, _ in data] # Predicted labels
    return accuracy_score(yTrue, yPred)

# N-fold cross-validation
def crossValidation(data, N, k, storeAll):
    kf = KFold(n_splits=N, shuffle=True, random_state=42)
    trainAccuracies = []
    testAccuracies = []
    
    for trainIndex, testIndex in kf.split(data): # each block is a tuple of train and test indices
        trainData = [data[i] for i in trainIndex] 
        testData = [data[i] for i in testIndex]
        
        model = trainModel(trainData, k, storeAll)
        
        trainAccuracy = evaluateModel(model, trainData, k)
        testAccuracy = evaluateModel(model, testData, k)
        
        trainAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)
    
    avgTrainAccuracy = np.mean(trainAccuracies)
    avgTestAccuracy = np.mean(testAccuracies)
    
    return avgTrainAccuracy, avgTestAccuracy

if __name__ == "__main__":
    filePath = "labeled-examples"
    data = readData(filePath) # formatted as (label, (x, y), name)
    N = 7 # Number of folds
    k = 5 # Number of nearest neighbors to consider
    storeAll = True # Change to True to store all instances, or False to store k-nearest neighbors
    
    avgTrainAccuracy, avgTestAccuracy = crossValidation(data, N, k, storeAll)
    
    print(f"Average Training Accuracy: {avgTrainAccuracy:.2f}")
    print(f"Average Testing Accuracy: {avgTestAccuracy:.2f}")