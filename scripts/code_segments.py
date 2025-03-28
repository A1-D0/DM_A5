'''
Description: This code segment reads in a gzipped json file, splits the data into training and development sets, and extracts the dates from the data.
Editor: Osvaldo Hernandez-Segura
References: 
'''
import json
import gzip
import os

# code to open gzipped json file and read reviews into a list
input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
dataset = []
for l in input_file:
    d = eval(l)
    dataset.append(d)
input_file.close()

# code to split the data
train_data = dataset[:int(len(dataset)*0.8)]
dev_data = dataset[int(len(dataset)*0.8):]

# code to extract dates
dates = []
for i in range(len(dataset)):
    dates.append(int(dataset[i]['date'][:4]))


# additional code
print("Number of reviews in the training set:", len(train_data))
print(type(train_data))
print(train_data[:5])

print("Number of reviews in the development set:", len(dev_data))
print(type(dev_data))
print(dev_data[:5], end="\n\n")

print(type(dates))
print(dates[:5])
