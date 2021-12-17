#wget https://raw.githubusercontent.com/fractor/nntailoring/master/capacityreq/capacityreq.py

import csv
import math
import sys

numcols = 0
energies = []
expcapreq = 0
rounding = -1
numrows = 0
numclass1 = 0

if (len(sys.argv) == 1) or (len(sys.argv) > 3):
    print("Usage:")
    print(sys.argv[0]+" <filename.csv> [<sigdigits>]")
    print("Estimates the maximum capacity needed for a neural network to train the data given in CSV file.")
    print("<filename.csv> -- comma separated value file with last row being the class")
    print("<sigdigits> -- optional: number of significant digits after decimal point (for noisy data), default all.")
    print()
    print("Note: This program only gives reliable results for balanced two-class problems.")
    print()
    sys.exit()

if (len(sys.argv) > 2):
    rounding = int(sys.argv[2])
    print("Significant digits: ", rounding)

class1 = ''

with open(sys.argv[1], 'r') as csvfile:
    has_header = csv.Sniffer().has_header(csvfile.read(1024))
    csvfile.seek(0)  # Rewind.
    csvreader = csv.reader(csvfile)
    if has_header:
        next(csvreader)  # Skip header row.

    for row in csvreader:
        numrows = numrows+1
        result = 0
        numcols = len(row[:-1])
        for elem in row[:-1]:
            result = result + float(elem)
        c = row[-1]
        if (class1 == ''):
            class1 = c
        if (c == class1):
            numclass1 = numclass1+1
        if (rounding != -1):
            result = int(result*math.pow(10, rounding))/math.pow(10, rounding)
        energies = energies+[(result, c)]
sortedenergies = sorted(energies, key=lambda x: x[0])
curclass = sortedenergies[0][1]
thresholds = 0
for item in sortedenergies:
    if (item[1] != curclass):
        thresholds = thresholds+1
        curclass = item[1]


thresholds = thresholds+1
#number of bits to memorize all biases & corresponding binary labels (upper limit)
MEC = math.ceil(math.log(thresholds)/math.log(2))

# The following assume two classes (binary classifier: Y ∈ {0,1})

#assuming each feature will approx. hold complexity of MEC
expcapreq = MEC*numcols

#maximum capacity approximation needed
maxcap = thresholds*numcols + thresholds +1

entropy = -((float(thresholds)/numrows)*math.log(float(thresholds)/numrows) +
            (float(numrows-thresholds)/numrows)*math.log(float(numrows-thresholds)/numrows))/math.log(2)

print("Input dimensionality: ", numcols, ". Number of rows:",
      numrows, ". Class balance:", float(numclass1)/numrows)
print("Eq. energy clusters: ", thresholds,
      "=> binary decisions/sample:", entropy)
print()
print("Number of thresholds: ", thresholds)
MEC = math.ceil(math.log(thresholds)/math.log(2))
print("Memory Equivalent Capacity, is log2 of thresholds", MEC, "bits")
print("Estimated capacity need: ", int(math.ceil(expcapreq)), "bits")
print()
print("Max capacity need: ", maxcap, "bits")
print("Really? Max cap after log2", int(math.ceil(math.log(maxcap)/math.log(2))))