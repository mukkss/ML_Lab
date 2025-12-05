# FIND-S ALGORITHM
import csv


with open('enjoysport.csv', 'r') as file:
    data = [row for row in csv.reader(file)]
    print("The total number of training instances are:", len(data)-1, '\n', data[1:])


num_attribute = len(data[0]) - 1
hypothesis = ['0'] * num_attribute


for i in range(0, len(data)):
    if data[i][num_attribute] == 'yes':
        for j in range(0, num_attribute):
            if hypothesis[j] == '0' or hypothesis[j] == data[i][j]:
                hypothesis[j] = data[i][j]
            else:
                hypothesis[j] = '?'


    print("\nThe hypothesis for the training instance {} is : \n".format(i), hypothesis)


print("\nThe Maximally specific hypothesis for the training instances is ", hypothesis)
