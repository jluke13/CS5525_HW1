import math
#Imports the values of the four features of the iris dataset into four lists, then calls the function to compute
#the chi-squared values for each pair of features.
def run() :
    import csv
    A = []
    B = []
    C = []
    D = []
    with open('iris.data.txt', newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=",")
        start = 0
        for row in file :
            if(len(row) > 0) :
                A.append(row[0])
                B.append(row[1])
                C.append(row[2])
                D.append(row[3])
        check(A,B)
        check(B,C)
        check(C,D)
        check(A,C)
        check(A,D)
        check(B,D)

#Computes the chi-squared value between the given two lists of features
def check(A,B) :
    chi = 0
    AB_dict = dict()
    for ai in range(0,len(A)) :
        if((A[ai]) in AB_dict) :
            AB_dict[A[ai]].append(B[ai])
        else :
            AB_dict[A[ai]] = [B[ai]]

    t = len(A)
    for k in iter(AB_dict) :
        alist = AB_dict[k]
        alist.sort()
        current = alist.pop(0)
        count = 1
        if(len(alist) == 0) :
            chi = chi + math.pow(count - (A.count(k)*B.count(current)/t),2)/(A.count(k)*B.count(current)/t)
        while(len(alist)>0) :
            if(alist[0] == current) :
                count = count + 1
            else :
                chi = chi + math.pow(count - (A.count(k)*B.count(current)/t),2)/(A.count(k)*B.count(current)/t)
                count = 1
            current = alist.pop(0)
            
    print(chi)
            
run()
