from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from scipy import stats
import numpy

#Takes in a number of folds k and performs k-fold cross-validation on a Gaussian Naive Bayes classifier on the Iris dataset.
#Prints out the overall average recall, precision, and f-statistic over all k folds.
def iris_k_fold(k) :
    iris = datasets.load_iris()
    NB = GaussianNB()

    #K-fold cross-validation
    kf = cross_validation.KFold(len(iris.data),n_folds = k, shuffle = True)
    fold_stats = []
    for train_index, test_index in kf:
            print(train_index, test_index)
            data_train, data_test = iris.data[train_index], iris.data[test_index]
            target_train, target_test = iris.target[train_index], iris.target[test_index]            
            NB.fit(data_train, target_train)
            find_error(NB.predict(data_test), target_test, fold_stats)
            
    #Determining average stats for k-fold cross-validation
    recall_overall_avg = 0.0
    precision_overall_avg = 0.0
    fstat_overall_avg = 0.0
    recall_list = []
    precision_list = []
    fstat_list = []
    for i in range(0,len(fold_stats)) :
        recall_overall_avg = recall_overall_avg + fold_stats[i][0]
        precision_overall_avg = precision_overall_avg + fold_stats[i][1]
        fstat_overall_avg = fstat_overall_avg + fold_stats[i][2]
        recall_list.append(fold_stats[i][0])
        precision_list.append(fold_stats[i][1])
        fstat_list.append(fold_stats[i][2])
    recall_overall_avg = recall_overall_avg/len(fold_stats)
    precision_overall_avg = precision_overall_avg/len(fold_stats)
    fstat_overall_avg = fstat_overall_avg/len(fold_stats)
    var_recall = numpy.var(recall_list)
    var_precision = numpy.var(precision_list)
    var_fstat = numpy.var(fstat_list)
    print("Average stats for", k, "-fold cross-validation:")
    print("Recall: ", recall_overall_avg, "with variance:", var_recall)
    print("Precision: ", precision_overall_avg, "with variance:", var_precision)
    print("F-statistic: ", fstat_overall_avg, "with variance:", var_fstat)        

#Takes in a list of predicted values from the classification and compares them to the actual labels
#Returns the values for the current fold as a list of size 3 in the appropriate index of the given fold_stats list
def find_error(predicted, actual, fold_stats) :
    recall = [0.0,0.0,0.0]
    precision = [0.0,0.0,0.0]
    fstat = [0.0,0.0,0.0]
    for i in range (0,3) :
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for j in range (0,len(actual)) :
            if (predicted[j] == i) :
                if (actual[j] == i) :
                    tp = tp + 1
                else :
                    fp = fp + 1
            else :
                if (actual[j] == i) :
                    fn = fn + 1
                else :
                    tn = tn + 1
        recall[i] = tp/(tp+fn)
        precision[i] = tp/(tp+fp)
        fstat[i] = stats.hmean([recall[i], precision[i]])
    recall_avg = (recall[0]+recall[1]+recall[2])/3
    precision_avg = (precision[0]+precision[1]+precision[2])/3
    fstat_avg = (fstat[0]+fstat[1]+fstat[2])/3
    fold_stats.append([recall_avg, precision_avg, fstat_avg])
    
iris_k_fold(10)
