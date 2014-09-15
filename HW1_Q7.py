import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
import numpy
import simplegoodturing
import math

punctuation = [".","!","?"]

#Reads in the files, tokenizes them, removes stopwords, and determines the n most frequent words in each text.
#Returns each text tokenized as well as the combined feature list, with no repetitions.
def preprocessing(n) :
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer("english")
    
    jane_file = open('janeausten.txt','r')
    jane_raw = jane_file.read()
    jane_file.close()
    jane_tokens = nltk.wordpunct_tokenize(jane_raw)
    jane_tokens = jane_tokens[2:]
    jane_all = nltk.Text(jane_tokens)
    jane = [w for w in jane_all if w not in stopwords]
    for i in range (0,len(jane)) :
        jane[i] = stemmer.stem(jane[i])
    jane_freq = nltk.FreqDist(w for w in jane if w not in punctuation)
    jane_features_freq = jane_freq.most_common(n)
    jane_features = []
    for f in jane_features_freq :
        jane_features.append(f[0])

    sherlock_file = open('sherlockholmes.txt','r')
    sherlock_raw = sherlock_file.read()
    sherlock_file.close()
    sherlock_tokens = nltk.wordpunct_tokenize(sherlock_raw)
    sherlock_tokens = sherlock_tokens[2:]  
    sherlock_all = nltk.Text(sherlock_tokens)
    sherlock = [w for w in sherlock_all if w not in stopwords]
    for i in range (0,len(sherlock)) :
        sherlock[i] = stemmer.stem(sherlock[i])
    sherlock_freq = nltk.FreqDist(w for w in sherlock if w not in punctuation)
    sherlock_features_freq = sherlock_freq.most_common(n)
    sherlock_features = []
    for f in sherlock_features_freq :
        sherlock_features.append(f[0])

    non_repeated_sherlock_features = [f for f in sherlock_features if f not in jane_features]
    features = []
    features.extend(jane_features)
    features.extend(non_repeated_sherlock_features)
    
    return (jane, sherlock, features)

#Create snippets of l consecutive sentences out of the given text, returning these in a list
def create_snippets(text, l) :
    snippets = []
    current_sentence = 0
    current_snippet = []
    for w in text :
        if w in punctuation :
            current_sentence = current_sentence + 1
            if current_sentence == l :
                snippets.append(current_snippet)
                current_snippet = []
                current_sentence = 0
        else :
            current_snippet.append(w)
    return snippets

#Takes in a list of predicted values from the classification and compares them to the actual labels
#Returns the values for the current fold as a list of size 2 in the appropriate index of the given fold_stats list
def find_error(predicted, actual, fold_stats) :
    recall = [0.0,0.0]
    precision = [0.0,0.0]
    fstat = [0.0,0.0]
    for i in range (0,2) :
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
        #print(i,j,tp,fp,fn,tn)
        if (tp > 0) :
            recall[i] = tp/(tp+fn)
            precision[i] = tp/(tp+fp)
        else :
            recall[i] = 0.0
            precision[i] = 0.0
        fstat[i] = stats.hmean([recall[i], precision[i]]) if recall[i] > 0 and precision[i] > 0 else 0.0
    recall_avg = numpy.mean(recall)
    precision_avg = numpy.mean(precision)
    fstat_avg = numpy.mean(fstat)
    fold_stats.append([recall_avg, precision_avg, fstat_avg])


#Classifies using the sklearn Naive Bayes classifier for either no smoothing or Laplace smoothing
def sk_learn_NB(a, jane_data, sherlock_data, k) :
    NB = MultinomialNB()
    NB.set_params(alpha=a)

    #Combine the data for each snippet into one data list and create a target list with corresponding class labels
    data = []
    target = []
    for d in jane_data :
        data.append(d)
        target.append(0)
    for d in sherlock_data :
        data.append(d)
        target.append(1)   
    data_array = numpy.array(data)
    target_array = numpy.array(target)
    
    #K-fold cross-validation
    kf = cross_validation.KFold(len(data), n_folds = k, shuffle = True)
    fold_stats = []
    for train_index, test_index in kf:
        data_train, data_test = data_array[train_index], data_array[test_index]
        target_train, target_test = target_array[train_index], target_array[test_index]            
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
    print("Average stats for", k, "-fold cross-validation with l=", l, "and e=", estimator)
    print("Recall: ", recall_overall_avg, "with variance:", var_recall)
    print("Precision: ", precision_overall_avg, "with variance:", var_precision)
    print("F-statistic: ", fstat_overall_avg, "with variance:", var_fstat)

#Classifies using the Good-Turing estimator
def good_turing(jane_data, sherlock_data, features, k) :
    #K-fold cross-validation
    kf_jane = cross_validation.KFold(len(jane_data), n_folds = k, shuffle = True)
    kf_sherlock = cross_validation.KFold(len(sherlock_data), n_folds = k, shuffle = True)

    jane_train_index = []
    jane_test_index = []
    for i, j in kf_jane :
        x = []
        y = []
        for a in i :
            x.append(a)
        for b in j :
            y.append(b)
        jane_train_index.append(x)
        jane_test_index.append(y)

    sherlock_train_index = []
    sherlock_test_index = []
    for i, j in kf_sherlock :
        x = []
        y = []
        for a in i :
            x.append(a)
        for b in j :
            y.append(b)
        sherlock_train_index.append(x)
        sherlock_test_index.append(y)
    
    fold_stats = []
    for i in range(0,k):
        jane_train = []
        sherlock_train = []
        test = []
        actual = []
        for index in jane_train_index[i] :
            for p in jane_data[index] :
                jane_train.append(p)
                print("1")
        for index in jane_test_index[i] :
            test.append(jane_data[index])
            actual.append(0)
            print("2")
        for index in sherlock_train_index[i] :
            for p in sherlock_data[index] :
                sherlock_train.append(p)
                print("3")
        for index in sherlock_test_index[i] :
            test.append(sherlock_data[index])
            actual.append(1)
            print("4")
        
        fd_jane = nltk.FreqDist(jane_train)
        fd_sherlock = nltk.FreqDist(sherlock_train)
        gt_jane = simplegoodturing.SimpleGoodTuringProbDist(fd_jane)
        gt_sherlock = simplegoodturing.SimpleGoodTuringProbDist(fd_sherlock)
        print("5")
        predicted = []
        jane_prob = math.log2(len(jane_train)/(len(jane_train)+len(sherlock_train)))
        sherlock_prob = math.log2(len(sherlock_train)/(len(jane_train)+len(sherlock_train)))
        for d in test :
            for f in features :
                if d.count(f) > 0 :
                    jane_prob = jane_prob + math.log2(gt_jane.prob(f))
                    sherlock_prob = sherlock_prob + math.log2(gt_sherlock.prob(f))
            if jane_prob > sherlock_prob :
                predicted.append(0)
            else :
                predicted.append(1)
            print("6")
        print("7")
        find_error(predicted, actual, fold_stats)
        
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
    print("Average stats for", k, "-fold cross-validation with l=", l, "and e=", estimator)
    print("Recall: ", recall_overall_avg, "with variance:", var_recall)
    print("Precision: ", precision_overall_avg, "with variance:", var_precision)
    print("F-statistic: ", fstat_overall_avg, "with variance:", var_fstat)
    
#Runs the classifier with k-fold cross-validation on l-sentence snippets, using the n most frequent words from
#each text as features (minus repeats) and using the given estimator(0 for maximum likelihood relative frequency
#estimator, 1 for Laplace estimator, and 2 for Good-Turing estimator).
def run(l, k, estimator, n) :
    preprocessing_return = preprocessing(n)
    jane = preprocessing_return[0]
    sherlock = preprocessing_return[1]
    features = preprocessing_return[2]
    
    jane_snippets = create_snippets(jane, l)
    sherlock_snippets = create_snippets(sherlock, l)

    #Count appearance of each feature for each snippet
    jane_data = []
    for s in jane_snippets :
        jane_feature_count = []
        for w in features :
            jane_feature_count.append(s.count(w))
        jane_data.append(jane_feature_count)

    sherlock_data = []
    for s in sherlock_snippets :
        sherlock_feature_count = []
        for w in features :
            sherlock_feature_count.append(s.count(w))
        sherlock_data.append(sherlock_feature_count)
    
    #Set estimator
    if(estimator == 0) :
        sk_learn_NB(0.00000000000001, jane_data, sherlock_data, k)
    elif(estimator == 1) :
        sk_learn_NB(1.0, jane_data, sherlock_data, k)
    else :
        good_turing(jane_snippets, sherlock_snippets, features, k)
        
run(1, 10, 2, 500)
'''
run(2, 10, 0, 500)
run(5, 10, 0, 500)
run(10, 10, 0, 500)
run(50, 10, 0, 500)
run(1, 10, 1, 500)
run(2, 10, 1, 500)
run(5, 10, 1, 500)
run(10, 10, 1, 500)
run(50, 10, 1, 500)
'''
