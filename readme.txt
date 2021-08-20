#0.Preparation 
    Colors and Markers maximum number is 8 each, we need to manually add more if having more than 8 classes.
    The last column of input CSV file must be the class/cluster name attribute.
    The CSV file must has attribute names.
    Putting CSV file to input folder
    I was using Python==3.6

#1.How to run code
    $conda create -n py36 python=3.6
    $python main.py iris_train.csv 3 2

#2.Concept
    I came across one K-means paper and it mentioned it has 
    better performance than conventional K-means methods. K-means algorithm
    is very sensitive to the initial cluster center selections. This paper mentions the key step is
    to find the optimal initial centers. Traditional k-means chooses random data 
    as its initial cluster centers and these random selections may not guarantee 
    unique grouping of clusters. Below are the key steps this paper proposes to find the optimal initial centers.

        # Calculate distance of each data from origin(0,0)
        # Sorting the distances
        # Divide distances to K euqal sets.
        # For each set, finding the middle data to be the initial cluster center.

    At first I assgined train/test data ratio to 3:1 and tested with traditional K-means and received 78% and 74% in terms 
    of training and test data. Improving K-means based on the above solution, this time training data accuracy increased to 89% and the test data is 85%. 
    Walking through the dataset, I found different attribute varies and I inferred it made accuracy lower. 
    deciding to normalize each attribute so each attribute value is between 0 and 1. 
    Then,I ran the pipeline again and the training data accuracy arrived 93.8% and the test data was 93.9% with L2 normalization.
    91.7% and 88% in terms of training and test data with L1 normalization.

    Although we have a higher accuracy result, I think the best way to test a model is through real data.
    Thus I prepared another dataset called "Wine dataset". It has 178 instances, 3 class groups and 13 attributes.
    It is tested with my pipeline and the training and test data accuracy are 93.2% and 91.1% with N2 normalization.
    Section 6 has been listed detailed information regaring the iris dataset.

#3.Input data:
    #3-1.output folder: The training and testing result csv.
    #3-2.images folder: L1,L2 point images with data normalization, and L2 without data normalization.
    #3-3.input folder: the test csv data.

#4.Key components
    #4-1.Kmeans class: It is Mainly to working on K-means algorithm. 
    #4-2.Utils class: It is Mainly to dealing with input, output data.

#5.Enum 
    #5-1.enums.py: It is Mainly for enum and common variables

#6.Evaluation

                 precision    recall  f1-score   support
 Iris-virginica       1.00      0.82      0.90        33
Iris-versicolor       0.84      1.00      0.91        32
    Iris-setosa       1.00      1.00      1.00        32
       accuracy                           0.94        97
      macro avg       0.95      0.94      0.94        97
   weighted avg       0.95      0.94      0.94        97


    #6-1.accuracy(major)
        In terms of accuracy, it was 78% when using traditional K-means. The final improved version is 94%. 
        I infer it is due to data normalization and improved K-means algorithm.

    #6-2.precision(major)
        In terms of precision, It shows that Iris-versicolor has the lowest value. It means it is 
        hard to find its unique features.
        
    #6-3.Recall(less important)
        In terms of recall, Iris-virginica has the lowest one. 
        I assume precision has higher importance than recall in this assignment. As eventually what we want to 
        know is how accurate this classifier distinguishes these three groups. So I would mark recall as "less important" in this case.


#7.Improvement and future work:
    #7-1.[Finish]
        data normalization.
        The code has been tested with another dataset(wine dataset).
        
    #7-2.[Future work] 
        #Try different K-means algorithms and compare them.
        #Principal component analysis
        #Using SSE curve or silhouette coefficient to find the optimal numbers of cluster

#8. Reference:
    #8-1.code:
        https://dev.to/rishitdagli/build-k-means-from-scratch-in-python-2140
    #8-2.paper:
        https://core.ac.uk/download/pdf/231162293.pdf
    #8-3.wine dataset:
        https://archive.ics.uci.edu/ml/datasets/wine
