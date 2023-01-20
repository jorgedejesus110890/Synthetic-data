# Synthetic-data

# 1. Introduction

The growth of smart devices, the Internet of Things, smart cities, and the myriad of sensor networks are causing the world to be inundated with a gigantic amount of data generated from numerous sources, such as social media, data from sensor networks, video streaming sites, bioinformatics, internet marketing and more. Extracting knowledge from such vast data sets is considered one of the biggest challenges for most traditional machine learning techniques [1].
A critical problem facing the vast majority of data today is class imbalance. A dataset is considered unbalanced if the number of instances in one class massively exceeds the instances in the other class. The main problem with the class imbalance problem is that the results will be skewed towards the majority class, which could give us inaccurate ranking results and lead us to make the wrong decisions. This problem occurs because most classifiers do not usually consider the data distribution when reducing global parameters, for example, the error rate. Therefore, a preprocessing phase is needed to solve the problem of unbalanced classes in such data sets before proceeding to the classification phase [2].

# 2. Theoretical Framework

## 2.1 Synthetic data

Synthetic data, as its name implies, is data that is artificially created, instead of being obtained through measurements or events, algorithms are generally used to increase the number of data, either because the tests require them, or there are not enough of them. instances to train an algorithm. Another name for the creation of synthetic data is “Data Augmentation”.
There are several reasons why synthetic data is important, some of which are:

- When data is required for testing, but is not currently available or does not exist.
- When you have a small number of instances and they are not enough to do both the training and the tests.
- When training data is required; however, it is expensive or unfeasible to carry them out.

### Advantages

- Flexibility to be able to test an algorithm and the data is not available or it takes a long time to be able to acquire or prepare enough data.
- freedom over the amount and distribution of data in a model.
- Protect the privacy of real data while working with models using synthetic data.

There are different types of synthetic data, which have different purposes. In general they can be divided as follows:

- Synthetic text.
- Multimedia synthetic data (videos, images, sounds, etc.).
- Synthetic tabular data.
 
Synthetic texts have been a complex issue until a few years ago. Creating text in a language that makes sense, with the complex grammar rules that exist in virtually any language is not a simple task. There are new machine learning models that can generate suitably good text using natural language generation algorithms.
Synthetic data can also be multimedia, usually created using similar characteristics to existing data either to add data to training or to replace sensitive or confidential multimedia data.
Tabular data resembles actual data that appears in the form of a table. These types of tables appear regularly in databases in the form of rows (instances) and columns (attributes). It is the most common type of synthetic data [3].

## 2.2 Unbalanced data

The data set is considered unbalanced if the number of samples in the majority class exceeds the samples in the minority class. For example, the patient population constitutes only a small part compared to normal healthy people. The most dangerous diseases have an even rarer number of cases, such as AIDS and cancer. Frankly speaking, it is very dangerous to identify one of the patients with infectious diseases as a healthy person and vice versa. That creates the unbalanced data set when we try to classify such data, which causes overfitting of the majority classes and could lead to skewed classification results and incorrect decisions. Traditional classifiers report very poor results when applied to unbalanced data sets. For example, the classifier might report very good performance in the majority class, but, on the other hand, might report very poor performance in the minority class, since they consider a balanced data distribution.

To solve the problem of unbalanced classes, many techniques have been proposed to solve this problem. Sampling is one of the techniques proposed for the preprocessing of the unbalanced classes problem [4].

## 2.3 Sampling

Sampling is mainly based on a simple idea which is to achieve balance between classes of data sets. Over-sampling and under-sampling are the main sampling methods. Sub-sampling is achieving data distribution equilibrium by removing some instances of the majority class. By contrast, oversampling tries to achieve equilibrium by doubling the instances of the minority class. Unfortunately, both algorithms suffer from severe drawbacks. Sub-sampling may neglect some important instances which may consequently affect the performance of the classification algorithms. Conversely, oversampling can create unnecessary minority class instances, which can consequently increase the execution time of the algorithm. In general, duplicating class instances can cause overfitting [5]. To solve the overfitting problem, Chawla, Bowyer, and Hall produced the SMOTE algorithm. The basic idea of the SMOTE algorithm is to generate synthetic instances of the minority class by using the attribute domain instead of the instance domain by creating synthetic instances of the minority class [6].

## 2.4 Roulette

Each of the data is assigned a part proportional to its roulette adjustment, in such a way that the sum of all the percentages is unity. The best data will receive a larger portion of the wheel than the worst. Generally the data is ordered based on fit so the largest portions are at the top of the wheel. To select a piece of data, it is enough to generate a random number from the interval [0..1] and return the data located in that position of the roulette wheel. This position is usually obtained by traversing the data and accumulating their spinner ratios until the sum exceeds the obtained value.
This method is similar to the stochastic roulette method that is widely used as a genetic selection operator in evolutionary computation.


## 2.5 SMOTE (Synthetic Minority Over-Sampling technique)


SMOTE is an algorithm for oversampling objects of the minority class. This algorithm generates synthetic objects from the k nearest neighbors of randomly chosen objects. SMOTE operates on the attribute space instead of the data space and was proposed with a k = 5 [6].

## 2.6 SMOTE algorithm

The SMOTE algorithm performs the following steps:

- Receives as a parameter the percentage of objects to be oversampled.
- Calculate the number of objects to generate.
- Find the k nearest neighbors of the objects of the minority class.

- Generates the synthetic objects as follows:
- For each object of the minority class, randomly choose the neighbor to use to create the new object.
- Calculates the difference between each of the attributes of the minority class object and the chosen neighbor.
- Multiply the difference obtained by a random number between 0 and 1.
- Add the value obtained in the multiplication to the value of the object of the minority class.
- Returns the set of synthetic examples.

# 3. Database

The unbalanced data set Abalone19 was chosen.
Attribute information:

| Attribute|Domain| Description|
|------|----------|------------|
|Sex| {M, F, I} |Sex|
|    ||              M (male)|
|    ||              F (female)|
|    ||              I ((infant)|
|Length| [0.075, 0.815] |Measured length of the longest shell (mm).|
|Diameter| [0.055, 0.65] |Diameter perpendicular to length (mm).|
|Height| [0.0, 1.13] |Height with meat in the shell (mm).|
|Whole_weight| [0.0020, 2.8255]| Whole weight in grams per abalone.|
|Shucked_weight| [0.0010, 1.488]| Weight of the abalone meat in grams.|
|Viscera_weight| [5.0E-4, 0.76]| Weight of the viscera in grams.|
|Shell_weight| [0.0015, 1.005]| Shell weight in grams.|
|Class| {positive, negative}| The positives belong to class 19 and the negatives belong to the rest.|

The Abalone19 dataset has 8 attributes and is composed of 4174 instances with the unbalanced ratio of 129.44. The number of positive instances is 32 (0.77%) and the number of negative instances is 4142 (99.23%).

# 4.	Experimentos y Resultados

The KNN classification algorithm was applied to the Abalone19 dataset to show the negative effect of having a class imbalance using the following metrics:

- Accuracy
- Accuracy
- Recall
- F1 Score
- Sensitivity
- Specificity

the confusion matrix was the following:

![alt text](https://github.com/jorgedejesus110890/Synthetic-data/blob/main/MC_Abalone.jpg?raw=true)

As can be seen, there are 829 true positives, 0 false positives, 9 false negatives and 0 true negatives.

The data obtained are described in the following table:

|Abalone19|	KNN|
|-|-|
|Accuracy|	0.9892|
|Precision|	1|
|Recall|	0.9892|
|F1 Score|	0.9946|
|Sensitivity|	0|
|Specificity|	0|

Table 1. Abalone19

In the table it can be seen that the Accuracy value is greater than 95% for the KNN classifier, this value is not reliable since there is a big difference between the number of instances in the two classes, the sensitivity and specificity values are 0 The main cause of this is the high rate of imbalance in this data set.

The Roulette model was applied to generate synthetic data on the abalone19 data set to increase the number of positive instances, the confusion matrix was as follows:

![alt text](https://github.com/jorgedejesus110890/Synthetic-data/blob/main/MCR_Abalone.jpg?raw=true)

As can be seen, there are 7541 true positives, 0 false positives, 19 false negatives and 36 true negatives.

The results obtained are shown in the following table.

|Abalone19(Roulette)|	KNN|
|-|-|
|Accuracy|	0.9975|
|Precision|	1|
|Recall|	0.9975|
|F1 Score|	0.9987|
|Sensitivity|	0.9975|
|Specificity|	1|

Table 2. Abalone19 Roulette method.

As shown in the Table doing a 10% data increase with the spinner method caused the Sensitivity, Specificity and F1_score values to improve for the KNN classifier, the Sensitivity value reached more than 99%, recall more than 99% and F1_score more than 99 in the KNN classifier. This reflects the effect of the instances added to the data set.
It can be seen in the following graphs the comparison of the distributions before and after applying the Roulette method, in blue they are shown before applying the Roulette method and in orange with the Roulette method.

![alt text](https://github.com/jorgedejesus110890/Synthetic-data/blob/main/Dist_Abalone.jpg?raw=true)

When applying the roulette method, the data preserved the distribution.

The SMOTE model was applied to generate synthetic data on the abalone19 dataset to increase the number of positive instances, the confusion matrix was as follows:

![alt text](https://github.com/jorgedejesus110890/Synthetic-data/blob/main/MCS_Abalone.jpg?raw=true)

As can be seen, there are 818 true positives, 11 false positives, 35 false negatives and 16 true negatives.
The results obtained are shown in the following table.

|Abalone19(SMOTE)|	KNN|
|-|-|
|Accuracy|	0.9477|
|Precision|	0.9867|
|Recall|	0.9590|
|F1 Score|	0.9727|
|Sensitivity|	0.9590|
|Specificity|	0.5926|

Table 3. Abalone19 SMOTE method.

As shown in the table, increasing the data with the Smote method caused the values to improve for the KNN classifier.
It can be seen in the following graphs the comparison of the distributions before and after applying the Smote method, in blue they are shown before applying the Roulette method and in orange with the Roulette method.

![alt text](https://github.com/jorgedejesus110890/Synthetic-data/blob/main/Dist_S_Abalone.jpg?raw=true)

When applying the Smote method, the data preserved the distribution.
We can see in tables 1, 2 and 3 that Roulette throws more positive instances than Smote.

# 5. Conclusions

In this paper the nature of unbalanced data is explored, since unbalanced data restricts the performance and accuracy of the classifier, two methods have been proposed, the roulette method and SMOTE to overcome the negative effects of data imbalance, it is presented a comparative study between the SMOTE and Roulette methods with the aim of addressing the classification of data using the KNN method. The Roulette method gave us a higher evaluation metric in the case of the proportion between the minority and the majority class, the values of precision, sensitivity and F1_score improved in the classifier, the values in the Abalone19 dataset improved with proportions greater than 98% in the KNN classifier due to the high unbalanced proportion in this data set.

# 6. Bibliografía 

[1] Amel Ali Alhussan, Doaa Sami Khafaga, El-Sayed M. El-Kenawy, Abdelhameed Ibrahim, Marwa Metwally Eid, Abdelaziz A. Abdelhamid, "Pothole and Plain Road Classification Using Adaptive Mutation Dipper Throated Optimization and Transfer Learning for Self Driving Cars", IEEE Access, vol.10, pp.84188-84211, 2022.

[2] S. García, Z.-L. Zhang, A. Altalhi, S. Alshomrani, and F. Herrera, ‘‘Dynamic ensemble selection for multi-class imbalanced datasets,’’ Inf. Sci., vols. 445–446, pp. 22–37, Jun. 2018.

[3] Aceves-Fernandez, Marco. (2021). Inteligencia Artificial para Programadores con Prisa. 

[4] S. Piri, D. Delen, and T. Liu, ‘‘A synthetic informative minority oversampling (SIMO) algorithm leveraging support vector machine to enhance learning from imbalanced datasets,’’ Decis. Support Syst., vol. 106, pp. 15–29, Feb. 2018.

[5] F. Hu, C. Yu, J. Dai, and K. Liu, ‘‘A mixed sampling method for imbalanced data based on neighborhood density,’’ in Proc. IEEE 4th Int. Conf. Cloud Comput. Big Data Anal. (ICCCBDA), Apr. 2019, pp. 94–98.

[6] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, ‘‘SMOTE: Synthetic minority over-sampling technique,’’ J. Artif. Intell. Res., vol. 16, no. 1, pp. 321–357, 2002.
