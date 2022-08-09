
------------------------------------>  Introduction to Machine Learning <----------------------------------------



Domain Knowledge Plays a Very Important Role in Machine Learning



---------------------> Supervised Learning Algorithms



Trained Labeled examples, i.e An Input where the desired OutPut is known

Eg. 

1. Spam vs Legitimate Emails       ------>  Someone went through the emails and we know the spams and the legitimate one
                                            so the program looks at the previous data and predicts the new one as spam/legitimate
                                            
2. Positive vs Negative Movie Reviews

The Algo get Inputs and Outputs, learns from it and then compare its outputs to the 'real outputs' to find error.
It then modifies the model.


Machine Learning Process :
    
                                            -----------> Test Data ------------>-----
                                           |                                         |
    Data Acquisition    ---> DataCleaning  ---> Model Training and Building ---> Model Testing ---> Model Deployment
                                                    |                                |
                                                    -<-- Adjust Model Parameters <---|                                            



1. We Acquire the data
2. Clean that Data and format it so that the machine learning model can accept it
3. Split the Data into Training Set, Validation Set, Test Set
4. Iterate and Tune the Parameters of the Model until its ready to deploy



To make the model more accurate we split the data into 3 sets

1. Training Data    ---> Used to Train Model Parameters.

2. Validation Data  ---> Used to Know which Parameters to Adjust.

3. Test Data        ---> Used to get final Parameter Metric, this is the data the model has never seen before.
                         This is how the model will work in the real world and we are not allowed to make anymore changes.
                         It will be the true performance of the model on UNSEEN DATA.



---------------------> Evaluating performance - Classification Error Metrics



Binary Classification - eg. Predict if the Image is a Dog/Cat

For Supervised Learning we FIT/TRAIN the model on TRAINING DATA and then TEST the model on TESTING Data. 
And then we compare the models prediction to the actual values

We organize the PREDICTED VALUES vs REAL VALUES in a CONFUSION MATRIX



------------> Classication Metric to Judge our Model (ARPF):
    
    
    
1. Accuracy :   <--------------------------------------------
    
    
    
    Accuracy ==         Total Number of Correct Predictions
                    --------------------------------------------
                            Total Number of Predictions
                            
                            

    Well Balanced Target Classes --->   Number of Images of Dog  ~= Number of Images of Cats
                                        51 Dog Images ~= 49 Cat images
                                        
    Un - Balanced Target Classes --->   Number of Images of Dog  >>> Number of Images of Cats
                                        99 Dog Images >>> 1 Cat images
                                        
    In the UnBalanced Target Class we will get 99% Accuracy which wont work in the actual world



2. Recall (Identification of 'Malignant Tumor') (Here we give only the 'Malignant Tumor' Images)  <-------------------------------------

    

    What Proportion of Actual Positive(Ground Truth) was identified correctly?
    
    Recall =                   True Positive                   =                     Correct Identification
                --------------------------------------------         -------------------------------------------------------
                       True Positive + False Negative                     Correct Identification + Wrong Identification
                       
                       
            Correct Identification = True Positive
            Wrong Identification   = False Negative


    Is it really able to recall what the 'Malignant Tumor' looks like?
    
    Eg. For a 'Malignant Tumor' identification if Recall = 0.11 then we can say that 
        the Model Correctly identifies 11% of all 'Malignant Tumor' 
        
        NOTE -  We give all the 'Malignant Tumor' images and then the model identifies if it is a 'Malignant Tumor'
                So, either it will say it is 'Malignant Tumor' and in reality it is 'Malignant Tumor', so it is TRUE POSITIVE
                or it will say it is not 'Malignant Tumor' and in reality it is 'Malignant Tumor', so it is FALSE NEGATIVE
                
                
                
3. Precision (Picking out 'Malignant Tumor' out of all the Tumors) (Here we give the Images of all Tumors)  <-------------------------------------------



    What proportion of positive identifications was actually correct?

    Precision =            True Positive                   =               Correct Answer
                ---------------------------------------         --------------------------------------
                    True Positive + False Positive                   Correct Answer + Wrong Answer
                       
                       
            Correct Answer = True Positive
            Wrong Answer   = False Positive


    Is it able to pick 'Malignant Tumor' out of all the 'Tumor' Images
    
    Eg. If the Precision for picking 'Malignant Tumor' = 0.4 then we can say that the model correctly picks 'Malignant Tumor' 40% of the Time

    NOTE -  We give the images of all 'Tumors' and the model picks 'Malignant Tumor' out of them so either it will be 'Malignant Tumor' or NOT
            So, either it will be say it is 'Malignant Tumor' and in reality it is 'Malignant Tumor', so it is TRUE POSITIVE
            or it will say it is 'Malignant Tumor' and in realtiy it is not 'Malignant Tumor', then it is FALSE POSITIVE



------------------------------------------------------------------------



NOTE -  There is a Tug of War between RECALL and PRECISION i.e
        If we increase Precision , then Recall Decreases
        & If we increase Recall, then Precision Decreases



------------------------------------------------------------------------



4. F1-Score   <------------------------------------------



    F1 Score is the Harmonic Mean of Recall and Precision


    
    Harmonic Mean =      2 * A * B                 
                    --------------------      
                           A + B



    F1 =      2 * Recall * Precision                
         --------------------------------       
                Recall + Precision


                
    We use Harmonic Mean cause it helps us deal with extreme values
    
    Eg. for Precision  = 1.0 and Recall =  0.0
    
    we have Harmonic Mean == 0 and Average  == 0.5


--------------> Confusion matrix

                            Prediction Positive         Prediction Negative

Condition Positive          True  Positive              False Negative

Condition Negative          False Positive              True  Negative



---------------------> Evaluating performance - Regression Error Metrics
 


Regression is a Task when a Model trys to predict continuous Values
(Classification is prediction of Categorical Values) 

Eg. Predict the price of a house given its features is a 'Regression Task'



----------------->  Evaluation Metric for Regression (MMR) :
    
    
    
1. Mean Absolute Error



Summation (i = 1, n)  |y(i) - y^(i)|
                    ------------------
                            n


y(i)            --->    Actual Value
y^(i) [y cap]   --->    Predicted Value by the model
n               --->    No. of Elements in the data


Mean Absolute Error does not punish large error

NOTE - Here Large Errors can create trouble so we have to use 'MEAN SQUARED ERROR'



2. Mean Squared Error



It is the 'Mean' of the 'Squared Error'



Summation (i = 1, n)  ( |y(i) - y^(i)| )**2
                    -----------------------
                               n


Here Large Error are more noted than MAE

NOTE : That Squaring of the Errors cause the squaring of the Actual Values as well

        Eg. Squaring of the Dollars in the House Price prediction which is difficult to interpret



3. Root Mean Square Error
    
    
        (    Summation (i = 1, n)  ( |y(i) - y^(i)| )**2      )
   SQRT (                        --------------------------   )
        (                                    n                )


y(i)            --->    Actual Value
y^(i) [y cap]   --->    Predicted Value by the model
n               --->    No. of Elements in the data



NOTE :  It Punishes the Large Error Values and has the same values as y

        To Get an Intuition of the Model Performance, Compare your 'ERROR METRIC' to the 'AVERAGE VALUE' in your data



---------------------> Machine Learning with Python



----------> Scikit learn



Every model in ScikitLEarn is used via an 'Estimator'

General Form of importing a Model :
    
    from sklearn.'family' import 'Model'
    
Eg. from sklearn.linear_model import LinearRegression

    Here, 'linear_model'     is a family of models  
          'LinearRegression' is an Estimator Object (i.e it has parameters which can be set when instantiated)
           Estimator Object  is the Model itself


Estimator Parameters : Params can be set when instantiated and have defult Values

After creating the model with our parameters, we then fit it on a training model

NOTE -  Remember to split the data in 'TRAINING SET' & 'TEST SET' 

#%%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X,y = np.arange(10).reshape(5,2), range(5)

print(X) 
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, shuffle=True)
print(y_test)
print(y_train)
print(X_test)
print(X_train)

#%%



NOTE - 'sklearn.cross_validation' has been deprecated and the new code is 'sklearn.model_selection'

1. We fit/train our model on a training data by 'model.fit(X_train, y_train   )' method
2. We predict values                         by 'model.predict(X_test)'          method
3. We then compare the values with the "TEST DATA", here the evaluation method depends on the ALGORITHM used
   eg. Regression, Classification, Clustering, etc



For All Estimators :
    model.fit()         : Fit training data
    model.fit(X, y)     : For Supervised Learning
    model.fit(X)        : For Un-Supervised Learning, since they are unlabeled data

    

For Supervised Estimators :
    model.predict(X_new)    : For a trained model, it predicts the label for each object in the array
    model.predict_proba()   : For Classification Problems some estimators provide this method.
                              It returns the Probability for each LABEL, label with highest probability is returned by model.predict()
    model.score()           : Return value b/w 0-1, For Classication & Regression, larger value mean greater fit              



For Un-Supervised Learning :
    model.predict()         : Predicts Label For Clustering Algorithms
    model.transform()       : Transform new data into new basis
    model.fit_tranform()    : Efficiently performs fit and transform on the same input data



------------------------------------>  Linear Regression <----------------------------------------
    




---------------------------------->  Cross-Validation & Bias-Variance Trade-Off <----------------------------------------



Used for understanding the models Performance

1. Its the point where we are adding just 'noise', by making the model more complex

2. The 'training error' goes down, But the 'test error' goes up

3. After the bias-variance trade-off the model begins to overfit


4 Types of Bias-Variance :
    
    
    
    Here, we will taking the example of a DartBoard to understand the concept
    
    
    
    1. Low-Bias     -   Low - Variance

        All The points are at the center and All the points are in the smallest circle of the dartboard 
    
    2. Low-Bias     -   High - Variance

        All The points are at the center and All the points are away from the smallest circle of the dartboard 
        
    3. High-Bias    -   Low - Variance  

        All The points are away from the center and All the points are close to each other
    
    4. High-Bias    -   High - Variance

        All The points are away from the center and All the points are far from each other


NOTE - For beginners the common mistake is adding complexity to the model i.e 'OVERFITTING THE MODEL' to the training SET.

This Causes the model to give out wrong predictions for new points


We want to balance out the bias & variance of our model to the point where the test data and the training data have reached some sort 
of minimum and grouping together



------------------------------------>  Logistic Regression <----------------------------------------



------------------------------------> K-Nearest Neighbors (Classification Algorithm) <----------------------------------------



Eg. We have a plot of DATA-POINTS with dogs and horses, with heights and weights

Looking at a new data points we have to predict whether it is a DOG/HORSE 



The Algorithm :


    
    1. Store all the Data

    2. Calculate the 'Distance' between 'x' and all points of the data  ---> Here, 'x' is a point from the Test Data
    
    3. Sort those Distances in an 'Ascending' Order
    
    4. Then Predict the 'Majority Label' of the 'k' closest points  ---> Here, 'k' is a Number



for k = 1 we pick up a lot of noise



---->   But as we increase the value of 'k' i.e k = 5, k = 10, k = 50 we create more bias in our model

        or a cleaner cut-off at the cost of mislabeling some points



Pro of K-Nearest Neighbors:


    1. Training is Trivial

    2. Works with any number of Classes

    3. Easy to add more data

    4. Has Few Parameters:  

        1.      k

        2.      Distance Metric         -----> Chp.4 of Intro to Stats Learning

                We are defining mathematically the distance between test points and the old training points
    
        
Con of K-Nearest Neighbors:  
    
    
    1. Doesnt work well with large data sets as it sorts the distance between the points which is quite an expensive process
    
    2. Not good with high dimensional data
    
    3. Categorical features dont work well




















