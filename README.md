# Statistics-in-Loan-Qualification
Decision tree and Logistic regression


# Introduction

Machine learning as an element of statistics is a commonly practiced application in many
businesses. While the application can go well beyond business processes by delving into politics,
healthcare, general day-to-day applications and more, it has been a driver behind educational
business choices for quite some time. The type of application and the problem being explored
are widely varied, but an increasingly popular reason to explore machine learning is to automate
current business processes, thereby reducing resource needs.
A strong business case for the application of machine learning to automate a business process is
that of home loan approvals. By generating an online application process and setting limits that
determine eligibility, financial institutions can take an automated approach to loan qualification.
In order to take the human component out of the process however, financial institutions need
their machine learning model to make the appropriate predictions on when to approve or deny an
applicant and that relies heavily on the identification of the appropriate predictor variables.
In order to understand how an automated machine learning model could be built to address loan
qualification, a dataset from Dream Housing Finance will be utilized. Historical records from
online application forms will be applied in a binary classification to determine whether or not to
approve a loan application. The automation process relies on using machine learning to identify
the customer segments that are eligible so resources can be targeted to those customers.
Due to this being a classification problem both logistic regression and a decision tree will be
employed to address the identification of the customer segments. By applying the dependent
variable as the loan status, algorithms can be built to understand the influence the independent
variables have on this target. Accuracy tests will be utilized in each method to understand at the
end, which model provides a more precise prediction.

# Statistical Methods

While many machine learning methods exist in the current market, when understanding
binary classification problems both logistic regression and decision trees are popular technique
choices. When approaching the business problem of understanding the customer segments
eligible for home loans, both logistic regression and decision trees will provide the probability of
a customer qualifying for a loan so that resources can be dedicated more effectively by the
financial institution.
For the purposes of building out both models a dataset of 614 online application forms
from Dream Housing Finance will be utilized. The forms consist of the following 12 independent
variables: Loan ID, Gender, Marital Status, Number of Dependents, Education Level, Self-
Employment Status, Applicant Income, Co-applicant Income, Loan Amount, Loan Term, Credit
History, and Property Area. Each model will use the independent variables to understand which
act as predictors for the dependent variable, Loan Status.
The first model utilized will be logistic regression. Logistic Regression is used when the
dependent variable is categorical and not continuous (Pandey, 2018). In the case of a home loan
application, the dependent variable of Loan Status has two possible outcomes, “Approved” or
“Denied” making the logistic regression binomial. The intent behind building a logistic
regression model is to predict the loan approval process for Dream Housing Finance. By creating
a training and testing set from the data the accuracy of the prediction to the actual value in the
testing set will establish how effective the machine learning algorithm is.
Decision trees are an alternative machine learning method to logistic regression and are
often viewed as being easy to interpret. By implementing a decision tree model on the Dream
Housing Finance dataset, the ability to understand how the independent variables impact Loan Status will be readily consumable. Many of the same factors are taken into consideration as they
are for logistic regression, such as the binomial element of two possible outcomes and the intent
of predicting when a loan is approved or not by applying a training and testing dataset, however
the outcomes are different both in their portrayal and results.
Both decision tree and logistic regression will highlight the relationship between the
existing variables to the target and will thereby address what variables an automated model
should consider when determining customer eligibility and providing focus groups to loan
officers. Ultimately the models will be compared to one another to understand which provides a
higher predictive accuracy.

# Establishing a Baseline

Since the application of advanced statistical methods are utilized in order to improve
accuracy in models, it is wise to understand the starting point from the most basic level. Based
on the goal of understanding the target audience for loan officers, predicting qualified customer
segments is the focus. The dataset provides that out of 614 applications 422 were approved. This
results in a baseline accuracy of 68.7%, as showcased in Figure 1. When applying logistic
regression and a decision tree, the intent will be to improve the accuracy beyond the baseline
model.


<img width="659" alt="Screen Shot 2020-11-16 at 11 27 32 PM" src="https://user-images.githubusercontent.com/66921930/99346834-9022b880-2863-11eb-9b6e-860541943ed5.png">

# Setting Up the Data
When applying machine learning models to a dataset a critical starting factor is ensuring
the data is set up appropriately. These steps include making sure the data is randomized, i.e.
shuffled, as well as ensuring it is suitably set up for the technique that is to be applied. Utilizing
R as the system of record for building the machine learning models, the online application
dataset is shuffled in order to ensure randomization and reduce the risk of any filters that may
have been in place generating bias when splitting the data. R code is provided in Figure 2 for the
shuffling process.

<img width="663" alt="Screen Shot 2020-11-16 at 11 28 11 PM" src="https://user-images.githubusercontent.com/66921930/99346836-90bb4f00-2863-11eb-981a-4ee4468f3899.png">

With the data sufficiently shuffled, the next step involves dataset cleaning and set up to
accommodate logistic regression. This stage can be as simple as removing unnecessary variables
to changing variable format and handling missing values. In the case of Dream Housing Finance
an obvious variable for removal is the Loan ID. This is a unique element for each application and
will not have any impact on predicting loan eligibility. Additionally, based on dataset size and
categorical factor, the decision is made to omit missing values from the model. The last
adjustment made is implemented entirely due to the method applied being logistic regression.Since logistic regression is a classification algorithm and is being used to predict a binary
outcome, the Loan Status dependent variable is changed from “Y”, “N” to “1”, “0” where 1
represents the desired outcome (Brid, 2018). The R code and sample of cleaned data is
represented in Figure 3.

<img width="658" alt="Screen Shot 2020-11-16 at 11 28 48 PM" src="https://user-images.githubusercontent.com/66921930/99346837-9153e580-2863-11eb-9e9e-9d5b8361652d.png">

At this stage the data is ready to be split into a training and testing dataset. This is done to
generate less bias when evaluating prediction accuracy. The training dataset is used to build the
model and the training set is used to evaluate the accuracy of the model itself. Using the R code
in Figure 4, the decision is made to use a popular split of 80% of the data falling into the training
category and 20% falling into the testing category.

<img width="639" alt="Screen Shot 2020-11-16 at 11 29 04 PM" src="https://user-images.githubusercontent.com/66921930/99346838-91ec7c00-2863-11eb-9954-446c632f3b01.png">


When building machine learning models, while not required, validating the appropriate
actions took place is a rational step to ensure accurate results are achieved further along the
process. In this case where the data is split, ensuring the split occurred at the appropriate
percentages and that the training and testing datasets are split roughly equally between the two
possible outcomes, is an easy validation step to take. From the R code in Figure 5, the split
shows that it computed appropriately and that the variance between outcomes is mostly
equivalent between the two datasets.

<img width="656" alt="Screen Shot 2020-11-16 at 11 31 01 PM" src="https://user-images.githubusercontent.com/66921930/99347050-14753b80-2864-11eb-9c0a-577f58d30b5e.png">


# Logistic Regression Application
With the data sufficiently set up and split, logistic regression is ready to be applied to
build the first model for understanding eligible customer segments for home loans. Using R, a
generalized linear model is constructed around the training dataset, applying Loan Status as the
dependent variable and the now remaining 11 variables as independent or potential predictors.
The application of binomial to the family function provides instruction to build a logistic
regression model (Pandey, 2018). R code and results can be found in Figure 6.


<img width="681" alt="Screen Shot 2020-11-16 at 11 31 33 PM" src="https://user-images.githubusercontent.com/66921930/99347051-14753b80-2864-11eb-8238-02461a8f2b2c.png">

From the R output, quite a few coefficients are built, however only a couple have
significance to the model. Both Credit History and Semi Urban Property Areas represent
themselves as having an impact on predicting loan eligibility, although based on the probability
value of each, Credit History is the main contributor. An Education Level of Not Graduate is
nearly a significant variable but does, just barely, exceed the threshold of 0.05.
With the model built and an understanding of the model predictors, predictions can be
applied to the training dataset. It is important to make predictions on the training set in order to
begin to identify the optimum threshold value to apply to the testing model to increase prediction accuracy. Application of the tapply function to the training dataset outputs the average prediction
for each true outcome (Pandey, 2018). From the R code and output in Figure 7 the prediction for
average probability of true loan rejections is 44% while the prediction for average probability of
true loan approvals is 80%.

<img width="615" alt="Screen Shot 2020-11-16 at 11 32 12 PM" src="https://user-images.githubusercontent.com/66921930/99347052-14753b80-2864-11eb-81d3-0a8c4017ad7f.png">


A threshold value is intended to increase model prediction accuracy by reducing false
positives and false negatives. Confusion matrices help evaluate a model’s sensitivity and
specificity and is one way of understanding the accuracy a threshold has on a training dataset. A
common threshold of 0.5 applied, the R output for the confusion matrix in Figure 8 showcases
that the model predicted 70 type I errors and 10 type II errors with an overall accuracy of 81%.


<img width="676" alt="Screen Shot 2020-11-16 at 11 32 35 PM" src="https://user-images.githubusercontent.com/66921930/99347055-150dd200-2864-11eb-9006-eced12ba6feb.png">


In an effort to determine the optimum threshold value beyond running confusion
matrices, the process can be further explored by establishing a Receiver Operator Characteristic
curve. Also referred to as a ROC curve, the application uses R’s performance function to define
a plot for the true positive rate, or sensitivity of the model, to the false positive rate, or 1 minus
the specificity (Pandey, 2018). In essence, a ROC curve provides how the sensitivity and specificity outcomes vary from one threshold value to another, allowing an optimal threshold to
be visually identified. The R code for building a ROC curve can be found in Figure 9 with the
ROC curve itself, displayed in Figure 10.




Based on the ROC curve the optimal threshold value looks to be around 0.4 for
maximizing the true positive rate while minimizing the false positive rate. This value can be
applied to a confusion matrix for the testing dataset to understand how the logistic regression
model works to predict customers for loan eligibility. Running the model in Figure 11, there are
0 type II errors, or false negatives, and 19 type I errors, or false positives. Overall the accuracy of
the logistic regression model is 82.08%. Specifically, this means that the model can accurately
identify eligible loan applicants based on testing data 82.08% of the time. Overall it is a
relatively efficient model for loan officers to utilize to target eligible applicants.



# Decision Tree Application
Another popular technique for classification problems, are decision trees. Decision trees
are popular for their ability to provide highly interpretable results. By applying a decision tree
algorithm to Dream Housing Finance’s data it will be very clear what the best predictors for
eligible customer segments are. Due to the advance work that went into preparing the dataset for
use in logistic regression and the cross over between data set up between the two methods,
minimal work has to be applied to prepare the existing datasets for use in a decision tree model.
In fact, further cleanup is done only to make the decision tree easier to read. By relabeling the
Loan Status from “0” and “1” to “Denied” and “Approved” the output of the decision tree will be much clearer. Figure 12 showcases the R code used to convert both the testing and training
datasets nomenclature for Loan Status.



As the intent behind the model has not changed between logistic regression and the
decision tree, Loan Status remains the dependent variable. Similar to logistic regression, the
decision tree model is built on the training dataset and will be applied later to the testing dataset
to understand predictive accuracy. Using the function rpart, Loan Status is set as the dependent
variable and the remaining 11 variables are included so that the model can determine the
significant predictors from the entire dataset. Since the decision tree model will be utilized to
understand whether a loan application is approved or denied, the algorithm uses a method of
“class” to specifically build a classification tree. R code is found in Figure 13.



