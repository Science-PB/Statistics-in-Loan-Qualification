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

<img width="685" alt="Screen Shot 2020-11-16 at 11 34 33 PM" src="https://user-images.githubusercontent.com/66921930/99347443-27d4d680-2865-11eb-858f-1c533f7dad75.png">


Based on the ROC curve the optimal threshold value looks to be around 0.4 for
maximizing the true positive rate while minimizing the false positive rate. This value can be
applied to a confusion matrix for the testing dataset to understand how the logistic regression
model works to predict customers for loan eligibility. Running the model in Figure 11, there are
0 type II errors, or false negatives, and 19 type I errors, or false positives. Overall the accuracy of
the logistic regression model is 82.08%. Specifically, this means that the model can accurately
identify eligible loan applicants based on testing data 82.08% of the time. Overall it is a
relatively efficient model for loan officers to utilize to target eligible applicants.

<img width="637" alt="Screen Shot 2020-11-16 at 11 38 43 PM" src="https://user-images.githubusercontent.com/66921930/99347444-29060380-2865-11eb-8b90-cd71f0819d75.png">

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

<img width="649" alt="Screen Shot 2020-11-16 at 11 39 49 PM" src="https://user-images.githubusercontent.com/66921930/99347446-29060380-2865-11eb-96f5-b2978ffbff17.png">

As the intent behind the model has not changed between logistic regression and the
decision tree, Loan Status remains the dependent variable. Similar to logistic regression, the
decision tree model is built on the training dataset and will be applied later to the testing dataset
to understand predictive accuracy. Using the function rpart, Loan Status is set as the dependent
variable and the remaining 11 variables are included so that the model can determine the
significant predictors from the entire dataset. Since the decision tree model will be utilized to
understand whether a loan application is approved or denied, the algorithm uses a method of
“class” to specifically build a classification tree. R code is found in Figure 13.

<img width="653" alt="Screen Shot 2020-11-16 at 11 40 17 PM" src="https://user-images.githubusercontent.com/66921930/99347447-299e9a00-2865-11eb-958d-df29f8b31e37.png">


The code in Figure 13 outputs the decision tree provided in Figure 14. The root node and
each subsequent node provides the probability of a loan being approved or denied as well as the
percentage of online applications utilized in each node. From the root node in Figure 14, it is
discerned that the loan approval rate of all applications is 69%. From the root node, decision
branches start breaking down the important predictor variables.

<img width="682" alt="Screen Shot 2020-11-16 at 11 42 07 PM" src="https://user-images.githubusercontent.com/66921930/99347683-a762a580-2865-11eb-9b90-3ab06059faeb.png">

As found in the logistic regression model, Credit History is seen as the most significant
variable in predicting loan eligibility. From the decision tree the first decision branch asks if the
loan applicant does not have a credit history. The left side of the decision tree is very concise, if
a loan applicant does not have a credit history there is only a 10% probability their loan will be
approved. The right side of the decision tree takes into account many more factors, however it
largely shows that applicants with a credit history have an 80% probability of being approved. 
Taking a step back from the decision tree, logically this makes sense as an established credit
history is an indicator highly correlated to understanding a person’s ability to pay an institution.
The decision tree takes into account more predictors if a loan applicant has a credit
history. These predictors include Property Area, the Applicant’s Income, the Loan Amount and
the Co-applicant’s Income. Expanding on Figure 14 results, if the decision tree is broken down
further it can be seen that an applicant with a credit history, who lives in either a rural or urban
area and has an Applicant Income greater than 1613 but less than 3362, has an 83% probability
of their loan being approved. Being able to quickly make these correlations, is what makes
decision trees such a popular choice when working with classification problems.
With the model built, it can now be used to predict the testing dataset. This is an
important step to understand how the predicted outputs compare to the actuals and therefore how
applicable the model is to the identified goal. The R code provided in Figure 15, runs the model
on the testing dataset and provides how well the model performed. When tested the model
accurately predicted 86 of the loan applications and misclassified 20. Turning these numbers into
an accuracy score, the model was found to be 81.13% accurate on the testing dataset, as seen in
Figure 15.

<img width="686" alt="Screen Shot 2020-11-16 at 11 42 40 PM" src="https://user-images.githubusercontent.com/66921930/99347685-a7fb3c00-2865-11eb-834b-138f9130b1d6.png">


Although the testing dataset has been applied to the model, to understand accuracy
further, steps can be taken to improve the score by tuning the model. Tuning parameters in a
decision tree is formally known as pruning. Pruning is used to reduce the likelihood of
overfitting by removing components that do not provide enough classification power (Hoare,
n.d.). There are several parameters available to be tuned, the maximum depth of any node, the
minimum number of observations in a node prior to splitting, the minimum number of
observations in the final leaf and the minimum increase of fit (Guru99, 2020).
By tuning the parameters, or pruning, the goal is to increase the accuracy of the model
while also decreasing the overall decision tree size if appropriate. The R code provided in Figure
16 builds the formulas for applying tuning and parameter fitting and demonstrates the parameters
that were found to increase the accuracy while reducing the overall tree size. From this, the
model is updated to build a pruned decision tree and provide the new accuracy score as shown in
Figure 17.

<img width="663" alt="Screen Shot 2020-11-16 at 11 43 31 PM" src="https://user-images.githubusercontent.com/66921930/99347686-a7fb3c00-2865-11eb-9d89-deadf9b70dc4.png">

<img width="674" alt="Screen Shot 2020-11-16 at 11 43 37 PM" src="https://user-images.githubusercontent.com/66921930/99347687-a893d280-2865-11eb-9f1b-44beae7ac939.png">

The result from the pruned decision tree is quite different, eliminating all but Credit
History as a predictor variable for loan eligibility. While marginal this smaller tree does increase
the accuracy of the model to 82.08%, the same score achieved from the logistic regression
model. Unlike the harder to decipher logistic regression model, the use of a decision tree
provides very clear direction to loan officers about the customer segments they should be
targeting. With the same score as logistic regression, the decision tree model is found to be as
viable, making it as much of an algorithm candidate for automating the home loan qualification
process as logistic regression is.

# Conclusion
Logistic regression and decision trees are both techniques utilized to help make
educational decisions on classification problems. In this case they were applied to understand
what customer segments lenders at Dream Housing Finance should target. This type of problem
exists throughout businesses across the world, how to efficiently and effectively allocate
resources. Addressing online applications one-by-one would be a time-consuming endeavor, and
in a world where time is money, its not a realistic solution. This is where statistics and more
specifically machine learning enter into the equation.
By applying different machine learning techniques, an automated process can now be
applied to online home loan applications with an 82% accuracy rating. While not a fool proof
solution, either logistic regression or a decision tree model provides sufficient algorithms to aide
loan officers in targeting the correct customers based on the identified significant predictor
variables which in both models highly reference Credit History.
A main difference between the models that needs to be understood is whether it is more
accurate for the effects of the predictors to be reviewed simultaneously or sequentially. Decision
trees look at sequential effects where logistic regression looks at the simultaneous effects of the
predictors (Bock, n.d.). Taking a step back and thinking about the loan approval process
logically, it is more akin to a check mark approach, i.e. an applicant being approved if they have
certain attributes, which would make decision trees a stronger selection choice in this particular
use case due to its sequential predictor interpretation. Since the goal is to identify customer
segments, decision trees perform this in an easier to understand way that leaves less room for
misinterpretation.To finalize, customer segments that display a credit history should be the target focus for
Dream Housing Finance. Both the tuned logistic regression and decision tree models output the
same accuracy rate, although decision tress are viewed as the safer selection due to their
comprehensibility and sequential effects on the predictors. The result to dedicate resources to
applicants with a credit history expands beyond statistics however, as it is a common lender
practice to require a credit history from applicants, whether the loan is a home loan or other
form. While the ethics factor could be in question, Dream Housing Finance could automate their
process to advance customers with a credit history to the top of the pile in order to more
effectively manage their resource pool.

# References

Pandey, P. (2018, Aug 18). A Guide to Machine Learning in R for Beginners: Logistic

Regression. Medium. Retrieved from: https://medium.com/analytics-vidhya/a-guide-to-machine-
learning-in-r-for-beginners-part-5-4c00f2366b90

Brid, R. (2018, Oct 17). Brief on Regression Analysis. Medium. Retrieved from:

https://medium.com/greyatom/logistic-regression-
89e496433063#:~:text=Their%20value%20strictly%20ranges%20from,%E2%80%9CYes%20or

%20No%E2%80%9D%20etc.
Guru99. (2020). Decision Tree in R with Example. Guru99. Retrieved from:
https://www.guru99.com/r-decision-trees.html#1

Hoare, J. (n.d.). Machine Learning: Pruning Decision Trees. DISPLAYR Blog. Retrieved from:
https://www.displayr.com/machine-learning-pruning-decision-trees/

Bock, T. (n.d.). Decision Tress Are Usually Better Than Logistic Regression. DISPLAYR Blog.

Retrieved from: https://www.displayr.com/decision-trees-are-usually-better-than-logistic-
regression/#:~:text=By%20contrast%2C%20logistic%20regression%20looks,decision%20trees

%20are%20much%20better.
Analytics Vidhya (n.d.). Loan Prediction Practice Problem (Using Python. Analytics Vidhya.

Retrieved from: https://courses.analyticsvidhya.com/courses/take/loan-prediction-practice-
problem-using-python/texts/6119689-model-building-part-i
