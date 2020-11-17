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





# Setting Up the Data
When applying machine learning models to a dataset a critical starting factor is ensuring
the data is set up appropriately. These steps include making sure the data is randomized, i.e.
shuffled, as well as ensuring it is suitably set up for the technique that is to be applied. Utilizing
R as the system of record for building the machine learning models, the online application
dataset is shuffled in order to ensure randomization and reduce the risk of any filters that may
have been in place generating bias when splitting the data. R code is provided in Figure 2 for the
shuffling process.



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



At this stage the data is ready to be split into a training and testing dataset. This is done to
generate less bias when evaluating prediction accuracy. The training dataset is used to build the
model and the training set is used to evaluate the accuracy of the model itself. Using the R code
in Figure 4, the decision is made to use a popular split of 80% of the data falling into the training
category and 20% falling into the testing category.
