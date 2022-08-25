## for data
from statistics import mean
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, svm
## for explainer
from lime import lime_tabular

############################### Data preprocessing ###############################

    ## 1. Import Data

# load the data from the CSV file
dtf = pd.read_csv("./data.csv")
print(dtf.head(), "\n")

# set the index and the classifier in the dataframe 
dtf = dtf.set_index("number")
# According to which features we are doing the classification, so rename in Y
dtf = dtf.rename(columns={"cohort":"Y"})


# display the percentage of A and B in the population to get an idea
ax = dtf["Y"].value_counts().sort_values().plot(kind="barh")
totals= []
for i in ax.patches:
    totals.append(i.get_width())
# all the tested person to calculate the percentage
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
ax.grid(axis="x")
plt.suptitle("Percentage of A and B in the group", fontsize=15)
plt.show()



    ## 2. Check categorical and missing data with a heat map

def utils_recognize_type(dtf, col, max_cat=20):
    '''
    Recognize whether a column is numerical or categorical.
    :parameter
        :param dtf: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of unique values to recognize a column as categorical
    :return
        "cat" if the column is categorical or "num" otherwise
    '''
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"

# plot and classify the data according to their type (numerical or categorical)
dic_cols = {col:utils_recognize_type(dtf, col) for col in dtf.columns}
heatmap = dtf.isnull()
for k,v in dic_cols.items():
 if v == "num":
   heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
 else:
   heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
# display the data set in a heatmap giving their type
fig, ax = plt.subplots()
a = sns.heatmap(heatmap, cbar=True, cbar_kws={'ticks': [0, 0.5, 1]}).set_title('Dataset Overview')
# Add colorbar, make sure to specify tick locations to match desired ticklabels
plt.show()

## Add the average of the columns for the missing data 
dtf["feature1"] = dtf["feature1"].fillna(dtf["feature1"].mean())
dtf["feature3"] = dtf["feature3"].fillna(dtf["feature3"].mean())

## Data type conversion and delete the older version in the dataframe (string type) and creation of new columns (coded 0 and 1)
dummy_Y = pd.get_dummies(dtf["Y"], prefix="Y", drop_first=True)
dtf= pd.concat([dtf, dummy_Y], axis=1)
dtf = dtf.drop("Y", axis=1)

dummy_cat = pd.get_dummies(dtf["categorical"], prefix="categorical", drop_first=True)
dtf= pd.concat([dtf, dummy_cat], axis=1)
dtf = dtf.drop("sex", axis=1)
dtf["feature2"] = dtf["feature2"].rank(method="dense", ascending=False).astype(int)

print(dtf.head(), "\n")



    ## 3. Shuffle data and prepare (delete the irrelevant data)

# delete some of the features which are just not relevant to the current study or containing missing values
dtf = dtf.drop("Unnamed", axis=1)
dtf = dtf.drop("time", axis=1)
# shuffle rows of the dataframe
dtf = dtf.sample(frac=1)

print(dtf.head(), "\n")



    ## 4. Visualize data 

# How feature2 are represented according to the feature1 of the persons. Plot a line graph
sns.displot(data=dtf, x="feature1", hue="feature2", kind="kde")
plt.show()

# Compare and display the feature3 according to the cohort
cohort, lenght = "Y_dum", "feature3"
fig1, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False)
fig1.suptitle("Cohort  vs  Lenght ", fontsize=15)
            
### Distribution
ax[0].title.set_text('density')
for i in dtf[cohort].unique():
    sns.kdeplot(dtf[dtf[cohort]==i][lenght], label=i, ax=ax[0])
ax[0].grid(True)
### Stacked
ax[1].title.set_text('bins')
breaks = np.quantile(dtf[lenght], q=np.linspace(0,1,11))
tmp = dtf.groupby([cohort, pd.cut(dtf[lenght], breaks, duplicates='drop')]).size().unstack().T
tmp = tmp[dtf[cohort].unique()]
tmp["tot"] = tmp.sum(axis=1)
for col in tmp.drop("tot", axis=1).columns:
     tmp[col] = tmp[col] / tmp["tot"]
tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
### Boxplot   
ax[2].title.set_text('outliers')
sns.boxplot(x=cohort, y=lenght, data=dtf, ax=ax[2])
ax[2].grid(True)
plt.show()


## Correlation matrix (features) 
corr_matrix = dtf.copy()
for col in corr_matrix.columns:
    if corr_matrix[col].dtype == "O":
         corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
corr_matrix = corr_matrix.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Pearson correlation")
plt.show()


# Give the correlation according to the p-value between 2 features
def testCorrelation(feature1, feature2) :
# Test the correlation between features
    model = smf.ols(feature1+' ~ '+feature2, data=dtf).fit()
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    p = round(p, 3)
    # Display the conclusion (correlated or not according to the p-value)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print(feature1, "and", feature2, "are", conclusion, "(p-value: "+str(p)+")\n")


# Check the correlation with the feature_
testCorrelation("feature_" , "Y_dum")


    ## delete some features in the dataframe which are not relevant (for different reasons)
# the feature5 introduce a biais (lots of the B in the group has the feature5 because of how they have been selected : the correlation isn't relevant)
# this feature doesn't seem relevant according to the study we are leading 
dtf = dtf.drop("feature5", axis=1)
# this feature have to much similarities to the result we are looking for
dtf = dtf.drop("Feature4", axis=1)
dtf = dtf.rename(columns={"Y_dum":"Y"})

print(dtf.head(), "\n")


## Rescale data between 0 and 1
scaler = preprocessing.RobustScaler()
X = scaler.fit_transform(dtf)
dtf= pd.DataFrame(X, columns=dtf.columns, index=dtf.index)
print(dtf.head(), "\n")



    ## 5. Split data into training and evaluation sets
dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.2)

## print info
print("X_train shape:", dtf_train.drop("Y",axis=1).shape, "| X_test shape:", dtf_test.drop("Y",axis=1).shape)
print("y_train mean:", round(np.mean(dtf_train["Y"]),2), "| y_test mean:", round(np.mean(dtf_test["Y"]),2))
print(dtf_train.shape[1], "features:", dtf_train.columns.to_list())



## Automatic selection of the features 

    ## Lasso 
X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns

## Anova
selector = feature_selection.SelectKBest(score_func = feature_selection.f_classif, k='all').fit(X,y)
anova_selected_features = feature_names[selector.get_support()]

## Lasso regularization
selector = feature_selection.SelectFromModel(estimator= linear_model.LogisticRegression(C=1, penalty="l1", solver='liblinear')).fit(X,y)
lasso_selected_features = feature_names[selector.get_support()]
 
## Plot
dtf_features = pd.DataFrame({"features":feature_names})
dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)
plt.show()



    ## Random forest to find the more important features
X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns.tolist()

## Importance
# check the importance of the features according to their entropy
model = ensemble.RandomForestClassifier(n_estimators=5000, criterion="entropy", random_state=0)
model.fit(X,y)
importances = model.feature_importances_

## Put in a pandas dtf
dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":feature_names}).sort_values("IMPORTANCE", ascending=False)
dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index("VARIABLE")
    
## Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')
dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel = "")
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()


## Selection of the features  

# collect the name of all the features we keep to classify the data
X_names = ["feature1", "feature2", "feature3"]

# split data into train and test
X_train = dtf_train[X_names].values
y_train = dtf_train["Y"].values
  
X_test = dtf_test[X_names].values
y_test = dtf_test["Y"].values



############################### Model selection ###############################


    ## 1. Search the best parameters with the model GradientBoosting Classifier

GBC = ensemble.GradientBoostingClassifier()

## Define hyperparameters combinations to try
param_dic_GBC = {'learning_rate':[0.05,0.01,0.005,0.001],      #weighting factor for the corrections by new trees when added to the model
'n_estimators':[100,250, 500],  #number of trees added to the model
'max_depth':[2,3,4,5,6,7],    #maximum depth of the tree
'min_samples_split':[2,4,6,8,10,20,40],    #sets the minimum number of samples to split
'min_samples_leaf':[1,3,5,7,9],     #the minimum number of samples to form a leaf
'max_features':[2,3,4, 5, 6, 7],     #square root of features is usually a good starting point
'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]       #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
}
## Random search of the best parameter
random_search_GBC = model_selection.RandomizedSearchCV(GBC, param_distributions=param_dic_GBC, n_iter=100, scoring="accuracy", n_jobs=-1).fit(X_train, y_train)

# display the best arguments to have the more accurate Gradient boosting classifier 
print("\nBest GBC Model parameters:", random_search_GBC.best_params_)
print("Best GBC Model mean accuracy:", random_search_GBC.best_score_)
GBC = random_search_GBC.best_estimator_


    ## 2. Try the model Logistic Regression Classifier

# This is the best configuration of the Logistic Regression classifier according to the set of data we have 
LRC = linear_model.LogisticRegression(solver='liblinear')


############################### Train  ###############################
# test the 2 classifiers (Gradient Boosting and Logistic Regression)
Classifiers = [GBC, LRC]

## train
list_predicted_prob = []
list_predicted = []
i=0
for model in Classifiers :
    model.fit(X_train, y_train)
    ## test
    list_predicted_prob.append(model.predict_proba(X_test)[:,1])
    list_predicted.append(model.predict(X_test))


############################### Evaluation ###############################

for i in range (len(list_predicted)):
    ## Accuray 
    accuracy = metrics.accuracy_score(y_test, list_predicted[i])
    auc = metrics.roc_auc_score(y_test, list_predicted_prob[i])
    print("\n--------------------Results",Classifiers[i],"--------------------\nAccuracy (overall correct predictions):", round(accuracy,2))
    print("Auc:", round(auc,2))
        
    ## Precision 
    recall = metrics.recall_score(y_test, list_predicted[i])
    precision = metrics.precision_score(y_test, list_predicted[i])
    print("Recall (all 1s predicted right):", round(recall,2))
    print("Precision (confidence when predicting a 1):", round(precision,2))
    print("Detail:")
    print(metrics.classification_report(y_test, list_predicted[i], target_names=[str(i) for i in np.unique(y_test)]))
    
# change label to A or B instead of 0, 1.
def changeLabels (list):
    classes = np.unique(list)
    new_label = []
    for elt in classes :
        if elt == 0 :
            new_label.append('A')
        else : 
            new_label.append('B')
    return new_label

## Confusion matrix
classes = changeLabels(y_test)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
for classifier, ax in zip(Classifiers, axes.flatten()):
    metrics.plot_confusion_matrix(classifier, X_test, y_test, ax=ax, cmap='Blues', display_labels=classes, colorbar = False)
    ax.title.set_text(type(classifier).__name__)
plt.tight_layout()  
plt.show()


    ## Cross-Validation 

## Cross validation with the k-fold method and ROC curves for each case of a fold
# list to store the accuracy of each fold
lst_accu_stratified = []
# list to contain the confusion matrix 
cms =[]

# Number of folds in the cross validation with the stratified K fold 
k=10
cv = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
tprs, aucs = [], []
mean_fpr = np.linspace(0,1,100)
fig = plt.figure()
i = 1

for train_index, test_index in cv.split(X_train, y_train):
    prediction = model.fit(X_train[train_index], y_train[train_index]).predict_proba(X_train[test_index])[:, 1]
    lst_accu_stratified.append(model.score(X_train[test_index], y_train[test_index]))    
    fpr, tpr, t = metrics.roc_curve(y_train[test_index], prediction)
    tprs.append(numpy.interp(mean_fpr, fpr, tpr))
    roc_auc = metrics.auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC fold {i} (AUC = {roc_auc:02f})')
    cm = metrics.confusion_matrix(y_train[test_index], model.predict(X_train[test_index]))
    cms.append(cm)
    i = i+1

print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:', max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:', min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:', mean(lst_accu_stratified)*100, '%')
   
plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
mean_tpr = np.mean(tprs, axis=0)
mean_auc = metrics.auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:02f} )', lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('K-Fold Validation')
plt.legend(loc="lower right")
plt.show()

# display all the confusion matrix of each fold of the cross validation
fig, ax = plt.subplots(2, 5)
for ind  in range(len(cms)):
    sns.heatmap(cms[ind],annot=True, fmt='d', ax=ax.flat[ind], cmap=plt.cm.Blues, cbar=False)
    ax.flat[ind].set_title(f"Fold number {ind+1}")
    ax.flat[ind].set_xlabel("Predicted")
    ax.flat[ind].set_ylabel("True")
plt.tight_layout()  
plt.show()



def explainResult (indice):
    print("True:", changeLabels(y_test[indice]) , "--> Pred:", changeLabels(list_predicted[0][indice]), "| Prob:", np.max(list_predicted_prob[0][indice]))
    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(y_train), mode="classification")
    explained = explainer.explain_instance(X_test[indice], model.predict_proba)
    explained.as_pyplot_figure() #show_in_notebook
    plt.show()



for i in range (0, len(y_test)):
    if y_test[i]==0 and list_predicted[0][i]==1 :
        explainResult (i)


# Reduction of the dimentionality to 2D 

## PCA
pca = decomposition.PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)
## train 2d model

model_2d = linear_model.LogisticRegression() 
model_2d.fit(X_train_2d, y_train)
    
## plot classification regions
from matplotlib.colors import ListedColormap
# define the 2 color for the colormap if they are A or B
colors = {np.unique(y_test)[0]:"green", np.unique(y_test)[1]:"red"}

X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=0.01), np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))

fig, ax = plt.subplots()
Y = model_2d.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
print(Y)
ax.contourf(X1, X2, Y, alpha=0.2, cmap=ListedColormap(list(colors.values())))
ax.set(xlim=[X1.min(),X1.max()], ylim=[X2.min(),X2.max()], title="Classification regions")
for i in np.unique(y_test):
    if i ==0 :
        title = "True A"
    else :
        title = "True B"
    ax.scatter(X_test[y_test==i, 0], X_test[y_test==i, 1], c=colors[i], label=title)  
plt.legend()
plt.show()

classes = changeLabels(y_test)
fig, ax = plt.subplots()
metrics.plot_confusion_matrix(model_2d , X_test_2d, y_test, ax=ax, cmap='Reds', display_labels=classes, colorbar = False)
plt.tight_layout()  
plt.show()

