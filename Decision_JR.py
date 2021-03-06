import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import plot_confusion_matrix

data = pd.read_csv('student_major.csv')
data_clean = data.copy()


# Use labelendcode with : all Rank and GPA_old_group 

le = LabelEncoder()
data_clean['GPA_old_group'] = le.fit_transform(data['GPA_old_group']) 
data_clean['Rank_grade_major'] = le.fit_transform(data['Rank_grade_major']) 
data_clean['Rank_grade_business'] = le.fit_transform(data['Rank_grade_business']) 
data_clean['Rank_grade_computer'] = le.fit_transform(data['Rank_grade_computer']) 
data_clean['Rank_grade_finance'] = le.fit_transform(data['Rank_grade_finance']) 
data_clean['Rank_grade_total'] = le.fit_transform(data['Rank_grade_total']) 
data_clean['Rank_study_group'] = le.fit_transform(data_clean['Rank_study_group']) 


# Use one-hot with : Study, Rank_study_group

study_onehot = pd.get_dummies(data['Study'],dummy_na=True)
data_clean = pd.concat([data_clean,study_onehot],axis = 1) #แทรก onehoe columns ทางขวา
data_clean = data_clean.drop(['Study'],axis=1) #ลบ column study เดิมทิ้งได้


# Copy data
test = data_clean.copy()

# Change data in colume Age_Group
test = test.replace('<=20','0')
test = test.replace('21-25','1')
test = test.replace('26-30','2')

# Drop column Major, Student_ID becuses Major, Student_ID don't use to train.
test = test.drop(['Major'],axis=1)
test = test.drop(['Student_ID'],axis=1)

# Create feature 
feature_names = test.columns

# transfrom Panda to np.arry
ary1 = np.array(test)
am = data_clean['Major'].to_numpy()
amcls = am.copy()
amcls = list(dict.fromkeys(amcls)) #Class_names



# train-test split
X = ary1
Y = am
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)


#Decision-Tree - GridSearchCV : max_depth

m_param_grid = {'max_depth': np.arange(1, 20)}
tree_model=DecisionTreeClassifier()
clf = GridSearchCV(tree_model, m_param_grid, cv=3) #GridSearchCV
clf.fit(X_train,Y_train) #Train

m_pred = clf.predict(X_test) #Predict
acm = np.mean(m_pred == Y_test)

print("GridSearchCV Criterion Max_depth  =",acm*100,"%")

plot_confusion_matrix(clf, X_test, Y_test)
plt.show()


#Decision-Tree - GridSearchCV : Entropy

param_grid = { 'criterion':['entropy'],'max_depth': np.arange(1, 20)}
tree_model=DecisionTreeClassifier()
tree_gscv = GridSearchCV(tree_model, param_grid, cv=3) #GridSearchCV
e_model = tree_gscv.fit(X_train,Y_train) #Train

y_pred = e_model.predict(X_test) #Predict
acc = np.mean(y_pred == Y_test)

print("GridSearchCV Criterion Entropy Accuracy  =",acc*100,"%")
plot_confusion_matrix(tree_gscv, X_test, Y_test)
plt.show()


#Decision-Tree - GridSearchCV : Gini

g_param_grid = { 'criterion':['gini'],'max_depth': np.arange(1, 20)}
g_tree_model=DecisionTreeClassifier() 
g_tree_gscv = GridSearchCV(g_tree_model, g_param_grid, cv=3,scoring='accuracy') #GridSearchCV
g_model = g_tree_gscv.fit(X_train,Y_train) #Train

g_pred = g_model.predict(X_test) #Predict
accg = np.mean(g_pred == Y_test)

print("GridSearchCV Criterion Gini Accuracy  =",accg*100,"%")
plot_confusion_matrix(g_tree_gscv, X_test, Y_test)
plt.show()



#Plot graph
print('')
print("Accuracy Gini VS Max_depth")
plt.plot(g_model.cv_results_['mean_test_score'],color = 'r',label = 'Gini')
plt.plot(clf.cv_results_['mean_test_score'],color = 'b',label = 'Max_depth')
plt.xlabel("Max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

print("Accuracy Entropy VS Max_depth")
plt.plot(tree_gscv.cv_results_['mean_test_score'],color = 'g',label = 'Entropy')
plt.plot(clf.cv_results_['mean_test_score'],color = 'b',label = 'Max_depth')
plt.xlabel("Max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#Create Dicision Tree
tree_dot = export_graphviz(    
    tree_gscv.best_estimator_,
    out_file =None,
    feature_names = feature_names,
    class_names = amcls,
    rounded=True,
    filled = True
    
)
print(tree_dot)


