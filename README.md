# Classification-Major-with-Decision-Tree

This project is Decision-Tree classification major form dataset (student_major.csv)


## Model

### Decision tree

![page7image40536928](blob:https://stackedit.io/c01b8c5d-5dcd-43d2-804b-68c57f04e8c7)  

  ### Import library
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV 
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    from sklearn.metrics import plot_confusion_matrix
        
  ### Import dataset and prepare 
    data = pd.read_csv('student_major.csv')
    data_clean = data.copy()
    
### Use labelendcode with : all Rank and GPA_old_group 

    le = LabelEncoder()
    data_clean['GPA_old_group'] = le.fit_transform(data['GPA_old_group']) 
    data_clean['Rank_grade_major'] = le.fit_transform(data['Rank_grade_major']) 
    data_clean['Rank_grade_business'] = le.fit_transform(data['Rank_grade_business']) 
    data_clean['Rank_grade_computer'] = le.fit_transform(data['Rank_grade_computer']) 
    data_clean['Rank_grade_finance'] = le.fit_transform(data['Rank_grade_finance']) 
    data_clean['Rank_grade_total'] = le.fit_transform(data['Rank_grade_total']) 
    data_clean['Rank_study_group'] = le.fit_transform(data_clean['Rank_study_group']) 

### Use one-hot with : Study, Rank_study_group

    study_onehot = pd.get_dummies(data['Study'],dummy_na=True)
    data_clean = pd.concat([data_clean,study_onehot],axis = 1) #Insert onehoe columns to the right.
    data_clean = data_clean.drop(['Study'],axis=1) #drop the original columnc study

### Copy data
Data test is data for train

    test = data_clean.copy()
    
### Change data in colume Age_Group
Because using one-hot with colume Age_Group will cause some training issues. Therefore, it is recommended to adjust the value manually.

    test = test.replace('<=20','0')
    test = test.replace('21-25','1')
    test = test.replace('26-30','2')


### Drop column Major, Student_ID becuses Major, Student_ID don't use to train.

    test = test.drop(['Major'],axis=1)
    test = test.drop(['Student_ID'],axis=1)
    
### Create feature 

    feature_names = test.columns

### Transfrom Panda to np.arry

    ary1 = np.array(test)
    am = data_clean['Major'].to_numpy()
    amcls = am.copy()
    amcls = list(dict.fromkeys(amcls)) #Class_names

### Train-test split

    X = ary1
    Y = am
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)

### Decision-Tree - GridSearchCV : max_depth

    m_param_grid = {'max_depth': np.arange(1, 20)}
    tree_model=DecisionTreeClassifier()
    
    clf = GridSearchCV(tree_model, m_param_grid, cv=3) #GridSearchCV
    
    clf.fit(X_train,Y_train) #Train
    
    m_pred = clf.predict(X_test) #Predict
    
    acm = np.mean(m_pred == Y_test)
    print("GridSearchCV Criterion Max_depth  =",acm*100,"%")
    plot_confusion_matrix(clf, X_test, Y_test)
    plt.show()

### Decision-Tree - GridSearchCV : Entropy

    param_grid = { 'criterion':['entropy'],'max_depth': np.arange(1, 20)}
    tree_model=DecisionTreeClassifier()
    
    tree_gscv = GridSearchCV(tree_model, param_grid, cv=3) #GridSearchCV
    
    e_model = tree_gscv.fit(X_train,Y_train) #Train
    
    y_pred = e_model.predict(X_test) #Predict
    
    acc = np.mean(y_pred == Y_test)
    print("GridSearchCV Criterion Entropy Accuracy  =",acc*100,"%")
    plot_confusion_matrix(tree_gscv, X_test, Y_test)
    plt.show()


### Decision-Tree - GridSearchCV : Gini

    g_param_grid = { 'criterion':['gini'],'max_depth': np.arange(1, 20)}
    g_tree_model=DecisionTreeClassifier() 
    
    g_tree_gscv = GridSearchCV(g_tree_model, g_param_grid, cv=3,scoring='accuracy') #GridSearchCV
    
    g_model = g_tree_gscv.fit(X_train,Y_train) #Train
    
    g_pred = g_model.predict(X_test) #Predict
    
    accg = np.mean(g_pred == Y_test)
    print("GridSearchCV Criterion Gini Accuracy  =",accg*100,"%")
    plot_confusion_matrix(g_tree_gscv, X_test, Y_test)
    plt.show()

### Plot graph
**Accuracy Gini VS Max_depth**

    print('')
    print("Accuracy Gini VS Max_depth")
    plt.plot(g_model.cv_results_['mean_test_score'],color = 'r')
    plt.plot(clf.cv_results_['mean_test_score'],color = 'b')
    plt.xlabel("Max_depth")
    plt.ylabel("Accuracy")
    plt.show()
**Accuracy Entropy VS Max_depth**

    print("Accuracy Entropy VS Max_depth")
    plt.plot(tree_gscv.cv_results_['mean_test_score'],color = 'g')
    plt.plot(clf.cv_results_['mean_test_score'],color = 'b')
    plt.xlabel("Max_depth")
    plt.ylabel("Accuracy")
    plt.show()
    
### Create Decision Tree

    tree_dot = export_graphviz(    
        tree_gscv.best_estimator_, # You can change model(tree_gscv) to change decision tree. 
        out_file =None,
        feature_names = feature_names,
        class_names = amcls,
        rounded=True,
        filled = True
        
    )
    print(tree_dot)

> Use output tree_dot in [webgraphviz](http://www.webgraphviz.com) then you will Decision Tree

# Extra : Decision Tree + Ensemble Learning
### Decision Tree + Bagging, Random Forest

    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

### Bagging

    bagging= BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.4, max_features = 0.9)
    bagging.fit(X_train, Y_train)
    
    
    y_pred = bagging.predict(X_test)
    acc = np.mean(y_pred == Y_test)*100
    print("Bagging + Decision Tree Accuracy", acc)
    
### Random Forest

    Randtree_clf = RandomForestClassifier(n_estimators = 10, max_depth = None, min_samples_split = 2, random_state = 0)
    Randtree_clf.fit(X_train, Y_train)
    y_pred = Randtree_clf.predict(X_test)
    acc = np.mean(y_pred == Y_test)*100
    print("Randomforest Accuracy ->", acc)
# Output

![2](https://user-images.githubusercontent.com/68366806/131213763-ef9415f8-9f13-48a5-875b-64079a85dfea.png)

