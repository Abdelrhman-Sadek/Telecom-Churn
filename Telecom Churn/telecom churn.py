import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
plt.rcParams["figure.figsize"]=(15,5)
#reading data 
df=pd.read_csv('E:\\py\\Data\\Orange_Telecom_Churn_Data.csv')
def data_info ():
    print(df.head())
    print(df.describe())
    print(df.info())
    print(df.isnull().sum())
    print(df.columns)
    print(df['churned'])
    print(df['voice_mail_plan'])
    print(df['voice_mail_plan'].head())

print(data_info ())
#Visulizing data to understand it more 
plt.figure(figsize=[5,5])
plt.title('Customer Attrition')
df['churned'].value_counts().plot.pie(autopct='%1.2f%%')
plt.show()

plt.rcParams['figure.figsize']=(15,6)
sns.countplot(x="state", data=df)

sns.distplot(df.total_day_charge)
plt.show()
sns.distplot(df.total_eve_charge)
plt.show()
sns.distplot(df.total_night_charge)
plt.show()
sns.distplot(df.total_intl_charge)
plt.show()
sns.barplot(x="state",y="churned" ,data=df)

#changeing (true,falses),(yes,no) to 1,0
from sklearn.preprocessing import LabelBinarizer
binarize = LabelBinarizer()
df['churned']=binarize.fit_transform(df['churned'])
df['voice_mail_plan']=binarize.fit_transform(df['voice_mail_plan'])
df['intl_plan']=binarize.fit_transform(df['intl_plan'])
#print(df['intl_plan'].head())

X=df.iloc[:,:-1]
y=df.iloc[:,-1]
#print(y)
#ploting the corlation bettwen the data
corlation=X.corrwith(y==1)
corlation.plot(kind='bar',title='corrlation bettwen the features and the out')
no_corlation=corlation[corlation<.04].index
X.drop(no_corlation,axis=1,inplace=True)
X.drop(['state','phone_number'],axis=1,inplace=True)
col=X.columns
print(X.info())
#print(X.shape)
#print(X.columns)

#spliting the data 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,shuffle=True)
print(X_test.shape)
print(X_train.shape)


#useing knn
from sklearn.neighbors import KNeighborsClassifier
error1=[]

for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred2= knn.predict(X_test)
    error1.append(np.mean(y_pred2))
    
plt.figure('test K')
plt.plot(range(1,15),error1,label="n_neighbors")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()

knn_clasfiction=KNeighborsClassifier(n_neighbors=11)
knn_clasfiction.fit(X_train,y_train)
knn_pred=knn_clasfiction.predict(X_test)
print("knn cnf_metrix : \n")
cnf_metrix = confusion_matrix(y_test,knn_pred)
cmap = sns.cubehelix_palette(50, as_cmap='.2f')
sns.heatmap(cnf_metrix,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt=".2f")
plt.xlabel('Predicted')
plt.ylabel('Actual')

#useing Ctree
from sklearn.tree import DecisionTreeClassifier
error1=[]
for k in range(1,15):
    tree1 = DecisionTreeClassifier(criterion="entropy",max_depth=k)
    tree1.fit(X_train,y_train)
    y_pred1= tree1.predict(X_test)
    error1.append(np.mean(y_pred1))

plt.figure('test K')
plt.plot(range(1,15),error1,label="tree_max_depth")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()


tree = DecisionTreeClassifier(criterion="entropy",max_depth=7)
tree.fit(X_train,y_train)
tree_Pred = tree.predict(X_test)

print("tree cnf_metrix : \n")
cnf_metrix2 = confusion_matrix(y_test,tree_Pred)
cmap = sns.cubehelix_palette(50, as_cmap='.2f')
sns.heatmap(cnf_metrix2,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt=".2f")
plt.xlabel('Predicted')
plt.ylabel('Actual')

#useing RFC
from sklearn.ensemble import RandomForestClassifier
error1=[]

for k in range(1,15):
    forest = RandomForestClassifier(max_depth =k, min_samples_split=3, n_estimators = 100, random_state = 1)
    forest.fit(X_train,y_train)
    y_pred1= forest.predict(X_test)
    error1.append(np.mean(y_pred1))
plt.figure('test K')
plt.plot(range(1,15),error1,label="forest_max_depth")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()

forest = RandomForestClassifier(max_depth =10, min_samples_split=5, n_estimators = 100, random_state = 1)
forest.fit(X_train,y_train)
forest_Pred = forest.predict(X_test)

print("RandomForestClassifier cnf_metrix : \n")
cnf_metrix3 = confusion_matrix(y_test,forest_Pred)
cmap = sns.cubehelix_palette(50, as_cmap='.2f')
sns.heatmap(cnf_metrix3,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt=".2f")
plt.xlabel('Predicted')
plt.ylabel('Actual')


print("Accuracy for forest :",forest.score(X_test, y_test)*100,"% \n")
print("classfing using forest classification_report \n",classification_report(y_test, forest_Pred))
print("Accuracy for KNN :",knn_clasfiction.score(X_test, y_test)*100,"% \n")
print("classfing using KNN classification_report \n",classification_report(y_test, knn_pred))
print("Accuracy for tree :",tree.score(X_test, y_test)*100,"% \n")
print("classfing using tree classification_report \n",classification_report(y_test, tree_Pred))

print('########################################################################')
print('forest f1 score is :',accuracy_score(y_test, tree_Pred))

















