#!/usr/bin/env python
# coding: utf-8

# # Adult Census Income Prediction Project
# ![image.png](attachment:image.png)

# # *Problem Statement*
# • The Goal is to predict whether a person has an income of more than 50K a year or not.
# • This is basically a binary classification problem where a person is classified into the >50K group or <=50K group.

# # Import Libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv("adult.csv", na_values="?", skipinitialspace = True)
df.head(15)


# # Exploratory Data Analysis (EDA) 

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.describe().T


# In[8]:


df.isnull().sum()


# In[9]:


round(df.isnull().sum() / df.shape[0] * 100, 2).astype(str) + "%"


# In[10]:


nan_values_columns = ['workclass', 'occupation', 'country']


# In[11]:


for col in nan_values_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


# In[12]:


df.nunique()


# In[13]:


round(df.nunique() / df.shape[0] * 100, 2).astype(str) + "%"


# In[14]:


df.duplicated().sum()


# In[15]:


df = df.drop_duplicates()
df.head(5)


# In[16]:


df.shape


# In[17]:


salary = df['salary'].value_counts(normalize=True)
round(salary * 100, 2).astype('str') + ' %'


# # Data Visualization

# In[18]:


# barplot plot for 'Age'
salary= df['salary'].value_counts()

plt.figure(figsize=(7,5))
plt.tick_params(labelsize=12)
sns.barplot(salary.index, salary.values, palette='colorblind')
plt.title('Distrubtion of Salary', fontdict={'fontname':'Monospace', 'fontsize':18, 'fontweight':'bold'})
plt.xlabel('Salary', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.ylabel('Number of People', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.show()


# In[19]:


# distribution plot for 'Age'
age = df['age'].value_counts()

plt.figure(figsize=(20,15))
sns.displot(df['age'], bins=20, kde=True)
plt.title('Distrubtion of Age', fontdict={'fontname':'Monospace', 'fontsize':18, 'fontweight':'bold'})
plt.xlabel('Age', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.ylabel('Number of People', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.tick_params(labelsize=12)
plt.show()


# In[20]:


# Barplot for "Education"
education = df['education'].value_counts()

plt.figure(figsize=(7,5))
sns.barplot(education.values, education.index, palette='bright')
plt.title('Distrubtion of Education', fontdict={'fontname':'Monospace', 'fontsize':18, 'fontweight':'bold'})
plt.xlabel("Number of People", fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.ylabel("Education", fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.tick_params(labelsize=14)
plt.show()


# In[21]:


# Barplot for "Years of Education"

education_num = df['education-num'].value_counts()
plt.figure(figsize=(7,5))
sns.barplot(education_num.index, education_num.values, palette='colorblind')
plt.title('Distribution Years of Education', fontdict={'fontname': 'Monospace', 'fontsize': 18, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={'fontname': 'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.ylabel('Number of People', fontdict={'fontname': 'Monospace', 'fontsize': 15, 'fontweight': 'bold'})
plt.tick_params(labelsize=13)
plt.show()


# In[22]:


# Pie chart for "Marital Status"
marital = df['marital-status'].value_counts()

plt.figure(figsize=(12,9))
plt.pie(marital.values, labels=marital.index, explode=(0.10,0.10,0.10,0.10,0.10,0.10,0.10), autopct='%1.1f%%')
plt.title("Marital_Status Distrubtion",fontdict={'fontname': 'Monospace', 'fontsize': 18, 'fontweight': 'bold'})
plt.legend()
plt.legend(prop={'size':12})
plt.axis('equal')
plt.show()


# In[23]:


# Chart for 'Age'
relationship = df['relationship'].value_counts()

plt.figure(figsize=(9,12))
plt.pie(relationship.values, labels=relationship.index, autopct='%1.1f%%')
center_part_circle = plt.Circle((0,0), 0.7, fc='White')
plt.title('Relationship distribution', fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
fig = plt.gcf()
fig.gca().add_artist(center_part_circle)
plt.axis('equal')
plt.legend(prop={'size':8})
plt.show()


# In[24]:


# borplot for 'Sex'

sex = df['sex'].value_counts()
plt.figure(figsize=(7,5))
sns.barplot(sex.index, sex.values, palette='bright')
plt.title('Distrubtion of Sex', fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={'fontname': 'Monospace', 'fontsize': 10, 'fontweight': 'bold'})
plt.ylabel('Number of People',  fontdict={'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.tick_params(labelsize=10)
plt.show()


# In[25]:


# barplot for 'Race'
race = df['race'].value_counts()

plt.figure(figsize=(6,4))
sns.barplot(race.values, race.index, palette='colorblind')
plt.title('Race distribution', fontdict={'fontname': 'Monospace', 'fontsize': 10, 'fontweight': 'bold'})
plt.xlabel("Number of People", fontdict={'fontname':'Monospace', 'fontsize':10, 'fontweight':'bold'})
plt.ylabel("Race", fontdict={'fontname':'Monospace', 'fontsize':10, 'fontweight':'bold'})
plt.tick_params(labelsize=10)
plt.show()


# In[26]:


# barplot for 'Hours per week'
hour = df['hours-per-week'].value_counts().head(15)

plt.figure(figsize=(7,5))
sns.barplot(hour.index, hour.values, palette='colorblind')
plt.title('Distrubtion of Hours of work per week', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.xlabel('Hours per week', fontdict={'fontname':'Monospace', 'fontsize':10, 'fontweight':'bold'})
plt.ylabel('Number of people', fontdict={'fontname':'Monospace', 'fontsize':10, 'fontweight':'bold'})
plt.tick_params(labelsize=8)
plt.show()


# # Bivariate Analysis

# Bivariate analysis is a statistical technique used to analyze the relationship between two variables. In coding, bivariate analysis can be performed using a variety of programming languages and statistical software packages. 

# In[27]:


# Countplot of Salary with Age
plt.figure(figsize=(15,12))
sns.countplot(df['age'], hue=df['salary'])
plt.title('Distrubtion of Salary', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.xlabel('Age', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.ylabel('Number of people', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.legend(loc=1, prop={'size':10})
plt.tick_params(labelsize=10)
plt.show()


# In[28]:


# countplot of salary by education

plt.figure(figsize=(20,15))
sns.countplot(df['education'], hue=df['salary'], palette='dark')
plt.title('Distrubtion of Salary with Education', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.xlabel('Education', fontdict={'fontname':'Monospace', 'fontsize':10, 'fontweight':'bold'})
plt.ylabel('Number of people',fontdict={'fontname':'Monospace', 'fontsize':10, 'fontweight':'bold'})
plt.legend(loc=1, prop={'size':15})
plt.tick_params(labelsize=10)
plt.show()


# In[29]:


# Countplot of salary with years of Education-num

plt.figure(figsize=(16,12))
sns.countplot(df['education-num'], hue=df['salary'])
plt.title('Disturbtion of salary with Education-num', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.xlabel('Years of Education', fontdict={'fontname':'Monospace', 'fontsize':12, 'fontweight':'bold'})
plt.ylabel('Number of People', fontdict={'fontname':'Monospace', 'fontsize':12, 'fontweight':'bold'})
plt.legend(loc=1, prop={'size':10})
plt.tick_params(labelsize=12)
plt.show()


# In[30]:


# creating of salary with Marital status

plt.figure(figsize=(16,12))
sns.countplot(df['marital-status'], hue=df['salary'], palette='deep')
plt.title('Disturbtion Salary with Martial-Status', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.xlabel('Martial-Status', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.ylabel('Number of People', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.legend(loc=1, prop={'size':8})
plt.tick_params(labelsize=10)
plt.show()


# In[31]:


# creating countplot of salary with race

plt.figure(figsize=(12,9))
sns.countplot(df['race'], hue=df['salary'], palette='colorblind')
plt.title('Disturbtion of Salary with Race', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.xlabel('Race', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.ylabel('Number of people', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.legend(loc=1, prop={'size':10})
plt.tick_params(labelsize=12)
plt.show()


# In[32]:


# creating countplot of salary with sex

plt.figure(figsize=(7,5))
sns.countplot(df['sex'], hue=df['salary'], palette='dark')
plt.title('Disturbtion of Salary with Age', fontdict={'fontname':'Monospace', 'fontsize':15, 'fontweight':'bold'})
plt.xlabel('Sex', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.ylabel('Number of People', fontdict={'fontname':'Monospace', 'fontsize':13, 'fontweight':'bold'})
plt.legend(loc=1, prop={'size':10})
plt.tick_params(labelsize=12)
plt.show()


# # Multiple Plot with help of Pairplot

# In[33]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])


# In[34]:


# pairplot of df (adult dataset)
sns.pairplot(df)
plt.show()


# # Correlation

# In[35]:


# Correlation matrix 
  
corr = df.corr()
f,ax = plt.subplots(figsize=(7,5))
sns.heatmap(corr, cbar = True, square = True, annot = True, fmt= '.1f', 
            xticklabels= True, yticklabels= True, vmax=0.3,
            cmap="coolwarm", linewidths=.5, ax=ax)
plt.show()


# # LabelEncoder

# In[36]:


for col in df.columns:
    if df[col].dtypes == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])


# In[37]:


X = df.drop('salary',axis=1)
Y = df['salary']

for col in X.columns:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[col].values.reshape(-1,1))
# # Modelling

# In[38]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)


# In[39]:


ros.fit(X, Y)


# In[40]:


X_resampled, Y_resampled = ros.fit_resample(X, Y)


# In[41]:


round(Y_resampled.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
X_resampled, Y_resampled, test_size=0.2, random_state=42)


# In[43]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape) 


# # Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=42)


# In[45]:


log_reg.fit(X_train, Y_train)


# In[46]:


Y_pred_log_reg = log_reg.predict(X_test)


# In[47]:


print("Logistic Regression:")
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_log_reg)* 100, 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_log_reg) * 100, 2))


# # KNN Classifier 

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[49]:


knn.fit(X_train, Y_train)


# In[50]:


Y_pred_knn = knn.predict(X_test)


# In[51]:


print('KNN Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_knn) * 100, 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_knn) * 100, 2))


# # Support Vector Classifier (SVC)

# In[52]:


from sklearn.svm import SVC
svc = SVC(random_state=42)


# In[53]:


svc.fit(X_train, Y_train)


# In[54]:


Y_pred_svc = svc.predict(X_test)


# In[55]:


print('Support Vector Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_svc) * 100 , 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_svc) * 100 , 2))


# # Naive Bayes Classifier

# In[56]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


# In[57]:


nb.fit(X_train, Y_train)


# In[58]:


Y_pred_nb = nb.predict(X_test)


# In[59]:


print('Support Vector Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_nb) * 100, 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_nb) * 100 , 2))


# # Decision Tree Classifier 

# In[60]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(random_state=42)


# In[61]:


dec_tree.fit(X_train, Y_train)


# In[62]:


Y_pred_dec_tree = dec_tree.predict(X_test)


# In[63]:


print('Support Vector Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_dec_tree)* 100, 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_dec_tree) * 100, 2))


# # Random Forest Classifier

# In[64]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)


# In[65]:


rf.fit(X_train, Y_train)


# In[66]:


Y_pred_rf = rf.predict(X_test)


# In[67]:


print('Support Vector Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_rf) * 100 , 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_rf)* 100 , 2))


# # XGB Classifier

# In[68]:


from xgboost import XGBClassifier
xgb = XGBClassifier()


# In[69]:


xgb.fit(X_train, Y_train)


# In[70]:


Y_pred_xgb = xgb.predict(X_test)


# In[71]:


print('Support Vector Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_xgb) * 100 , 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_xgb) * 100 , 2))


# # AdaBoost Classifier

# In[72]:


from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(random_state=42)


# In[73]:


adaboost.fit(X_train, Y_train)


# In[74]:


Y_pred_adaboost = adaboost.predict(X_test)


# In[75]:


print('Support Vector Classifier: ')
print('Accuracy Score: ', round(accuracy_score(Y_test, Y_pred_adaboost) * 100, 2))
print('F1 Score: ', round(f1_score(Y_test, Y_pred_adaboost) * 100, 2))


# # Confusion Matrix

# In[76]:


from sklearn.metrics import confusion_matrix
con_mat = confusion_matrix(Y_test, Y_pred_rf)


# In[77]:


sns.heatmap(con_mat, annot=True, cmap='coolwarm', fmt='d')
plt.show()


# # Classification Report

# In[78]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_rf))

