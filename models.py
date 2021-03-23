#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing, metrics
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from RegscorePy import *
from statsmodels.compat import lzip
import matplotlib.gridspec as gridspec


# In[3]:


# load data
df = pd.read_csv('AUT_data.csv', encoding='utf-8')
df = df.dropna()
df.rename(columns={'rareness':'1/freq'}, inplace=True)


# In[4]:


X = df.iloc[:, 7:-1]
y = df.iloc[:, -1]
X = preprocessing.scale(X)
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)


# In[5]:


# cross-validation to find alpha
lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

#lasso regression
lasso = Lasso(max_iter = 10000, normalize=True)
lasso.set_params(alpha=lassocv.alpha_)
#print("Alpha =", lassocv.alpha_)
lasso_model = lasso.fit(X_train, y_train)

print("R2 = ", r2_score(y_test, lasso.predict(X_test)))
#np.corrcoef(lasso.predict(X_test), df.iloc[y_test.index, -1])


# In[5]:


r2_score(y_train, lasso.predict(X_train))


# In[19]:


features = df.columns[7:-1]
coef = lasso.coef_
top10 = sorted(zip(features, abs(coef)), key=lambda t: t[1], reverse=True)[:10]
plt.bar(range(len(top10)), [val[1] for val in top10], align='center')
plt.xticks(range(len(top10)), [val[0] for val in top10])
plt.xticks(rotation=70)
plt.title("Absolute Coefficient of LASSO regression")
plt.savefig('lasso.pdf', bbox_inches='tight')


# In[6]:


x, y = pd.Series(lasso.predict(X_test), name="fitted value"), pd.Series(y_test-lasso.predict(X_test), name="residual")
ax = sns.regplot(x,y)
plt.title("Residuals of LASSO Regression")
plt.savefig('lasso_residuals.pdf')


# In[7]:


x, y = pd.Series(lasso.predict(X_test), name="Predictions"), pd.Series(y_test, name="Mean Expert Ratings")
ax = sns.regplot(x,y)
plt.title("LASSO Regression Performance")
plt.savefig('lasso.png')


# In[8]:


alphas = 10**np.linspace(6,-2,50)*0.5
ridgecv = RidgeCV(alphas=alphas, normalize=True)
ridgecv.fit(X_train, y_train)

#print("Alpha=", ridgecv.alpha_)
ridge = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge.fit(X_train, y_train)
#print("mse = ",mean_squared_error(y_test, ridge.predict(X_test)))
print("R2 = ", r2_score(y_test, ridge.predict(X_test)))
np.corrcoef(ridge.predict(X_test), df.iloc[y_test.index, -1])


# In[9]:


r2_score(y_train, ridge.predict(X_train))


# In[18]:


features = df.columns[7:-1]
coef = ridge.coef_
top10 = sorted(zip(features, abs(coef)), key=lambda t: t[1], reverse=True)[:10]

plt.bar(range(len(top10)), [val[1] for val in top10], align='center')
plt.xticks(range(len(top10)), [val[0] for val in top10])
plt.xticks(rotation=70)
plt.title("Absolute Coefficients of Ridge Regression")
plt.savefig('ridge.pdf', bbox_inches='tight')


# In[10]:


x, y = pd.Series(ridge.predict(X_test), name="fitted value"), pd.Series(y_test-ridge.predict(X_test), name="residual")
ax = sns.regplot(x,y)
plt.title("Residuals of Ridge Regression")
plt.savefig('ridge_residuals.pdf')


# In[12]:


x, y = pd.Series(ridge.predict(X_test), name="Predictions"), pd.Series(y_test, name="Mean Expert Ratings")
ax = sns.regplot(x,y)
plt.title("Ridge Regression Performance")
plt.savefig('ridge.png')


# In[13]:


# Create the parameter grid based on the results of random search 
param_grid = {   
    'max_depth': [5, 7, 10],
    'n_estimators': [800, 1000], 
    'max_features': ['sqrt', 'log2'],
#     'min_samples_leaf': [2, 3],
#     'min_samples_split': [2, 5],
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv=5, n_jobs=-1, verbose=2)


# In[14]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_


# In[15]:


best_grid


# In[16]:


best_grid.score(X_train, y_train)


# In[17]:


best_grid.score(X_test, y_test)


# In[9]:


features = df.columns[7:-1]
coef = best_grid.feature_importances_
top10 = sorted(zip(features, abs(coef)), key=lambda t: t[1], reverse=True)[:10]
plt.bar(range(len(top10)), [val[1] for val in top10], align='center')
plt.xticks(range(len(top10)), [val[0] for val in top10])
plt.xticks(rotation=70)
plt.title("Feature Importance of Random Forest")
plt.savefig('rf.pdf', bbox_inches='tight')


# In[56]:


df0 = df.sample(n=10, random_state=19)
df0 = df0[['cleaned_response', 'vec_248', 'vec_281', 'mean']]
p1=sns.regplot(data=df0, x="vec_248", y="vec_281", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':400})
 
# add annotations one by one with a loop
for i in range(0,df0.shape[0]):
    row = df0.iloc[i]
    p1.text(
        row.vec_248+0.01, row.vec_281+0.02, row.cleaned_response,
        horizontalalignment='right', size='small', color='black')

fig = p1.get_figure()
fig.savefig("vectors.png")


# In[19]:


x, y = pd.Series(best_grid.predict(X_test), name="Predictions"), pd.Series(y_test, name="Mean Expert Ratings")
ax = sns.regplot(x,y)
plt.title("Performance Random Forest Regressor")
plt.savefig('random forest.png')


# In[54]:


cols = [x[0] for x in top10]
cols.append('mean')
df1 = df[cols]
sns.heatmap(df1.corr(), annot=True)


# In[283]:


df.head()


# In[ ]:


df1 = df.drop(columns=['wup_similarity', 'path_similarity'])


# In[317]:


X1 = df1.iloc[:, 7:-1]
y1 = df1.iloc[:, -1]
X1 = preprocessing.scale(X1)
#y[y>=0] = np.log(y[y>=0])
X1_train, X1_test , y1_train, y1_test = train_test_split(X1, y1, test_size = 0.3, random_state=1)
# Fit the grid search to the data
grid_search.fit(X1_train, y1_train)
best_grid = grid_search.best_estimator_


# In[318]:


best_grid.score(X1_test, y1_test)


# In[319]:


df2 = df.drop(columns=['freq', '1/freq'])
X2 = df2.iloc[:, 7:-1]
y2 = df2.iloc[:, -1]
X2 = preprocessing.scale(X2)
#y[y>=0] = np.log(y[y>=0])
X2_train, X2_test , y2_train, y2_test = train_test_split(X2, y2, test_size = 0.3, random_state=1)
# Fit the grid search to the data
grid_search.fit(X2_train, y2_train)
best_grid = grid_search.best_estimator_
best_grid.score(X2_test, y2_test)


# In[322]:


df2.head()

