from sklearn import datasets, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import pandas as pd

# if u have troubles with downloading dataset
# https://www.dropbox.com/s/fgijjemxd7x574s/20news-bydate_py3.pkz?dl=0
# download this cache and put it in scikit_learn_data folder

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
feature_mapping = vectorizer.get_feature_names()

"""grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
print(gs.best_params_)"""  # this sh... says that the best is C = 1.0

cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241, C=1.0)
clf.fit(X, y)

indices = np.argsort(np.abs(clf.coef_.toarray()).reshape(-1))[-10:]
# it's not my string but it's so cool

# print(clf.coef_.toarray())  # transfer matrix to np.array
# print(np.abs(clf.coef_.toarray()))  # take absolute values
# print(np.abs(clf.coef_.toarray()).reshape(-1))  # reshaping and -1 dimension of array from 2D to 1D
# print(np.argsort(np.abs(clf.coef_.toarray()).reshape(-1)))  # sorting and return indexes
# print(np.argsort(np.abs(clf.coef_.toarray()).reshape(-1))[-10:])  # take last 10 elements
words = []

for i in indices:
    words.append(feature_mapping[i])

with open('answers/task1.txt', 'w') as task:
    for i in sorted(words)[:len(words)-1]:
        task.write(i + ' ')
    task.write(sorted(words)[-1])
print('❤️')
