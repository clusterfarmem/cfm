import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# import numpy as np
import time
import os

"""
pid = os.getpid()
os.sched_setaffinity(pid, {21,22,23,24})

# Path to the cgroup v2 hierarchy
cgroup_path = "/cgroup2"

# Name of the new cgroup
cgroup_name = "my_cgroup_131"

# Path to the new cgroup
new_cgroup_path = os.path.join(cgroup_path, cgroup_name)

# Create the new cgroup
os.makedirs(new_cgroup_path, exist_ok=True)
# Set the memory limit (in bytes)
with open(os.path.join(new_cgroup_path, "memory.high"), "w") as f:
    f.write("4250M")

with open(os.path.join(new_cgroup_path, "cgroup.procs"), "w") as f:
    f.write(str(pid))"""

start_time = time.time()


data = pd.read_csv('/users/YuqiLi/HIGGS.csv', header=None)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X = X.astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)


param = {'max_depth': 9, 'eta': 0.03, 'objective': 'binary:logistic', 'nthread': 4}
num_round = 1000  


#bst = xgb.train(param, dtrain, num_round, [(dval, 'eval')], early_stopping_rounds=50)
bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtest)

end_time = time.time()

print('AUC: ', roc_auc_score(y_test, preds))
print('Total running time: %s seconds' % (end_time - start_time))