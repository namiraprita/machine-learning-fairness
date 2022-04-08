import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC#导入SVM模型
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss


df = pd.read_csv("compas-scores-two-years(cleaned).csv")

df['sex'] = df['sex'].apply(lambda sex: 0 if sex == 'Female' else 1)
df['age_cat'] = df['age_cat'].apply(lambda age_cat: 2 if age_cat == '> 45' else(1 if age_cat == '25 - 45' else 0))
df['race'] = df['race'].apply(lambda race: 0 if race == 'African-American' else 1)
df['c_charge_degree'] = df['c_charge_degree'].apply(lambda c_charge_degree: 0 if c_charge_degree == 'M' else 1)
features = ['sex', 'age_cat', 'priors_count', 'c_charge_degree', 'length_of_stay']
sensitive = 'race'
target = 'two_year_recid'
def process_df(df):
    y_label = df[target]
    protected_attr = df[sensitive]
    df_new = df[features]
    y_label, protected_attr, df_new = shuffle(y_label, protected_attr, df_new, random_state = 617)
    
    return y_label.to_numpy(), protected_attr.to_numpy(), df_new.to_numpy()

# Split data into train and test
y_label, protected_attr, df_new =  process_df(df)
train_index = int(len(df_new) * 0.7)
x_train, y_train, race_train = df_new[:train_index], y_label[:train_index], protected_attr[:train_index]
x_test, y_test, race_test = df_new[train_index:], y_label[train_index:],protected_attr[train_index:]
def p_rule(sensitive_var, y_pred):
    protected = np.where(sensitive_var == 1)[0]
    not_protected = np.where(sensitive_var == 0)[0]
    protected_pred = np.where(y_pred[protected] == 1)
    not_protected_pred = np.where(y_pred[not_protected] == 1)
    protected_percent = protected_pred[0].shape[0]/protected.shape[0]
    not_protected_percent = not_protected_pred[0].shape[0]/not_protected.shape[0]
    ratio = min(protected_percent/not_protected_percent, not_protected_percent/protected_percent)
    
    return ratio, protected_percent, not_protected_percent


svm_model = SVC(kernel='rbf', probability=True)

# Train model and print results
clf = svm_model.fit(x_train, y_train)
optimal_loss = log_loss(y_train, clf.predict_proba(x_train))
print_results = {"Set": ["Train", "Test"],
                 "Accuracy (%)": [clf.score(x_train, y_train)*100, clf.score(x_test, y_test)*100],
                 "P-rule (%)": [p_rule(race_train, clf.predict(x_train))[0]*100, p_rule(race_test, clf.predict(x_test))[0]*100],
                 "Protected (%)": [p_rule(race_train, clf.predict(x_train))[1]*100, p_rule(race_test, clf.predict(x_test))[1]*100],
                 "Not protected (%)": [p_rule(race_train, clf.predict(x_train))[2]*100, p_rule(race_test, clf.predict(x_test))[2]*100]}
pd.DataFrame(print_results)
print(print_results)
