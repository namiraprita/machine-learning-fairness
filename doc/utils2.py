# Load modules
import itertools
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.utils import shuffle
import copy
import math

def process_df2(compas_df): 
    # Drop NA values and select variables
    compas_df.dropna()
    compas_subset = compas_df[["sex","age","age_cat","race","priors_count","c_charge_degree","c_jail_in", "c_jail_out",'two_year_recid']]
    compas_subset["two_year_recid"] = compas_subset["two_year_recid"].apply(lambda x: -1 if x==0 else 1)
    
    # Filter for African-Americans and Caucasians and apply 0 and 1 encoding respectively
    compas_subset = compas_subset[(compas_subset["race"]=='Caucasian') | (compas_subset["race"]=='African-American') ]
    compas_subset["race_cat"] = compas_subset["race"].apply(lambda x: 1 if x == "Caucasian" else 0)
    compas_subset = compas_subset.drop(columns = "race")
    
    # Encode 0 and 1 for females and males respectively
    compas_subset["gender_cat"] = compas_subset["sex"].apply(lambda x: 1 if x == "Female" else 0)
    compas_subset = compas_subset.drop(columns = "sex")
    
    # Encode 0 and 1 for c_charge_degree M and F respectively 
    compas_subset["charge_cat"] = compas_subset["c_charge_degree"].apply(lambda x: 1 if x == "F" else 0)
    compas_subset = compas_subset.drop(columns = "c_charge_degree")
    
    # Compute length of stay and encoding 0, 1 and 2 for x<=7, 7<x<=90 and x>90 respectively
    compas_subset["length_stay"] = pd.to_datetime(compas_subset["c_jail_out"]) - pd.to_datetime(compas_subset['c_jail_in'])
    compas_subset["length_stay"] = compas_subset["length_stay"].apply(lambda x: x.days)
    compas_subset = compas_subset.drop(columns = ["c_jail_in","c_jail_out"])
    compas_subset['length_stay'] = compas_subset["length_stay"].apply(lambda x: 0 if x <= 7 else x)
    compas_subset['length_stay'] = compas_subset["length_stay"].apply(lambda x: 1 if 7 < x <= 90 else x)
    compas_subset['length_stay'] = compas_subset["length_stay"].apply(lambda x: 2 if x > 90 else x)
    
    # Encoding priors count into 0, 1 and 2 for x=0, 1<=x<=3 and x>3 respectively
    compas_subset["priors_count"] = compas_subset["priors_count"].apply(lambda x: 0 if x == 0 else x)
    compas_subset["priors_count"] = compas_subset["priors_count"].apply(lambda x: 1 if (1 <= x <= 3) else x)
    compas_subset["priors_count"] = compas_subset["priors_count"].apply(lambda x: 2 if x > 3 else x)
    
    # Encode age cat into 0, 1 and 2 for >45, 25-45 and <25
    compas_subset = compas_subset.drop(columns = ["age"])
    compas_subset['age_cat'] = compas_subset['age_cat'].apply(lambda age_cat: 2 if age_cat == '> 45' else(1 if age_cat == '25 - 45' else 0))
    
    compas_subset = compas_subset.dropna()
    y_label = compas_subset["two_year_recid"]
    protected_attr = compas_subset["race_cat"]
    df = compas_subset.drop(columns=["two_year_recid","race_cat"])
    y_label, protected_attr, df = shuffle(y_label, protected_attr, df, random_state = 0)

    return y_label.to_numpy(), protected_attr.to_numpy(), df.to_numpy()


# Print function
def shapley_df(data1,data2):
    feature_names = ['age_cat', 'priors_count', 'sex', 'c_charge_degree', 'length_of_stay']
    shapley_df = pd.DataFrame(list(zip(feature_names, data1, data2)),
                          columns=["Feature", "Shapley Discrimination", "Shapley Accuracy"])
    shapley_df = shapley_df.sort_values(by=["Shapley Discrimination"], ascending=False, ignore_index=True)
    
    return shapley_df

# Calculate the Shapley coefficients
def shapley_Cal(train_set):
    shap_acc = []
    shap_disc = []
    for i in range(5):
        acc_i = get_shapley_acc_i(train_set[0],train_set[1], train_set[2],i)
        disc_i = get_shapley_disc_i(train_set[0],train_set[1], train_set[2], i)

        shap_acc.append(acc_i)
        shap_disc.append(disc_i)
    return [shap_acc, shap_disc]

def set_split_train(a, b ,c):
    size = len(c)
    split_propotion = 7/10
    index = int(size* split_propotion)
     
    return [a[: index], c[:index], b[:index]]

def set_split_test(a, b ,c):
    size = len(c)
    split_propotion = 7/10
    index = int(size* split_propotion)
     
    return [a[index:], c[index:], b[index:]]


def get_uniq_vals_in_arr(arr):
    uniq_vals = []
    for id_col in range(arr.shape[1]):
        uniq_vals.append(np.unique(arr[:, id_col]).tolist())
    
    return uniq_vals


def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item


def get_info_coef(left, right):
    # Both arrays NEED same number of rows
    assert left.shape[0] == right.shape[0]
    num_rows = left.shape[0]
    num_left_cols = left.shape[1]
        
    concat_mat = np.concatenate((left, right), axis=1)
    concat_uniq_vals = get_uniq_vals_in_arr(concat_mat)
    concat_combos = list(itertools.product(*concat_uniq_vals))
    p_sum = 0
    for vec in concat_combos:
        p_r1_r2 = len(np.where((concat_mat == vec).all(axis=1))[0]) / num_rows
        p_r1 = len(np.where((left == vec[:num_left_cols]).all(axis=1))[0]) / num_rows
        p_r2 = len(np.where((right == vec[num_left_cols:]).all(axis=1))[0]) / num_rows
        
        if p_r1_r2 == 0 or p_r1 == 0 or p_r2 == 0:
            p_iter = 0
        else:
            p_iter = p_r1_r2 * np.log(p_r1_r2 / p_r1) / p_r1
        p_sum += np.abs(p_iter)
    return p_sum


def get_conditional_info_coef(left, right, conditional): 
    assert (left.shape[0] == right.shape[0]) and (left.shape[0] == conditional.shape[0])
    num_rows = left.shape[0]
    num_left_cols = left.shape[1]
    num_right_cols = right.shape[1]

    right_concat_mat = np.concatenate((right, conditional), axis=1)    
    concat_mat = np.concatenate((left, right_concat_mat), axis=1)
    concat_uniq_vals = get_uniq_vals_in_arr(concat_mat)
    concat_combos = list(itertools.product(*concat_uniq_vals))
    p_sum = 0
    for vec in concat_combos:
        p_r1_r2 = len(np.where((concat_mat == vec).all(axis=1))[0]) / num_rows
        p_r1 = len(np.where((left == vec[:num_left_cols]).all(axis=1))[0]) / num_rows
        p_r2 = len(np.where((concat_mat[:, num_left_cols: -num_right_cols] == vec[num_left_cols: -num_right_cols]).all(axis=1))[0]) / num_rows
        
        try:
            p_r1_given_r3 = len(np.where((concat_mat[:, :num_left_cols] == vec[:num_left_cols]).all(axis=1) & (concat_mat[:, -num_right_cols:] == vec[-num_right_cols:]).all(axis=1))[0]) / len(np.where((concat_mat[:, -num_right_cols:] == vec[-num_right_cols:]).all(axis=1))[0])
        except ZeroDivisionError:
            p_r1_given_r3 = 0
        
        if p_r1_r2 == 0 or p_r1 == 0 or p_r2 == 0 or p_r1_given_r3 == 0:
            p_iter = 0
        else:
            p_iter = p_r1_r2 * np.log(p_r1_r2 / p_r2) / p_r1_given_r3
        p_sum += np.abs(p_iter)
    return p_sum


def get_acc_coef(y, x_s, x_s_c, protected_attr):
    conditional = np.concatenate((x_s_c, protected_attr), axis=1)
    return get_conditional_info_coef(y, x_s, conditional)


def get_disc_coef(y, x_s, protected_attr):
    x_s_a = np.concatenate((x_s, protected_attr), axis=1)
    return get_info_coef(y, x_s_a) * get_info_coef(x_s, protected_attr) * get_conditional_info_coef(x_s, protected_attr, y)


def get_shapley_acc_i(y, x, protected_attr, i):
    num_features = x.shape[1]
    lst_idx = list(range(num_features))
    lst_idx.pop(i)
    power_set = [x for x in powerset(lst_idx) if len(x) > 0]
    
    shapley = 0
    for set_idx in power_set:
        coef = math.factorial(len(set_idx)) * math.factorial(num_features - len(set_idx) - 1) / math.factorial(num_features)
        
        # Calculate v(T U {i})
        idx_xs_incl = copy.copy(set_idx)
        idx_xs_incl.append(i)
        idx_xsc_incl = list(set(list(range(num_features))).difference(set(idx_xs_incl)))
        acc_incl = get_acc_coef(y.reshape(-1, 1), x[:, idx_xs_incl], x[:, idx_xsc_incl], protected_attr.reshape(-1, 1))
        
        # Calculate v(T)
        idx_xsc_excl = list(range(num_features))
        idx_xsc_excl.pop(i)
        idx_xsc_excl = list(set(idx_xsc_excl).difference(set(set_idx)))
        acc_excl = get_acc_coef(y.reshape(-1, 1), x[:, set_idx], x[:, idx_xsc_excl], protected_attr.reshape(-1, 1))
        
        marginal = acc_incl - acc_excl
        shapley = shapley + coef * marginal
    return shapley


def get_shapley_disc_i(y, x, protected_attr, i):
    num_features = x.shape[1]
    lst_idx = list(range(num_features))
    lst_idx.pop(i)
    power_set = [x for x in powerset(lst_idx) if len(x) > 0]
    
    shapley = 0
    for set_idx in power_set:
        coef = math.factorial(len(set_idx)) * math.factorial(num_features - len(set_idx) - 1) / math.factorial(num_features)
        
        # Calculate v_D(T U {i})
        idx_xs_incl = copy.copy(set_idx)
        idx_xs_incl.append(i)
        disc_incl = get_disc_coef(y.reshape(-1, 1), x[:, idx_xs_incl], protected_attr.reshape(-1, 1))
        
        # Calculate v_D(T)
        disc_excl = get_disc_coef(y.reshape(-1, 1), x[:, set_idx], protected_attr.reshape(-1, 1))
        
        marginal = disc_incl - disc_excl
        shapley = shapley + coef * marginal
    return shapley
