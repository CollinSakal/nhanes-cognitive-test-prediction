# File for selecting features in the CatBoost model


# Imports
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler

# Initializing stuff
dir = 'data/nhanes/derived/cv-folds'
nfolds = 10
candidate_features = [
    'age', 'education', 'avg_sed', 'std_sed', 'avg_lgt', 'std_lgt', 'avg_mvp', 'std_mvp',
    'avg_slp_onset', 'avg_slp_wakeup', 'std_slp_onset', 'std_slp_wakeup', 'avg_slp_dur',
    'avg_slp_eff', 'std_slp_dur', 'std_slp_eff', 'avg_L5_midpoint', 'std_L5_midpoint', 'avg_L5_activity',
    'std_L5_activity', 'avg_M10_midpoint', 'std_M10_midpoint', 'avg_M10_activity', 'std_M10_activity', 'avg_RA',
    'std_RA', 'IS', 'avg_IV', 'std_IV', 'avg_activity_nonsleep', 'std_activity_nonsleep', 'avg_activity_sleep',
    'std_activity_sleep', 'avg_mims', 'med_mims', 'min_mims', 'max_mims', 'std_mims', 'q10_mims', 'q25_mims',
    'q75_mims', 'q90_mims', 'skw_mims', 'krt_mims', 'frq_01_mims', 'frq_02_mims', 'frq_03_mims',
    'frq_04_mims', 'frq_05_mims', 'frq_06_mims', 'frq_07_mims', 'frq_08_mims', 'frq_09_mims', 'frq_10_mims',
    'frq_11_mims', 'frq_12_mims', 'frq_13_mims', 'frq_14_mims', 'frq_15_mims', 'amp_8h_mims', 'amp_12h_mims',
    'amp_24h_mims', 'avg_L5_lux', 'std_L5_lux', 'avg_M10_lux', 'std_M10_lux',
    'avg_lux_nonsleep', 'std_lux_nonsleep', 'avg_lux_sleep', 'std_lux_sleep', 'avg_lux', 'med_lux', 'min_lux',
    'max_lux', 'std_lux', 'q10_lux', 'q25_lux', 'q75_lux', 'q90_lux', 'skw_lux', 'krt_lux', 'frq_01_lux', 'frq_02_lux',
    'frq_03_lux', 'frq_04_lux', 'frq_05_lux', 'frq_06_lux', 'frq_07_lux', 'frq_08_lux', 'frq_09_lux', 'frq_10_lux',
    'frq_11_lux', 'frq_12_lux', 'frq_13_lux', 'frq_14_lux', 'frq_15_lux', 'amp_8h_lux', 'amp_12h_lux', 'amp_24h_lux'
]

params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': 0.015,
    'iterations': 500,
    'depth': 4,
    'subsample': 0.80,
    'scale_pos_weight': 3
}

# Data
df_full = pl.read_csv('data/nhanes/derived/df-final-nhanes.csv', null_values=['NA'], infer_schema_length=1500)

# RECALL MODEL
target = 'cerad_low'

# Train default model
x = df_full.select(candidate_features).to_numpy()
y = df_full.get_column(target).to_numpy()
pool = Pool(x, y, feature_names=candidate_features)

model = CatBoostClassifier(**params)
model.fit(pool, eval_set=pool)

# Get mean absolute shap values (returns objects x nfeatures +1, exclude last column)
importance_vals = model.get_feature_importance(pool, type='ShapValues')[:,:-1]
importance_vals = np.abs(importance_vals)
importance_vals = np.mean(importance_vals, axis=0)

df_importance = pl.DataFrame({
    'feature':candidate_features,
    'importance':importance_vals
}).sort(pl.col('importance'), descending=True)
df_importance.write_csv(f'data/nhanes/derived/catboost-mean-shap-{target}.csv')

# Forward selection: 10-fold CV for default model adding one feature at a time
added_feature = []
auc_temp = []
auc_avg = []
auc_std = []

for feature in df_importance.get_column('feature').to_list():

    # Features that will be used to train the model
    added_feature.append(feature)

    for fold in range(1,nfolds+1):

        scaler = MinMaxScaler()

        x_train = pl.read_csv(f'{dir}/x-train-{fold}.csv').select(added_feature).to_numpy()
        x_valid = pl.read_csv(f'{dir}/x-valid-{fold}.csv').select(added_feature).to_numpy()

        y_train = pl.read_csv(f'{dir}/y-train-{fold}.csv').get_column(target).to_numpy()
        y_valid = pl.read_csv(f'{dir}/y-valid-{fold}.csv').get_column(target).to_numpy()

        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

        pool_train = Pool(x_train, y_train, feature_names=added_feature)
        pool_valid = Pool(x_valid, y_valid, feature_names=added_feature)

        cat = CatBoostClassifier(**params)
        cat.fit(pool_train, eval_set=pool_valid)

        cat_preds_valid = cat.predict_proba(pool_valid)

        auc_temp.append(roc_auc_score(y_valid, cat_preds_valid[:,1]))

        if fold == nfolds-1:
            avg_auc_temp = np.mean(auc_temp)
            std_auc_temp = np.std(auc_temp)
            auc_avg.append(avg_auc_temp)
            auc_std.append(std_auc_temp)
            auc_temp = []

            print(f'Average AUC: {avg_auc_temp} .... std: {std_auc_temp}')

# Create data frame to save selection metrics
df_selection = pl.DataFrame({
    'added_feature':added_feature,
    'auc_avg':auc_avg,
    'auc_std':auc_std
})

df_selection.write_csv(f'data/nhanes/derived/catboost-feature-selection-{target}.csv')

print(f'Finished procedure for {target}')

# DSST MODEL
target = 'dsst_low'

# Train default model
x = df_full.select(candidate_features).to_numpy()
y = df_full.get_column(target).to_numpy()
pool = Pool(x, y, feature_names=candidate_features)

model = CatBoostClassifier(**params)
model.fit(pool, eval_set=pool)

# Get mean absolute shap values (returns objects x nfeatures +1, exclude last column)
importance_vals = model.get_feature_importance(pool, type='ShapValues')[:,:-1]
importance_vals = np.abs(importance_vals)
importance_vals = np.mean(importance_vals, axis=0)

df_importance = pl.DataFrame({
    'feature':candidate_features,
    'importance':importance_vals
}).sort(pl.col('importance'), descending=True)
df_importance.write_csv(f'data/nhanes/derived/catboost-mean-shap-{target}.csv')

# Forward selection: 10-fold CV for default model adding one feature at a time
added_feature = []
auc_temp = []
auc_avg = []
auc_std = []

for feature in df_importance.get_column('feature').to_list():

    # Features that will be used to train the model
    added_feature.append(feature)

    for fold in range(1,nfolds+1):

        scaler = MinMaxScaler()

        x_train = pl.read_csv(f'{dir}/x-train-{fold}.csv').select(added_feature).to_numpy()
        x_valid = pl.read_csv(f'{dir}/x-valid-{fold}.csv').select(added_feature).to_numpy()

        y_train = pl.read_csv(f'{dir}/y-train-{fold}.csv').get_column(target).to_numpy()
        y_valid = pl.read_csv(f'{dir}/y-valid-{fold}.csv').get_column(target).to_numpy()

        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

        pool_train = Pool(x_train, y_train, feature_names=added_feature)
        pool_valid = Pool(x_valid, y_valid, feature_names=added_feature)

        cat = CatBoostClassifier(**params)
        cat.fit(pool_train, eval_set=pool_valid)

        cat_preds_valid = cat.predict_proba(pool_valid)

        auc_temp.append(roc_auc_score(y_valid, cat_preds_valid[:,1]))

        if fold == nfolds-1:
            avg_auc_temp = np.mean(auc_temp)
            std_auc_temp = np.std(auc_temp)
            auc_avg.append(avg_auc_temp)
            auc_std.append(std_auc_temp)
            auc_temp = []

            print(f'Average AUC: {avg_auc_temp} .... std: {std_auc_temp}')

# Create data frame to save selection metrics
df_selection = pl.DataFrame({
    'added_feature':added_feature,
    'auc_avg':auc_avg,
    'auc_std':auc_std
})

df_selection.write_csv(f'data/nhanes/derived/catboost-feature-selection-{target}.csv')

print(f'Finished procedure for {target}')

# AFT MODEL
target = 'af_low'

# Train default model
x = df_full.select(candidate_features).to_numpy()
y = df_full.get_column(target).to_numpy()
pool = Pool(x, y, feature_names=candidate_features)

model = CatBoostClassifier(**params)
model.fit(pool, eval_set=pool)

# Get mean absolute shap values (returns objects x nfeatures +1, exclude last column)
importance_vals = model.get_feature_importance(pool, type='ShapValues')[:,:-1]
importance_vals = np.abs(importance_vals)
importance_vals = np.mean(importance_vals, axis=0)

df_importance = pl.DataFrame({
    'feature':candidate_features,
    'importance':importance_vals
}).sort(pl.col('importance'), descending=True)
df_importance.write_csv(f'data/nhanes/derived/catboost-mean-shap-{target}.csv')

# Forward selection: 10-fold CV for default model adding one feature at a time
added_feature = []
auc_temp = []
auc_avg = []
auc_std = []

for feature in df_importance.get_column('feature').to_list():

    # Features that will be used to train the model
    added_feature.append(feature)

    for fold in range(1,nfolds+1):

        scaler = MinMaxScaler()

        x_train = pl.read_csv(f'{dir}/x-train-{fold}.csv').select(added_feature).to_numpy()
        x_valid = pl.read_csv(f'{dir}/x-valid-{fold}.csv').select(added_feature).to_numpy()

        y_train = pl.read_csv(f'{dir}/y-train-{fold}.csv').get_column(target).to_numpy()
        y_valid = pl.read_csv(f'{dir}/y-valid-{fold}.csv').get_column(target).to_numpy()

        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

        pool_train = Pool(x_train, y_train, feature_names=added_feature)
        pool_valid = Pool(x_valid, y_valid, feature_names=added_feature)

        cat = CatBoostClassifier(**params)
        cat.fit(pool_train, eval_set=pool_valid)

        cat_preds_valid = cat.predict_proba(pool_valid)

        auc_temp.append(roc_auc_score(y_valid, cat_preds_valid[:, 1]))

        if fold == nfolds-1:
            avg_auc_temp = np.mean(auc_temp)
            std_auc_temp = np.std(auc_temp)
            auc_avg.append(avg_auc_temp)
            auc_std.append(std_auc_temp)
            auc_temp = []

            print(f'Average AUC: {avg_auc_temp} .... std: {std_auc_temp}')

# Create data frame to save selection metrics
df_selection = pl.DataFrame({
    'added_feature':added_feature,
    'auc_avg':auc_avg,
    'auc_std':auc_std
})

df_selection.write_csv(f'data/nhanes/derived/catboost-feature-selection-{target}.csv')

print(f'Finished procedure for {target}')