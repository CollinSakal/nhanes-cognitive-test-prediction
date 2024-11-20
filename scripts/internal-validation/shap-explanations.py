# File for getting SHAP plots for all the models

# Imports
import shap
import numpy as np
import polars as pl
import xgboost as xgb

from sklearn.impute import KNNImputer
from catboost import Pool, CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Define columns
X_cols = [
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

y_cols = ['dsst_low', 'cerad_low', 'af_low']

# Data and features
df = pl.read_csv('data/nhanes/derived/df-final-nhanes.csv', infer_schema_length=1500, null_values=['NA'])

# Impute
imputer = KNNImputer(n_neighbors=1)
df_features = df.select(X_cols).to_numpy()
df_features = imputer.fit_transform(df_features)
df_features = pl.from_numpy(df_features, schema=X_cols)

df = df_features.hstack(df.select(y_cols))

# Get SHAP stuff
X_dsst = pl.read_csv('results/features/features-dsst.csv').get_column('feature_names').to_list()
X_cerad = pl.read_csv('results/features/features-cerad.csv').get_column('feature_names').to_list()
X_af = pl.read_csv('results/features/features-aft.csv').get_column('feature_names').to_list()

y_dsst = df.get_column('dsst_low').to_numpy()
y_cerad = df.get_column('cerad_low').to_numpy()
y_af = df.get_column('af_low').to_numpy()

scaler_dsst = MinMaxScaler()
scaler_cerad = MinMaxScaler()
scaler_af = MinMaxScaler()

x_dsst = df.select(X_dsst).to_numpy()
x_cerad = df.select(X_cerad).to_numpy()
x_af = df.select(X_af).to_numpy()

x_dsst = scaler_dsst.fit_transform(x_dsst)
x_cerad = scaler_cerad.fit_transform(x_cerad)
x_af = scaler_af.fit_transform(x_af)

# ----------------------------------------------------------------------------------------------------------------------
# CATBOOST
# ----------------------------------------------------------------------------------------------------------------------

# Hyperparameters
df_params_dsst = pl.read_csv(f'results/hyperparameters/wearable-based-dsst-params-cat.csv')
df_params_cerad = pl.read_csv(f'results/hyperparameters/wearable-based-cerad-params-cat.csv')
df_params_af = pl.read_csv(f'results/hyperparameters/wearable-based-aft-params-cat.csv')

params_dsst = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': df_params_dsst[0,'learning_rate'],
    'iterations': df_params_dsst[0,'iterations'],
    'depth': df_params_dsst[0,'depth'],
    'subsample': df_params_dsst[0,'subsample'],
    'scale_pos_weight': 3
}

params_cerad = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': df_params_cerad[0,'learning_rate'],
    'iterations': df_params_cerad[0,'iterations'],
    'depth': df_params_cerad[0,'depth'],
    'subsample': df_params_cerad[0,'subsample'],
    'scale_pos_weight': 3
}

params_af = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': False,
    'random_seed': 19970507,
    'learning_rate': df_params_af[0,'learning_rate'],
    'iterations': df_params_af[0,'iterations'],
    'depth': df_params_af[0,'depth'],
    'subsample': df_params_af[0,'subsample'],
    'scale_pos_weight': 3
}

pool_dsst = Pool(x_dsst, y_dsst, feature_names=X_dsst)
pool_cerad = Pool(x_cerad, y_cerad, feature_names=X_cerad)
pool_af = Pool(x_af, y_af, feature_names=X_af)

# Models and predictions
model_dsst = CatBoostClassifier(**params_dsst)
model_cerad = CatBoostClassifier(**params_cerad)
model_af = CatBoostClassifier(**params_af)

model_dsst.fit(pool_dsst)
model_cerad.fit(pool_cerad)
model_af.fit(pool_af)

preds_dsst = model_dsst.predict_proba(pool_dsst)
preds_cerad = model_cerad.predict_proba(pool_cerad)
preds_af = model_af.predict_proba(pool_af)

# Get SHAP explainers and values
explainer_dsst = shap.TreeExplainer(model_dsst)
explainer_cerad = shap.TreeExplainer(model_cerad)
explainer_af = shap.TreeExplainer(model_af)

shap_dsst = explainer_dsst.shap_values(x_dsst)
shap_cerad = explainer_cerad.shap_values(x_cerad)
shap_af = explainer_af.shap_values(x_af)

# Barplots
shap.summary_plot(shap_dsst, x_dsst, plot_type='bar')
shap.summary_plot(shap_cerad, x_dsst, plot_type='bar')
shap.summary_plot(shap_af, x_dsst, plot_type='bar')

mean_abs_shap_dsst = np.abs(shap_dsst).mean(axis=0)
mean_abs_shap_cerad = np.abs(shap_cerad).mean(axis=0)
mean_abs_shap_af = np.abs(shap_af).mean(axis=0)

df_mean_abs_shap_dsst = pl.DataFrame({
    'feature': X_dsst,
    'shap_meanabs': mean_abs_shap_dsst
})

df_mean_abs_shap_cerad = pl.DataFrame({
    'feature': X_cerad,
    'shap_meanabs': mean_abs_shap_cerad
})

df_mean_abs_shap_af = pl.DataFrame({
    'feature': X_af,
    'shap_meanabs': mean_abs_shap_af
})

df_mean_abs_shap_dsst.write_csv('results/shap/df-meanabs-shap-cat-dsst.csv')
df_mean_abs_shap_cerad.write_csv('results/shap/df-meanabs-shap-cat-cerad.csv')
df_mean_abs_shap_af.write_csv('results/shap/df-meanabs-shap-cat-af.csv')

# Beeswarm plots
shap.plots.beeswarm(explainer_dsst(x_dsst), max_display=len(X_dsst))
shap.plots.beeswarm(explainer_cerad(x_cerad), max_display=len(X_cerad))
shap.plots.beeswarm(explainer_af(x_af), max_display=len(X_af))

# Make SHAP data frames to create beeswarm plots in ggplot
df_shap_dsst = pl.DataFrame(shap_dsst, schema=[col+'_shap' for col in X_dsst])
df_shap_cerad = pl.DataFrame(shap_cerad, schema=[col+'_shap' for col in X_cerad])
df_shap_af = pl.DataFrame(shap_af, schema=[col+'_shap' for col in X_af])

# Add feature values
df_shap_dsst = df_shap_dsst.hstack(pl.DataFrame(x_dsst, schema=X_dsst))
df_shap_cerad = df_shap_cerad.hstack(pl.DataFrame(x_cerad, schema=X_cerad))
df_shap_af = df_shap_af.hstack(pl.DataFrame(x_af, schema=X_af))

# Add Predictions
df_shap_dsst = df_shap_dsst.with_columns(pl.Series(name='preds', values=preds_dsst[:,1]))
df_shap_cerad = df_shap_cerad.with_columns(pl.Series(name='preds', values=preds_cerad[:,1]))
df_shap_af = df_shap_af.with_columns(pl.Series(name='preds', values=preds_af[:,1]))

# Save
df_shap_dsst.write_csv('results/shap/df-shap-dsst-cat.csv')
df_shap_cerad.write_csv('results/shap/df-shap-cerad-cat.csv')
df_shap_af.write_csv('results/shap/df-shap-af-cat.csv')


# ----------------------------------------------------------------------------------------------------------------------
# XGBOOST
# ----------------------------------------------------------------------------------------------------------------------

# Hyperparameters
df_params_cerad = pl.read_csv(f'results/hyperparameters/wearable-based-cerad-params-xgb.csv')
df_params_af = pl.read_csv(f'results/hyperparameters/wearable-based-aft-params-xgb.csv')
df_params_dsst = pl.read_csv(f'results/hyperparameters/wearable-based-dsst-params-xgb.csv')

params_cerad = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 19970507,
    'eta': df_params_cerad[0,'learning_rate'],
    'max_depth': df_params_cerad[0,'depth'],
    'subsample': df_params_cerad[0,'subsample'],
    'scale_pos_weight': 3
}

params_af = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 19970507,
    'eta': df_params_af[0,'learning_rate'],
    'max_depth': df_params_af[0,'depth'],
    'subsample': df_params_af[0,'subsample'],
    'scale_pos_weight': 3
}

params_dsst = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'seed': 19970507,
    'eta': df_params_dsst[0,'learning_rate'],
    'max_depth': df_params_dsst[0,'depth'],
    'subsample': df_params_dsst[0,'subsample'],
    'scale_pos_weight': 3
}

# Get data into the correct format for XGBoost
dmat_train_cerad = xgb.DMatrix(x_cerad, label=y_cerad)
dmat_valid_cerad = xgb.DMatrix(x_cerad, label=y_cerad)

dmat_train_af = xgb.DMatrix(x_af, label=y_af)
dmat_valid_af = xgb.DMatrix(x_af, label=y_af)

dmat_train_dsst = xgb.DMatrix(x_dsst, label=y_dsst)
dmat_valid_dsst = xgb.DMatrix(x_dsst, label=y_dsst)

# Initilize and fit models
model_cerad = xgb.train(
    params_cerad,
    dtrain=dmat_train_cerad,
    num_boost_round=df_params_cerad[0, 'iterations'],
    verbose_eval=False
)

model_af = xgb.train(
    params_af,
    dtrain=dmat_train_af,
    num_boost_round=df_params_af[0, 'iterations'],
    verbose_eval=False
)

model_dsst = xgb.train(
    params_dsst,
    dtrain=dmat_train_dsst,
    num_boost_round=df_params_dsst[0, 'iterations'],
    verbose_eval=False
)

# Get SHAP explainers and values
explainer_dsst = shap.TreeExplainer(model_dsst)
explainer_cerad = shap.TreeExplainer(model_cerad)
explainer_af = shap.TreeExplainer(model_af)

shap_dsst = explainer_dsst.shap_values(x_dsst)
shap_cerad = explainer_cerad.shap_values(x_cerad)
shap_af = explainer_af.shap_values(x_af)

# Barplots
shap.summary_plot(shap_dsst, x_dsst, plot_type='bar')
shap.summary_plot(shap_cerad, x_dsst, plot_type='bar')
shap.summary_plot(shap_af, x_dsst, plot_type='bar')

mean_abs_shap_dsst = np.abs(shap_dsst).mean(axis=0)
mean_abs_shap_cerad = np.abs(shap_cerad).mean(axis=0)
mean_abs_shap_af = np.abs(shap_af).mean(axis=0)

df_mean_abs_shap_dsst = pl.DataFrame({
    'feature': X_dsst,
    'shap_meanabs': mean_abs_shap_dsst
})

df_mean_abs_shap_cerad = pl.DataFrame({
    'feature': X_cerad,
    'shap_meanabs': mean_abs_shap_cerad
})

df_mean_abs_shap_af = pl.DataFrame({
    'feature': X_af,
    'shap_meanabs': mean_abs_shap_af
})

df_mean_abs_shap_dsst.write_csv('results/shap/df-meanabs-shap-xgb-dsst.csv')
df_mean_abs_shap_cerad.write_csv('results/shap/df-meanabs-shap-xgb-cerad.csv')
df_mean_abs_shap_af.write_csv('results/shap/df-meanabs-shap-xgb-af.csv')

# Beeswarm plots
shap.plots.beeswarm(explainer_dsst(x_dsst), max_display=len(X_dsst))
shap.plots.beeswarm(explainer_cerad(x_cerad), max_display=len(X_cerad))
shap.plots.beeswarm(explainer_af(x_af), max_display=len(X_af))

# Make SHAP data frames to create beeswarm plots in ggplot
df_shap_dsst = pl.DataFrame(shap_dsst, schema=[col+'_shap' for col in X_dsst])
df_shap_cerad = pl.DataFrame(shap_cerad, schema=[col+'_shap' for col in X_cerad])
df_shap_af = pl.DataFrame(shap_af, schema=[col+'_shap' for col in X_af])

# Add feature values
df_shap_dsst = df_shap_dsst.hstack(pl.DataFrame(x_dsst, schema=X_dsst))
df_shap_cerad = df_shap_cerad.hstack(pl.DataFrame(x_cerad, schema=X_cerad))
df_shap_af = df_shap_af.hstack(pl.DataFrame(x_af, schema=X_af))

# Add Predictions
df_shap_dsst = df_shap_dsst.with_columns(pl.Series(name='preds', values=preds_dsst[:,1]))
df_shap_cerad = df_shap_cerad.with_columns(pl.Series(name='preds', values=preds_cerad[:,1]))
df_shap_af = df_shap_af.with_columns(pl.Series(name='preds', values=preds_af[:,1]))

# Save
df_shap_dsst.write_csv('results/shap/df-shap-dsst-xgb.csv')
df_shap_cerad.write_csv('results/shap/df-shap-cerad-xgb.csv')
df_shap_af.write_csv('results/shap/df-shap-af-xgb.csv')

# ----------------------------------------------------------------------------------------------------------------------
# RANDOM FOREST
# ----------------------------------------------------------------------------------------------------------------------

# Hyperparameters
df_params_cerad = pl.read_csv(f'results/hyperparameters/wearable-based-cerad-params-rft.csv')
df_params_af = pl.read_csv(f'results/hyperparameters/wearable-based-aft-params-rft.csv')
df_params_dsst = pl.read_csv(f'results/hyperparameters/wearable-based-dsst-params-rft.csv')

params_cerad = {
    'n_estimators': df_params_cerad[0,'iterations'],
    'max_depth': df_params_cerad[0,'depth'],
    'n_jobs': 10
}

params_af = {
    'n_estimators': df_params_af[0,'iterations'],
    'max_depth': df_params_af[0,'depth'],
    'n_jobs': 10
}

params_dsst = {
    'n_estimators': df_params_dsst[0,'iterations'],
    'max_depth': df_params_dsst[0,'depth'],
    'n_jobs': 10
}

model_cerad = RandomForestClassifier(**params_cerad)
model_af = RandomForestClassifier(**params_af)
model_dsst = RandomForestClassifier(**params_dsst)

model_cerad.fit(x_cerad, y_cerad)
model_af.fit(x_af, y_af)
model_dsst.fit(x_dsst, y_dsst)

# Get SHAP explainers and values
explainer_dsst = shap.Explainer(model_dsst)
explainer_cerad = shap.Explainer(model_cerad)
explainer_af = shap.Explainer(model_af)

shap_dsst = explainer_dsst.shap_values(x_dsst)
shap_cerad = explainer_cerad.shap_values(x_cerad)
shap_af = explainer_af.shap_values(x_af)

# Barplots
shap.summary_plot(shap_dsst[1], x_dsst, plot_type='bar')
shap.summary_plot(shap_cerad[1], x_dsst, plot_type='bar')
shap.summary_plot(shap_af[1], x_dsst, plot_type='bar')

mean_abs_shap_dsst = np.abs(shap_dsst[1]).mean(axis=0)
mean_abs_shap_cerad = np.abs(shap_cerad[1]).mean(axis=0)
mean_abs_shap_af = np.abs(shap_af[1]).mean(axis=0)

df_mean_abs_shap_dsst = pl.DataFrame({
    'feature': X_dsst,
    'shap_meanabs': mean_abs_shap_dsst
})

df_mean_abs_shap_cerad = pl.DataFrame({
    'feature': X_cerad,
    'shap_meanabs': mean_abs_shap_cerad
})

df_mean_abs_shap_af = pl.DataFrame({
    'feature': X_af,
    'shap_meanabs': mean_abs_shap_af
})

df_mean_abs_shap_dsst.write_csv('results/shap/df-meanabs-shap-rft-dsst.csv')
df_mean_abs_shap_cerad.write_csv('results/shap/df-meanabs-shap-rft-cerad.csv')
df_mean_abs_shap_af.write_csv('results/shap/df-meanabs-shap-rft-af.csv')


# Beeswarm Plots
shap.summary_plot(shap_dsst[1], x_dsst, plot_type='dot')
shap.summary_plot(shap_cerad[1], x_dsst, plot_type='dot')
shap.summary_plot(shap_af[1], x_dsst, plot_type='dot')

# Make SHAP data frames to create beeswarm plots in ggplot
df_shap_dsst = pl.DataFrame(shap_dsst[1], schema=[col+'_shap' for col in X_dsst])
df_shap_cerad = pl.DataFrame(shap_cerad[1], schema=[col+'_shap' for col in X_cerad])
df_shap_af = pl.DataFrame(shap_af[1], schema=[col+'_shap' for col in X_af])

# Add feature values
df_shap_dsst = df_shap_dsst.hstack(pl.DataFrame(x_dsst, schema=X_dsst))
df_shap_cerad = df_shap_cerad.hstack(pl.DataFrame(x_cerad, schema=X_cerad))
df_shap_af = df_shap_af.hstack(pl.DataFrame(x_af, schema=X_af))

# Add Predictions
df_shap_dsst = df_shap_dsst.with_columns(pl.Series(name='preds', values=preds_dsst[:,1]))
df_shap_cerad = df_shap_cerad.with_columns(pl.Series(name='preds', values=preds_cerad[:,1]))
df_shap_af = df_shap_af.with_columns(pl.Series(name='preds', values=preds_af[:,1]))

# Save
df_shap_dsst.write_csv('results/shap/df-shap-dsst-rft.csv')
df_shap_cerad.write_csv('results/shap/df-shap-cerad-rft.csv')
df_shap_af.write_csv('results/shap/df-shap-af-rft.csv')