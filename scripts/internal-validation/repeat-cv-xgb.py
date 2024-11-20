# Imports
import random; random.seed(7051997)
import polars as pl
import xgboost as xgb
import statistics as stats

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score


# Hyperparameter files
df_params_cerad = pl.read_csv(f'results/hyperparameters/wearable-based-cerad-params-xgb.csv')
df_params_af = pl.read_csv(f'results/hyperparameters/wearable-based-aft-params-xgb.csv')
df_params_dsst = pl.read_csv(f'results/hyperparameters/wearable-based-dsst-params-xgb.csv')

# Initialize hyperparameters
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

# Initializing stuff
nfolds = 10
nrepeats = 20

X_dsst = pl.read_csv('results/features/features-dsst.csv').get_column('feature_names').to_list()
X_cerad = pl.read_csv('results/features/features-cerad.csv').get_column('feature_names').to_list()
X_af = pl.read_csv('results/features/features-aft.csv').get_column('feature_names').to_list()

aucs_cerad, aucs_af, aucs_dsst = [], [], []
auprcs_cerad, auprcs_af, auprcs_dsst = [], [], []

dir = 'data/nhanes/derived/cv-folds'

for i in range(1,nfolds*nrepeats+1):

    # Initialize scalers
    scaler_cerad = MinMaxScaler()
    scaler_af = MinMaxScaler()
    scaler_dsst = MinMaxScaler()

    # Read in the features and transform them
    x_train = pl.read_csv(f'{dir}/x-train-{i}.csv')
    x_valid = pl.read_csv(f'{dir}/x-valid-{i}.csv')

    x_train_dsst = x_train.select(X_dsst).to_numpy()
    x_train_cerad = x_train.select(X_cerad).to_numpy()
    x_train_af = x_train.select(X_af).to_numpy()

    x_train_dsst = scaler_dsst.fit_transform(x_train_dsst)
    x_train_cerad = scaler_cerad.fit_transform(x_train_cerad)
    x_train_af = scaler_af.fit_transform(x_train_af)

    x_valid_dsst = x_valid.select(X_dsst).to_numpy()
    x_valid_cerad = x_valid.select(X_cerad).to_numpy()
    x_valid_af = x_valid.select(X_af).to_numpy()

    x_valid_dsst = scaler_dsst.transform(x_valid_dsst)
    x_valid_cerad = scaler_cerad.transform(x_valid_cerad)
    x_valid_af = scaler_af.transform(x_valid_af)

    # Read in and isolate the targets
    targets_train = pl.read_csv(f'{dir}/y-train-{i}.csv')
    targets_valid = pl.read_csv(f'{dir}/y-valid-{i}.csv')

    y_train_cerad = targets_train.get_column('cerad_low').to_numpy()
    y_train_af = targets_train.get_column('af_low').to_numpy()
    y_train_dsst = targets_train.get_column('dsst_low').to_numpy()

    y_valid_cerad = targets_valid.get_column('cerad_low').to_numpy()
    y_valid_af = targets_valid.get_column('af_low').to_numpy()
    y_valid_dsst = targets_valid.get_column('dsst_low').to_numpy()

    # Get data into the correct format for XGBoost
    dmat_train_cerad = xgb.DMatrix(x_train_cerad, label=y_train_cerad)
    dmat_valid_cerad = xgb.DMatrix(x_valid_cerad, label=y_valid_cerad)

    dmat_train_af = xgb.DMatrix(x_train_af, label=y_train_af)
    dmat_valid_af = xgb.DMatrix(x_valid_af, label=y_valid_af)

    dmat_train_dsst = xgb.DMatrix(x_train_dsst, label=y_train_dsst)
    dmat_valid_dsst = xgb.DMatrix(x_valid_dsst, label=y_valid_dsst)

    # Initilize and fit models
    model_cerad = xgb.train(
        params_cerad,
        dtrain=dmat_train_cerad,
        num_boost_round=df_params_cerad[0,'iterations'],
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

    preds_cerad = model_cerad.predict(dmat_valid_cerad)
    preds_af = model_af.predict(dmat_valid_af)
    preds_dsst = model_dsst.predict(dmat_valid_dsst)

    # Calculate performance metrics
    aucs_cerad.append(roc_auc_score(y_valid_cerad, preds_cerad))
    aucs_af.append(roc_auc_score(y_valid_af, preds_af))
    aucs_dsst.append(roc_auc_score(y_valid_dsst, preds_dsst))

    auprcs_cerad.append(average_precision_score(y_valid_cerad, preds_cerad))
    auprcs_af.append(average_precision_score(y_valid_af, preds_af))
    auprcs_dsst.append(average_precision_score(y_valid_dsst, preds_dsst))

    # Output message to track everything
    print(f'{i} of {nfolds*nrepeats} files processed')

# Print results
print(f'Mean AUC for the CERAD model: {stats.mean(aucs_cerad)}, sd: {stats.stdev(aucs_cerad)}')
print(f'Mean AUC for the AF model: {stats.mean(aucs_af)}, sd: {stats.stdev(aucs_af)}')
print(f'Mean AUC for the DSST model: {stats.mean(aucs_dsst)}, sd: {stats.stdev(aucs_dsst)}')

print(f'Mean AUPRC for the CERAD model: {stats.mean(auprcs_cerad)}, sd: {stats.stdev(auprcs_cerad)}')
print(f'Mean AUPRC for the AF model: {stats.mean(auprcs_af)}, sd: {stats.stdev(auprcs_af)}')
print(f'Mean AUPRC for the DSST model: {stats.mean(auprcs_dsst)}, sd: {stats.stdev(auprcs_dsst)}')

# Write to data frames and save
df_output = pl.DataFrame().with_columns(
     pl.Series(name='aucs_cerad_acc', values=aucs_cerad),
     pl.Series(name='aucs_af_acc', values=aucs_af),
     pl.Series(name='aucs_dsst_acc', values=aucs_dsst),
     pl.Series(name='auprcs_cerad_acc', values=auprcs_cerad),
     pl.Series(name='auprcs_af_acc', values=auprcs_af),
     pl.Series(name='auprcs_dsst_acc', values=auprcs_dsst)
)

df_output.write_csv(f'results/performance-metrics/internal-validation-wearable-based-xgb.csv')
