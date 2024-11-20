# Imports
import random; random.seed(7051997)
import polars as pl
import statistics as stats

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score

# Hyperparameter files
df_params_cerad = pl.read_csv(f'results/hyperparameters/wearable-based-cerad-params-rft.csv')
df_params_af = pl.read_csv(f'results/hyperparameters/wearable-based-aft-params-rft.csv')
df_params_dsst = pl.read_csv(f'results/hyperparameters/wearable-based-dsst-params-rft.csv')

# Initialize hyperparameters
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

    # Initilize and fit models
    model_cerad = RandomForestClassifier(**params_cerad)
    model_af = RandomForestClassifier(**params_af)
    model_dsst = RandomForestClassifier(**params_dsst)

    model_cerad.fit(x_train_cerad, y_train_cerad)
    model_af.fit(x_train_af, y_train_af)
    model_dsst.fit(x_train_dsst, y_train_dsst)

    preds_cerad = model_cerad.predict_proba(x_valid_cerad)
    preds_af = model_af.predict_proba(x_valid_af)
    preds_dsst = model_dsst.predict_proba(x_valid_dsst)

    # Calculate performance metrics
    aucs_cerad.append(roc_auc_score(y_valid_cerad, preds_cerad[:,1]))
    aucs_af.append(roc_auc_score(y_valid_af, preds_af[:,1]))
    aucs_dsst.append(roc_auc_score(y_valid_dsst, preds_dsst[:,1]))

    auprcs_cerad.append(average_precision_score(y_valid_cerad, preds_cerad[:,1]))
    auprcs_af.append(average_precision_score(y_valid_af, preds_af[:,1]))
    auprcs_dsst.append(average_precision_score(y_valid_dsst, preds_dsst[:,1]))

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

df_output.write_csv(f'results/performance-metrics/internal-validation-wearable-based-rft.csv')
