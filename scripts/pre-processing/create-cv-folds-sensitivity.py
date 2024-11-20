# File for creating complete case CV folds for the sensitivity analysis

# File for creating the folds for repeated cross validation

# Imports
import random
import polars as pl

from sklearn import model_selection

# Initializing stuff
niter=1
nfolds=10
nrepeats=20
random.seed(7051997)

# Data
df_final = pl.read_csv('data/nhanes/derived/df-final-nhanes.csv', null_values=['NA'], infer_schema_length=1500).drop_nulls()

# Get names of features and targets
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
y_cols = ['cerad_low', 'af_low', 'dsst_low']

# Creating folds
for i in range(nrepeats):

    # Create folds column
    df = df_final.with_columns(pl.lit(-1).alias('fold'))

    skf = model_selection.StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random.randint(500,10000))
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X=df.select(X_cols), y=df.get_column('dsst_low'))):
        df[valid_idx, 'fold'] = fold

    # Save each split for one round of CV
    for k in range(nfolds):

        # Isolate targets (cerad, aft, dsst)
        y_train = df.filter(pl.col('fold') != k).select(y_cols)
        y_valid = df.filter(pl.col('fold') == k).select(y_cols)

        # Isolate and impute on features
        x_train = df.filter(pl.col('fold') != k).select(X_cols)
        x_valid = df.filter(pl.col('fold') == k).select(X_cols)

        # Save
        x_train.write_csv(f'data/nhanes/derived/cv-folds-sensitivity/x-train-{niter}.csv')
        x_valid.write_csv(f'data/nhanes/derived/cv-folds-sensitivity/x-valid-{niter}.csv')

        y_train.write_csv(f'data/nhanes/derived/cv-folds-sensitivity/y-train-{niter}.csv')
        y_valid.write_csv(f'data/nhanes/derived/cv-folds-sensitivity/y-valid-{niter}.csv')

        # Add one to niter, output tracking messgae
        print(f'{niter} of {nfolds*nrepeats} files processed'); niter+=1