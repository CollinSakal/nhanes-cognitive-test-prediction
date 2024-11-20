# Imports
import polars as pl
from os import listdir
from scripts.utils.getwearablefeaturesnhanes import get_wearable_features

# Files paths
dir = 'data/nhanes/individual-accelerometer-files/'
flist = listdir(dir)
fpaths = [f'{dir}{x}' for x in flist]

# Get the features
numfiles = 0
for file in fpaths:
    id_temp = file[43:48]
    df_temp = pl.read_csv(file, ignore_errors=True).select(['PAXMTSM', 'PAXPREDM', 'PAXLXMM'])

    mims_orig = df_temp.get_column('PAXMTSM')
    lux_orig = df_temp.get_column('PAXLXMM')
    mims_excl = df_temp.filter(pl.col('PAXPREDM') != 3).get_column('PAXMTSM') # Exclude non-wear
    lux_excl = df_temp.filter(pl.col('PAXPREDM') != 3).get_column('PAXLXMM')

    outpath = f'data/nhanes/derived/wearable-features-python/{id_temp}-wearable-features.csv'

    get_wearable_features(
        mims_orig=mims_orig,
        lux_orig=lux_orig,
        mims_excl=mims_excl,
        lux_excl=lux_excl,
        id=id_temp
    ).write_csv(outpath, null_value='NA')

    numfiles+=1; print(f'{numfiles} of {len(fpaths)} processed')
