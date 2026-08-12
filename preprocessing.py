import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = 'SalePrice'

NONE_COLS = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']

QUAL_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
QUAL_COLS = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

BSMT_EXPOSURE_MAP = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
BSMT_FINTYPE_MAP = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
GARAGE_FINISH_MAP = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
FUNCTIONAL_MAP = {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4,
                   'Min2': 5, 'Min1': 6, 'Typ': 7}
LOTSHAPE_MAP = {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}
LANDSLOPE_MAP = {'Sev': 0, 'Mod': 1, 'Gtl': 2}
PAVEDDRIVE_MAP = {'N': 0, 'P': 1, 'Y': 2}
UTILITIES_MAP = {'NoSeWa': 0, 'AllPub': 1}
CENTRALAIR_MAP = {'N': 0, 'Y': 1}

ORDINAL_MAPS = {
    'BsmtExposure': BSMT_EXPOSURE_MAP,
    'BsmtFinType1': BSMT_FINTYPE_MAP,
    'BsmtFinType2': BSMT_FINTYPE_MAP,
    'GarageFinish': GARAGE_FINISH_MAP,
    'Functional': FUNCTIONAL_MAP,
    'LotShape': LOTSHAPE_MAP,
    'LandSlope': LANDSLOPE_MAP,
    'PavedDrive': PAVEDDRIVE_MAP,
    'Utilities': UTILITIES_MAP,
    'CentralAir': CENTRALAIR_MAP,
}

LOTFRONTAGE_COL = 'LotFrontage'


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Limpieza e ingeniería de variables específica del dataset Ames Housing."""

    def fit(self, X, y=None):
        X = X.copy()
        for col in NONE_COLS:
            if col in X.columns:
                X[col] = X[col].fillna('None')
        self.frontage_medians_ = X.groupby('Neighborhood')[LOTFRONTAGE_COL].median()
        self.frontage_global_median_ = X[LOTFRONTAGE_COL].median()
        return self

    def transform(self, X):
        X = X.copy()

        for col in NONE_COLS:
            if col in X.columns:
                X[col] = X[col].fillna('None')

        X[LOTFRONTAGE_COL] = X.apply(
            lambda row: self.frontage_medians_.get(row['Neighborhood'],
                                                     self.frontage_global_median_)
            if pd.isna(row[LOTFRONTAGE_COL]) else row[LOTFRONTAGE_COL],
            axis=1
        )

        if 'MasVnrArea' in X.columns:
            X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
        if 'GarageYrBlt' in X.columns:
            X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)
        if 'Electrical' in X.columns:
            X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0])

        for col in QUAL_COLS:
            if col in X.columns:
                X[col] = X[col].fillna('None').map(QUAL_MAP)

        for col, mapping in ORDINAL_MAPS.items():
            if col in X.columns:
                X[col] = X[col].fillna(list(mapping.keys())[0]).map(mapping)

        return X