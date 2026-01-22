import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

class HousePricePreprocessor:
    def __init__(self):
        self.encoders = {}
        self.cols_ordinal = [
            'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 
            'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 
            'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 
            'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold'
        ]

    def preprocess(self, df: pd.DataFrame, is_training=False):
        df = df.copy()
        
        # 1. Handling Missing Values
        none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
                     'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 
                     'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
        for col in none_cols:
            if col in df.columns: df[col] = df[col].fillna('None')

        zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 
                     'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
        for col in zero_cols:
            if col in df.columns: df[col] = df[col].fillna(0)

        if 'LotFrontage' in df.columns:
            df['LotFrontage'] = df['LotFrontage'].fillna(70.0) 

        mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
        for col in mode_cols:
            if col in df.columns: 
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
        
        if 'Utilities' in df.columns: df = df.drop(['Utilities'], axis=1)

        # 2. Feature Engineering
        if all(c in df.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        
        for col in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:
            if col in df.columns: df[col] = df[col].astype(str)

        # 3. Label Encoding
        for c in self.cols_ordinal:
            if c in df.columns:
                if is_training:
                    lbl = LabelEncoder()
                    lbl.fit(list(df[c].values))
                    self.encoders[c] = lbl
                    df[c] = lbl.transform(list(df[c].values))
                else:
                    if c in self.encoders:
                        lbl = self.encoders[c]
                        # Handle unseen labels with fallback to 0
                        df[c] = df[c].map(lambda s: lbl.transform([s])[0] if s in lbl.classes_ else 0)

        # 4. One Hot Encoding
        df = pd.get_dummies(df)
        
        return df