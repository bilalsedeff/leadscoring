from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessor(num_features=None, cat_features=None, target_encoder=False, remainder='drop'):
    """
    Sayısal ve kategorik özellikleri işleyen bir pipeline oluşturur.
    
    Args:
        num_features: Sayısal özellik listesi
        cat_features: Kategorik özellik listesi
        target_encoder: Hedef kodlama kullanılacak mı?
        remainder: Kolonlara uygulanmayan sütunlar için işlem ('drop', 'passthrough')
        
    Returns:
        ColumnTransformer: Önişleme pipeline'ı
    """
    transformers = []
    
    # Sayısal özellikler
    if num_features and len(num_features) > 0:
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        transformers.append(('num', num_transformer, num_features))
    
    # Kategorik özellikler
    if cat_features and len(cat_features) > 0:
        if target_encoder:
            # Müşteri segmentlerinde hedef kodlama kullanmak için
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', TargetEncoder())
            ])
        else:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        transformers.append(('cat', cat_transformer, cat_features))
    
    # ID kolonlarını otomatik düşürmek için remainder='drop'
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder  # 'drop' or 'passthrough'
    )
    
    return preprocessor
