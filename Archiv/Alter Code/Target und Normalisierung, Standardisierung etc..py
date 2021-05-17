#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Normalisierung und Logarithmierung
X = df_zimmergröße_breitengrad.drop(columns=["angebotspreis"])
y = df_zimmergröße_breitengrad['angebotspreis']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size=0.2)
train_X

catCols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object']
catCols
train_X_cat = train_X[catCols].copy()
val_X_cat = val_X[catCols].copy()

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
simple_imputer = SimpleImputer(strategy='most_frequent')

target_encoder = TargetEncoder()

train_X_targetenc = target_encoder.fit_transform(train_X_cat, train_y)
val_X_targetenc = target_encoder.transform(val_X_cat)
train_X_labelenc

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return np.sqrt(metrics.mean_squared_error(y_valid, preds))
simple_imputer = SimpleImputer()
numCols = [cname for cname in train_X.columns if train_X[cname].dtype != "object"]
numCols
train_X_num = pd.DataFrame(simple_imputer.fit_transform(train_X[numCols]), columns=numCols, index=train_X.index)
val_X_num = pd.DataFrame(simple_imputer.transform(val_X[numCols]), columns=numCols, index=val_X.index)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_X_num_normalized = pd.DataFrame(scaler.fit_transform(train_X_num), 
                                      columns=train_X_num.columns, index=train_X_num.index)
val_X_num_normalized = pd.DataFrame(scaler.transform(val_X_num), 
                                    columns=train_X_num.columns, index=val_X_num.index)
train_X_logGains = train_X_num_normalized.copy()
val_X_logGains = val_X_num_normalized.copy()

train_X_logGains['loggrundst'] = np.log1p(train_X_logGains['grundstuecksflaeche'])
val_X_logGains['loggrundst'] = np.log1p(val_X_logGains['grundstuecksflaeche'])

train_X_target_num = train_X_logGains.join(train_X_targetenc.add_suffix("_targetenc"))
val_X_target_num = val_X_logGains.join(val_X_targetenc.add_suffix("_targetenc"))
print("RSME: {}".
      format(score_dataset(train_X_target_num, val_X_target_num, train_y, val_y)))


# In[ ]:


#Normalisierung
X = df_zimmergröße_breitengrad.drop(columns=["angebotspreis"])
y = df_zimmergröße_breitengrad['angebotspreis']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size=0.2)
train_X

catCols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object']
catCols
train_X_cat = train_X[catCols].copy()
val_X_cat = val_X[catCols].copy()

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
simple_imputer = SimpleImputer(strategy='most_frequent')

target_encoder = TargetEncoder()

train_X_targetenc = target_encoder.fit_transform(train_X_cat, train_y)
val_X_targetenc = target_encoder.transform(val_X_cat)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return np.sqrt(metrics.mean_squared_error(y_valid, preds))
simple_imputer = SimpleImputer()
numCols = [cname for cname in train_X.columns if train_X[cname].dtype != "object"]
numCols
train_X_num = pd.DataFrame(simple_imputer.fit_transform(train_X[numCols]), columns=numCols, index=train_X.index)
val_X_num = pd.DataFrame(simple_imputer.transform(val_X[numCols]), columns=numCols, index=val_X.index)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_X_num_normalized = pd.DataFrame(scaler.fit_transform(train_X_num), 
                                      columns=train_X_num.columns, index=train_X_num.index)
val_X_num_normalized = pd.DataFrame(scaler.transform(val_X_num), 
                                    columns=train_X_num.columns, index=val_X_num.index)
train_X_target_num = train_X_num_normalized.join(train_X_targetenc.add_suffix("_targetenc"))
val_X_target_num = val_X_num_normalized.join(val_X_targetenc.add_suffix("_targetenc"))
print("RSME: {}".
      format(score_dataset(train_X_target_num, val_X_target_num, train_y, val_y)))


# In[ ]:


#Standardisierung und Logarithmierung
X = df_zimmergröße_breitengrad.drop(columns=["angebotspreis"])
y = df_zimmergröße_breitengrad['angebotspreis']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size=0.2)
train_X

catCols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object']
catCols
train_X_cat = train_X[catCols].copy()
val_X_cat = val_X[catCols].copy()

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
simple_imputer = SimpleImputer(strategy='most_frequent')

target_encoder = TargetEncoder()

train_X_targetenc = target_encoder.fit_transform(train_X_cat, train_y)
val_X_targetenc = target_encoder.transform(val_X_cat)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return np.sqrt(metrics.mean_squared_error(y_valid, preds))
simple_imputer = SimpleImputer()
numCols = [cname for cname in train_X.columns if train_X[cname].dtype != "object"]
numCols
train_X_num = pd.DataFrame(simple_imputer.fit_transform(train_X[numCols]), columns=numCols, index=train_X.index)
val_X_num = pd.DataFrame(simple_imputer.transform(val_X[numCols]), columns=numCols, index=val_X.index)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_X_num_standardized = pd.DataFrame(scaler.fit_transform(train_X_num), 
                                        columns=train_X_num.columns, index=train_X_num.index)
val_X_num_standardized = pd.DataFrame(scaler.transform(val_X_num), 
                                      columns=train_X_num.columns, index=val_X_num.index)

train_X_logGains = train_X_num_standardized.copy()
val_X_logGains = val_X_num_standardized.copy()

train_X_logGains['loggrundst'] = np.log1p(train_X_logGains['grundstuecksflaeche'])
val_X_logGains['loggrundst'] = np.log1p(val_X_logGains['grundstuecksflaeche'])

train_X_target_num = train_X_logGains.join(train_X_targetenc.add_suffix("_targetenc"))
val_X_target_num = val_X_logGains.join(val_X_targetenc.add_suffix("_targetenc"))
print("RSME: {}".
      format(score_dataset(train_X_target_num, val_X_target_num, train_y, val_y)))


# In[ ]:


#Standardisierung

X = df_zimmergröße_breitengrad.drop(columns=["angebotspreis"])
y = df_zimmergröße_breitengrad['angebotspreis']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size=0.2)
train_X

catCols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object']
catCols
train_X_cat = train_X[catCols].copy()
val_X_cat = val_X[catCols].copy()

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
simple_imputer = SimpleImputer(strategy='most_frequent')

target_encoder = TargetEncoder()

train_X_targetenc = target_encoder.fit_transform(train_X_cat, train_y)
val_X_targetenc = target_encoder.transform(val_X_cat)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return np.sqrt(metrics.mean_squared_error(y_valid, preds))
simple_imputer = SimpleImputer()
numCols = [cname for cname in train_X.columns if train_X[cname].dtype != "object"]
numCols
train_X_num = pd.DataFrame(simple_imputer.fit_transform(train_X[numCols]), columns=numCols, index=train_X.index)
val_X_num = pd.DataFrame(simple_imputer.transform(val_X[numCols]), columns=numCols, index=val_X.index)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_X_num_standardized = pd.DataFrame(scaler.fit_transform(train_X_num), 
                                        columns=train_X_num.columns, index=train_X_num.index)
val_X_num_standardized = pd.DataFrame(scaler.transform(val_X_num), 
                                      columns=train_X_num.columns, index=val_X_num.index)
train_X_target_num = train_X_num_standardized.join(train_X_targetenc.add_suffix("_targetenc"))
val_X_target_num = val_X_num_standardized.join(val_X_targetenc.add_suffix("_targetenc"))
print("RSME: {}".
      format(score_dataset(train_X_target_num, val_X_target_num, train_y, val_y)))


# In[ ]:


#Logarithmierung
X = df_zimmergröße_breitengrad.drop(columns=["angebotspreis"])
y = df_zimmergröße_breitengrad['angebotspreis']
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0, test_size=0.2)
train_X

catCols = [cname for cname in train_X.columns if train_X[cname].dtype == 'object']
catCols
train_X_cat = train_X[catCols].copy()
val_X_cat = val_X[catCols].copy()

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
simple_imputer = SimpleImputer(strategy='most_frequent')

target_encoder = TargetEncoder()

train_X_targetenc = target_encoder.fit_transform(train_X_cat, train_y)
val_X_targetenc = target_encoder.transform(val_X_cat)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = GradientBoostingRegressor(max_depth=4,
                                     subsample=0.9,
                                     max_features=0.75,
                                     n_estimators=1000,
                                     random_state=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return np.sqrt(metrics.mean_squared_error(y_valid, preds))
simple_imputer = SimpleImputer()
numCols = [cname for cname in train_X.columns if train_X[cname].dtype != "object"]
numCols
train_X_num = pd.DataFrame(simple_imputer.fit_transform(train_X[numCols]), columns=numCols, index=train_X.index)
val_X_num = pd.DataFrame(simple_imputer.transform(val_X[numCols]), columns=numCols, index=val_X.index)

train_X_logGains = train_X_num.copy()
val_X_logGains = val_X_num.copy()

train_X_logGains['loggrundst'] = np.log1p(train_X_logGains['grundstuecksflaeche'])
val_X_logGains['loggrundst'] = np.log1p(val_X_logGains['grundstuecksflaeche'])
train_X_target_num = train_X_logGains.join(train_X_targetenc.add_suffix("_targetenc"))
val_X_target_num = val_X_logGains.join(val_X_targetenc.add_suffix("_targetenc"))
print("RSME: {}".
      format(score_dataset(train_X_target_num, val_X_target_num, train_y, val_y)))

