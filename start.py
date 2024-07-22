import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle
import json

file_paths = []

for path, _, files in os.walk('Resale Flat Prices'):
    for f in files:
        file_paths.append(fr'{path}\{f}') if f.endswith('.csv') else None

df = pd.concat([pd.read_csv(p) for p in file_paths])
df.reset_index(drop=True, inplace=True)

# df.flat_type.replace('MULTI-GENERATION', 'MULTI GENERATION', inplace=True)
df.replace({'flat_type': 'MULTI-GENERATION'},
           {'flat_type': 'MULTI GENERATION'}, inplace=True)

df.loc[:, 'reg_year'] = df.month.str.split('-').str[0]
df.loc[:, 'reg_month'] = df.month.str.split('-').str[1]

indices = {}

df.block, indices['block'] = df.block.factorize(sort=True)
df.flat_type, indices['flat_type'] = df.flat_type.factorize(sort=True)
df.flat_model, indices['flat_model'] = df.flat_model.factorize(sort=True)
df.street_name, indices['street_name'] = df.street_name.factorize(sort=True)
df.town, indices['town'] = df.town.factorize(sort=True)
df.storey_range, indices['storey_range'] = df.storey_range.factorize(sort=True)

x = df.drop('resale_price', axis=1)
y = df.resale_price

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)

decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(x_train, y_train)
y_pred = decision_tree_regressor.predict(x_test)

filename = 'decision_tree_regressor.pkl'
pickle.dump(decision_tree_regressor, open(filename, 'wb'))

indices_dict = {x: pd.Series(indices[x]).to_dict() for x in indices}
with open('Encoded_Column_Data.json', 'w') as f:
    json.dump(indices_dict, f, indent=4)
