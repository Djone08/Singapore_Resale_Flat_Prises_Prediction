import pandas as pd
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime as dt
import pickle
import json


def ohe(_df: pd.DataFrame):
    _indices = {}

    _df.block, _indices['block'] = _df.block.factorize(sort=True)
    _df.flat_type, _indices['flat_type'] = _df.flat_type.factorize(sort=True)
    _df.flat_model, _indices['flat_model'] = _df.flat_model.factorize(sort=True)
    _df.street_name, _indices['street_name'] = _df.street_name.factorize(sort=True)
    _df.town, _indices['town'] = _df.town.factorize(sort=True)
    _df.storey_range, _indices['storey_range'] = _df.storey_range.factorize(sort=True)
    return _df, _indices


def data_collect():
    file_paths = []

    for path, _, files in os.walk('Resale Flat Prices'):
        for f in files:
            if f.endswith('.csv'):
                file_paths.append(fr'{path}\{f}')

    _df = pd.concat([pd.read_csv(p) for p in file_paths])
    _df.drop_duplicates(inplace=True)
    _df.reset_index(drop=True, inplace=True)

    # _df.flat_type.replace('MULTI-GENERATION', 'MULTI GENERATION', inplace=True)
    _df.replace({'flat_type': 'MULTI-GENERATION'},
               {'flat_type': 'MULTI GENERATION'}, inplace=True)
    _df.flat_model = _df.flat_model.str.upper()
    _df.flat_type = _df.flat_type.str.upper()

    _df.loc[:, 'reg_year'] = _df.month.str.split('-').str[0]
    _df.loc[:, 'reg_month'] = _df.month.str.split('-').str[1]
    _df.drop(columns=['month', 'remaining_lease'], inplace=True)
    return _df


def save_model(_model, _indices):
    filename = 'decision_tree_regressor.pkl'
    pickle.dump(_model, open(filename, 'wb'))

    indices_dict = {_x: list(_indices[_x]) for _x in _indices}
    with open('Encoded_Column_Data.json', 'w') as f:
        json.dump(indices_dict, f, indent=4)
    return indices_dict


def run_model():
    _df, _indices = ohe(data_collect())

    x = _df.drop('resale_price', axis=1)
    y = _df.resale_price

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)

    _model = DecisionTreeRegressor()
    _model.fit(x_train, y_train)
    # y_pred = _model.predict(x_test)

    indices = save_model(_model, _indices)
    st.session_state['accuracy'] = f'{_model.score(x_test, y_test):.0%}'
    return _model, indices


def get_model() -> tuple:
    with open('decision_tree_regressor.pkl', 'rb') as _f:
        _model = pickle.load(_f)

    with open(r'Encoded_Column_Data.json', 'r') as _f:
        _indices = json.load(_f)

    return _model, _indices


if __name__ == '__main__':
    model, ids = get_model()
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    if st.button('**Re-Fresh Model**'):
        model, ids = run_model()
    st.markdown("### :orange[Predicting Resale Price (Regression Task)]")
    if st.session_state.get('accuracy'):
        st.markdown('### :green[Model Accuracy: %(accuracy)s]' % st.session_state)

    col1, col2 = st.columns(2, gap='large')

    with col1:
        town = st.selectbox('Select the **Town**', ids['town'])
        flat_type = st.selectbox('Select the **Flat Type**', ids['flat_type'])
        block = st.selectbox('Select the **Block**', ids['block'])
        storey_range = st.selectbox('Select the **Storey Range**', ids['storey_range'])
        street_name = st.selectbox('Select the **Street Name**', ids['street_name'])

    with col2:
        floor_area_sqm = st.number_input('Select the **floor_area_sqm**',
                                         value=60.0, min_value=28.0, max_value=173.0, step=1.0)
        flat_model = st.selectbox('Select the **flat_model**', ids['flat_model'])
        lease_commence_date = st.number_input('Enter the **Lease Commence Year**',
                                              min_value=1966, max_value=2022, value=2017)
        reg_year = st.number_input("Select the **Registration Year** which you want",
                                   min_value=1990, max_value=2024, value=dt.now().year)
        reg_month = st.number_input("Select the **Registration Month** which you want",
                                    min_value=1, max_value=12, value=dt.now().month)

    st.markdown('Click below button to predict the **Flat Resale Price**')

    # Prediction logic
    test_data = [
        ids['town'].index(town),
        ids['flat_type'].index(flat_type),
        ids['block'].index(block),
        ids['street_name'].index(street_name),
        ids['storey_range'].index(storey_range),
        floor_area_sqm,
        ids['flat_model'].index(flat_model),
        lease_commence_date,
        reg_year,
        reg_month,
    ]

    if st.button('**Predict**'):
        predicted_price = model.predict(pd.DataFrame([test_data]))
        st.markdown(f"### :blue[Flat Resale Price is] :green[${round(predicted_price[0], -3):.0f}]")
