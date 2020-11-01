# %%
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgbm

def load_data(nrows):
    data = pd.read_csv('data/stored/X_test_sample.csv', nrows=nrows)
    return data


def load_model(path):
    lgbm_str = open(path, "r").read()
    model = lgbm.Booster(model_str=lgbm_str)
    return model

st.title('Hotel search ranker webapp')


df = load_data(10000)


# FIXED
search_id = 0
visitor_hist_starrating = np.nan
visitor_hist_adr_usd = np.nan
anomaly = -0.44199505659659266
random_bool = 0
promotion_flag = 0
srch_saturday_night_bool = 0
srch_query_affinity_score = -50
orig_destination_distance = 1000
day_of_week = 4
month = 11


# FLEX
site_id = st.selectbox(
    'site id?',
    np.sort(df['site_id'].unique())
)
visitor_location_country_id = st.selectbox(
    'visitor_location_country id?',
    np.sort(df['visitor_location_country_id'].unique())
)
prop_country_id = st.selectbox(
    'prop_country_id?',
    np.sort(df['prop_country_id'].unique())
)
prop_id = st.selectbox(
    'prop_id ',
    np.sort(df['prop_id'].unique())
)
prop_starrating = st.radio(
    'prop_starrating?',
    np.sort(df['prop_starrating'].unique())
)
prop_review_score = st.radio(
    'prop_review_score',
    np.sort(df['prop_review_score'].unique())
)
prop_brand_bool = st.radio(
    'prop_brand_bool',
    np.sort(df['prop_brand_bool'].unique())
)
prop_location_score1 = st.slider(
    'prop_location_score1',
    min_value=0.0,
    max_value=7.0,
    step=0.5
)
prop_location_score2 = st.slider(
    'prop_location_score2',
    min_value=0.0, 
    max_value=1.0,
    step=0.05
)
prop_log_historical_price = st.slider(
    'prop_log_historical_price',
    min_value=3.0, 
    max_value=6.0,
    step=0.5
)
price_usd = st.slider(
    'price_usd',
    min_value=20,
    max_value=500,
    step=20
)
srch_destination_id = st.selectbox(
    'srch_destination_id',
    np.sort(df['srch_destination_id'].unique())
)
srch_length_of_stay = st.selectbox(
    'srch_length_of_stay',
    np.sort(df['srch_length_of_stay'].unique())
)
srch_booking_window = st.slider(
    'srch_booking_window',
    min_value=1, 
    max_value=90,
    step=1
)
srch_adults_count = st.selectbox(
    'srch_adults_count',
    np.sort(df['srch_adults_count'].unique())
)
srch_children_count = st.selectbox(
    'srch_children_count',
    np.sort(df['srch_children_count'].unique())
)
srch_room_count = st.selectbox(
    'srch_room_count',
    np.sort(df['srch_room_count'].unique())
)

input_list = [
    0, site_id, visitor_location_country_id,
    visitor_hist_starrating, visitor_hist_adr_usd, prop_country_id,
    prop_id, prop_starrating, prop_review_score, prop_brand_bool,
    prop_location_score1, prop_location_score2,
    prop_log_historical_price, price_usd, promotion_flag,
    srch_destination_id, srch_length_of_stay, srch_booking_window,
    srch_adults_count, srch_children_count, srch_room_count,
    srch_saturday_night_bool, srch_query_affinity_score,
    orig_destination_distance, random_bool, day_of_week, month, anomaly
]

# %%
st.title('Model output')

model = load_model(path="model/lgbm.txt")
output = model.predict(pd.DataFrame(input_list).T)
st.write('rank score is: ', output[0])

st.write('local interpretation will be added later ;) an impression of what to expect below')
st.image('plots/SHAP_ex.png')

st.write('Global model Interpretation')
st.image('plots/SHAP.png')


st.title('Model inputs (for validation)')
st.write('Variable inputs')
st.write('You selected site_id: ', site_id)
st.write('You selected visitor_location_country_id: ', visitor_location_country_id)
st.write('You selected country_id: ', prop_country_id)
st.write('You selected prop_id: ', prop_id)
st.write('You selected prop_starrating: ', prop_starrating)
st.write('You selected prop_review_score: ', prop_review_score)
st.write('You selected prop_brand_bool: ', prop_brand_bool)
st.write('You selected prop_location_score1: ', prop_location_score1)
st.write('You selected prop_location_score2: ', prop_location_score2)
st.write('You selected prop_log_historical_price: ', prop_log_historical_price)
st.write('You selected price_usd: ', price_usd)
st.write('You selected srch_destination_id: ', srch_destination_id)
st.write('You selected srch_booking_window: ', srch_booking_window)
st.write('You selected srch_adults_count: ', srch_adults_count)
st.write('You selected srch_children_count: ', srch_children_count)
st.write('You selected srch_room_count: ', srch_room_count)
st.write('You selected srch_query_affinity_score: ', srch_query_affinity_score)
st.write('You selected orig_destination_distance: ', orig_destination_distance)

st.write('----------------------------------------')
st.write('Fixed inputs are set to')
st.write('visitor_hist_starrating: ', visitor_hist_starrating)
st.write('visitor_hist_adr_usd: ', visitor_hist_adr_usd)
st.write('anomaly: ', anomaly)
st.write('random_bool: ', random_bool)
st.write('promotion_flag: ', promotion_flag)
st.write('srch_saturday_night_bool: ', srch_saturday_night_bool)
st.write('srch_query_affinity_score: ', srch_query_affinity_score)
st.write('orig_destination_distance: ', orig_destination_distance)
st.write('day_of_week: ', day_of_week)
st.write('month: ', month)
