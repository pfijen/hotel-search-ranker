# %%
import logging

from src.modelling import Recommender
from src.processing import Processor

logging.basicConfig(
    format="[%(levelname)s %(asctime)s] - %(message)s", level=logging.INFO
)

# %%
logger = logging.getLogger()
data = Processor(logger)

df = data.read_data("data/input/bookings.csv")
df_processed = data.process(df)
df_train, df_val, df_test = data.split_data(df_processed)
data.save_data(df_test, path="data/stored/test_sample.csv", nrows=10000)

X_train, y_train = data.split_Xy(df_train)
X_val, y_val = data.split_Xy(df_val)
X_test, y_test = data.split_Xy(df_test)


# %%
model = Recommender(logger)

on_columns = data.numeric_and_complete_cols(df_processed)
X_train, X_val, X_test = model.anomaly_score(
    on_columns, X_train, X_val, X_test
)

train_data = model.pandas_to_lgbm(X_train, y_train)
val_data = model.pandas_to_lgbm(X_val, y_val)

model.fit(train_data, val_data)
#model.load(path="model/lgbm.txt")

X_test_predicted = model.predict(X_test)
model.score(X_test_predicted, y_test)

model.explain(X_test, nrows=10000)

model.plot_shap(shap_values_path='model/shap_values.npy', X=X_test)
logger.info("Program terminated")

# %%
