# %%
import logging

from src.modelling import Recommender
from src.processing import Processor

logging.basicConfig(
    format="[%(levelname)s %(asctime)s] - %(message)s", level=logging.INFO
)

if __name__ == "__main__":
    logger = logging.getLogger()
    data = Processor(logger)
    df = data.read_data("data/input/bookings.csv")

    df_processed = data.process(df)
    df_train, df_val, df_test = data.split_data(df_processed)

    model = Recommender(logger)

    X_train, y_train = data.split_Xy(df_train)
    train_data = model.pandas_to_lgbm(X_train, y_train)

    X_val, y_val = data.split_Xy(df_val)
    val_data = model.pandas_to_lgbm(X_val, y_val)

    model.fit(train_data, val_data)

    X_test, y_test = data.split_Xy(df_test)
    X_test = model.predict(X_test)
    model.score(X_test, y_test)

    model.explain(X_test.drop(columns="y_hat"))
    pass

# %%
