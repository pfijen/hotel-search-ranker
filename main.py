import logging

from src.modelling import Recommender
from src.processing import Processor

# Configurate logger format
logging.basicConfig(
    format="[%(levelname)s %(asctime)s] - %(message)s", level=logging.INFO
)

if __name__=='__main__':

    logger = logging.getLogger()
    data = Processor(logger)

    # Read data and apply basic processing
    df = data.read_data("data/input/bookings.csv")
    df_processed = data.process(df)

    # Split data in 3 sets (70/20/10%) and store test sample
    df_train, df_val, df_test = data.split_data(df_processed)
    data.save_data(df_test, path="data/stored/test_sample.csv", nrows=10000)

    X_train, y_train = data.split_Xy(df_train)
    X_val, y_val = data.split_Xy(df_val)
    X_test, y_test = data.split_Xy(df_test)

    model = Recommender(logger)

    # SKlearn isolation forest only works with non-missing numerical values
    # retrieve these columns and generate anomaly scores for the 3 datasets
    on_columns = data.numeric_and_complete_cols(df_processed)
    X_train, X_val, X_test = model.anomaly_score(
        on_columns, X_train, X_val, X_test
    )

    # Transform X data to LightGBM Dataset format
    train_data = model.pandas_to_lgbm(X_train, y_train)
    val_data = model.pandas_to_lgbm(X_val, y_val)

    # Train model with a train set
    # Use validation set for early stopping
    model.fit(train_data, val_data)

    # Store the model for later use
    # model.load(path="model/lgbm.txt")

    # Score test set and calculate metric
    X_test_predicted = model.predict(X_test)
    model.score(X_test_predicted, y_test)

    # Use SHAP to explain our gradient boosting tree
    # NOTE: if the stored model is loaded the explainer
    # does not work, to be fixed. (probably since model object is missing)
    model.explain(X_test, nrows=10000)
    model.plot_shap(shap_values_path='model/shap_values.npy', X=X_test)

    logger.info("Program terminated")
