# %%
import lightgbm as lgbm
import numpy as np
import shap
from lightgbm.basic import Dataset
from pandas import DataFrame, Series


class Recommender(object):
    """Class with all modelling related functions"""

    def __init__(self, logger) -> None:
        self.logger = logger
        self.parameters = {
            # fixed params
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": 5,
            "boosting": "gbdt",
            "num_threads": 2,
            "force_row_wise": True,
            "zero_as_missing": True,
            "seed": 2,
            "label_gain": [0, 1, 5],  # weights for ndcg metric
            "verbose": 1,
            "ignore_column": 0,  # column 0 specifies the search_id
            # hyperparams below
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "feature_fraction": 0.95,
            "learning_rate": 0.15,
            "min_data_in_leaf": 25,
        }
        pass

    def pandas_to_lgbm(self, X: DataFrame, y: Series) -> Dataset:
        """Transforms the X, y from pandas format to LightGBM
        LightGBM requires Dataset format as input
        """
        lgbm_dataset = lgbm.Dataset(
            X,
            label=y,
            categorical_feature=[0, 1, 2, 5, 6, 15],
            group=list(X.groupby("srch_id").count()["site_id"]),
        )
        return lgbm_dataset

    def fit(self, lgbm_train: Dataset, lgbm_val: Dataset) -> None:
        """Train the model with class parameters and store model"""
        self.model = lgbm.train(
            params=self.parameters,
            train_set=lgbm_train,
            valid_sets=lgbm_val,
            num_boost_round=20,
            early_stopping_rounds=10,
        )
        
        text_file = open("model/lgbm.txt", "w")
        text_file.write(self.model.model_to_string())
        text_file.close()
        self.logger.info("Saved model to model/ dir")
        pass

    def load(self, path: str) -> None:
        """Load pre trained model from model dir"""
        lgbm_str = open(path, 'r').read()
        self.model = lgbm.Booster(model_str=lgbm_str)
        pass

    def predict(self, df: DataFrame) -> DataFrame:
        """Create predictions with LightGBM
        Save predictions into df under name 'y_hat'
        y_hat is the estimated relative ranking score
        """
        self.logger.info("Making predictions on the test set")
        predictions = self.model.predict(df)
        df = df.assign(y_hat=predictions)
        return df

    def score(self, X: DataFrame, y: Series) -> None:
        """Calculate the raking score on the provided dataframe
        Loop over all unique searches, calculate the ndcg
        per search, final score is the average ndcg@5
        """
        self.logger.info("Calculating the NDCG@5 of test set")
        df = X.join(y)

        score = []
        ids = df.srch_id.unique()
        for ID in ids:
            df_id = df[df["srch_id"] == ID].sort_values("y_hat", ascending=False)
            score.append(self._ndcg_at_k(list(df_id["r"]), k=5))
        self.logger.info(f"NDCG@5 performance on test set {np.mean(score)}")
        pass

    def explain(self, X: DataFrame, nrows: int) -> None:
        """SHAP is an innovative method to explain complex ML models
        Works on a global model level like in this case
        SHAP also works for explaining individual predictions
        """
        self.logger.info("Calculating SHAP values for explainibility")
        explainer = shap.TreeExplainer(self.model)
        self._shap_values = explainer.shap_values(X[:nrows].values)
        np.save('model/shap_values.npy', self._shap_values)
        pass

    def plot_shap(self, shap_values_path: str, X: DataFrame, nrows=10000):
        self.logger.info("Creating global model explainability plot")
        self._shap_values = np.load(shap_values_path)
        shap.summary_plot(
            shap_values=self._shap_values,
            features=X[:nrows],
        )
        pass

    def _dcg_at_k(self, r: list, k: int) -> float:
        """dcg = discounted cumulative gain"""
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(
                np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2))
            )
        return 0.0

    def _ndcg_at_k(self, r: list, k: int) -> float:
        """ndcg is the noralized version of dcg"""
        idcg = self._dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 1.0
        return self._dcg_at_k(r, k) / idcg


# %%
