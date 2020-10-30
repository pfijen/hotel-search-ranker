import lightgbm as lgbm
import numpy as np
import shap
from IPython.display import display
from lightgbm.basic import Dataset
from pandas import DataFrame, Series


class Recommender(object):
    """Class with all modelling related functions"""

    def __init__(self, logger) -> None:
        self.logger = logger
        self.model: lgbm.Booster
        self.parameters = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": 5,
            "boosting": "gbdt",
            "num_threads": 4,
            "max_leaves": 50,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "feature_fraction": 0.95,
            "learning_rate": 0.15,
            "verbose": 1,
            "min_data_in_leaf": 25,
            "ignore_column": 0,
            "label_gain": [0, 1, 2, 3, 4, 5],
            "zero_as_missing": True,
            "seed": 2,
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
        """Train the model with class parameters"""
        self.model = lgbm.train(
            params=self.parameters,
            train_set=lgbm_train,
            valid_sets=lgbm_val,
            num_boost_round=100,
            early_stopping_rounds=10,
        )
        display(lgbm.plot_metric(self.model))
        lgbm.plot_metric(self.model)
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

    def explain(self, X: DataFrame) -> None:
        """SHAP is an innovative method to explain complex ML models
        Works on a global level like in this case
        As well as for individual predictions
        """
        self.logger.info("Creating global model explainability plot")
        explainer = shap.TreeExplainer(self.model)
        self._shap_values = explainer.shap_values(X.values)
        shap.summary_plot(
            self._shap_values,
            X,
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
