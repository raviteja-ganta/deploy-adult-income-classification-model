# from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, RareLabelEncoder
# from feature_engine.imputation import (
#     AddMissingIndicator,
#     CategoricalImputer,
#     MeanMedianImputer,
# )
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

from feature_engine.imputation import (
    CategoricalImputer,
)
from feature_engine.selection import DropCorrelatedFeatures
# from feature_engine.wrappers import SklearnTransformerWrapper
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from classification_model.config.core import config
from classification_model.processing import features as pp

class_pipeline = Pipeline(
    [
        # ===== Data Cleaning =====
        # Remove lead or trail space
        (
            "remove_lead_trail_space",
            pp.RemoveLeadingTrailSpace(variables=config.model_config.categorical_vars),
        ),
        # ===== Data Cleaning =====
        # impute categorical variables with string missing
        (
            "categorical_clean",
            pp.CategoricalClean(variables=config.model_config.categorical_vars),
        ),
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars_with_na_missing,
            ),
        ),
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.model_config.categorical_vars_with_na_frequent,
            ),
        ),
        # == CATEGORICAL ENCODING
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.05, n_categories=1, variables=config.model_config.categorical_vars
            ),
        ),
        # encode categorical and discrete variables using the target mean
        (
            "categorical_encoder",
            OneHotEncoder(
                top_categories=None,
                drop_last=True,
                variables=config.model_config.categorical_vars,
            ),
        ),
        ("scaler", StandardScaler()),
        # make dataframe from the input of previous step
        (
            "makedataframe",
            pp.MakePandasDataFrame(config.model_config.columns_after_one_hot),
        ),
        (
            "variable_selection_correlation",
            DropCorrelatedFeatures(
                variables=None,
                method="pearson",
                threshold=config.model_config.drop_threshold,
            ),
        ),
        (
            "Adaboost model",
            AdaBoostClassifier(
                DecisionTreeClassifier(
                    random_state=config.model_config.random_state,
                    max_depth=config.model_config.max_depth,
                ),
                n_estimators=config.model_config.n_estimators,
                learning_rate=config.model_config.learning_rate,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
