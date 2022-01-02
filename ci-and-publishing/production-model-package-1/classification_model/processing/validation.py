from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.categorical_vars_with_na_frequent
        + config.model_config.categorical_vars_with_na_missing
        and validated_data[var].isnull().sum() > 0
    ]
    # Here we are dropping those rows when na in new_vars_with_na column
    # this logic can be changed depending on our project
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # Rename some column names
    input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        AdultIncomeDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


# This is pydantic schema
# what we expect the type to be
class AdultIncomeInputSchema(BaseModel):
    Workclass: Optional[str]
    Education: Optional[str]
    Marital_Status: Optional[str]
    Occupation: Optional[str]
    Relationship: Optional[str]
    Race: Optional[str]
    Sex: Optional[str]
    Native_country: Optional[str]
    Age: Optional[int]
    fnlgwt: Optional[int]
    Capital_Gain: Optional[int]
    Capital_Loss: Optional[int]
    HoursPerWeek: Optional[int]  # renamed
    Education_num: Optional[int]


class AdultIncomeDataInputs(BaseModel):
    inputs: List[AdultIncomeInputSchema]
