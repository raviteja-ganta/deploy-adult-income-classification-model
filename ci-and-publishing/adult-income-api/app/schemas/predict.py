from typing import Any, List, Optional

from pydantic import BaseModel
from classification_model.processing.validation import AdultIncomeInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class AdultIncomeDataInputs(BaseModel):
    inputs: List[AdultIncomeInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Age": 20,
                        "Workclass": "State-gov",
                        "fnlgwt": 77516,
                        "Education": 'Bachelors',
                        "Education_num": "13",
                        "Marital_Status": 'Never-married',
                        "Occupation": "Adm-clerical",
                        "Relationship": "Not-in-family",
                        "Race": "White",
                        "Sex": "Male",
                        "Capital_Gain": 2174,
                        "Capital_Loss": 0,
                        "HoursPerWeek": 40,
                        "Native_country": "United-States",
                    }
                ]
            }
        }
