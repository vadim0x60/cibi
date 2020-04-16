"""
teams.py

All combinations and configurations of developers we have available
When trying out new settings, we recommend adding a new team to this 
file for reproducibility
"""

from cibi.senior_developer import SeniorDeveloper
from cibi.junior_developer import JuniorDeveloper
from cibi.lm import LanguageModel

teams = [
    [SeniorDeveloper({}, LanguageModel)],
    [SeniorDeveloper({}, LanguageModel), JuniorDeveloper()]   
]