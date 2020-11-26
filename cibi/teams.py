"""
teams.py

All combinations and configurations of developers we have available
When trying out new settings, we recommend adding a new team to this 
file for reproducibility
"""

from cibi.senior_developer import SeniorDeveloper
from cibi.junior_developer import JuniorDeveloper
from cibi.tester import Tester
from cibi.lm import LanguageModel

teams = [
    [SeniorDeveloper({}, LanguageModel), Tester()],
    [SeniorDeveloper({}, LanguageModel), JuniorDeveloper(eps=0), Tester()],
    [JuniorDeveloper(eps=0), Tester()],
    [SeniorDeveloper({}, LanguageModel), JuniorDeveloper(), Tester()],
    [JuniorDeveloper(), Tester()],
    [SeniorDeveloper({'policy_lstm_sizes': [10]}, LanguageModel, name='senior10'), 
     SeniorDeveloper({'policy_lstm_sizes': [50]}, LanguageModel, name='senior50'), 
     SeniorDeveloper({'policy_lstm_sizes': [256]}, LanguageModel, name='senior256'),
     SeniorDeveloper({'policy_lstm_sizes': [10,10]}, LanguageModel, name='senior10-10'), 
     SeniorDeveloper({'policy_lstm_sizes': [50,50]}, LanguageModel, name='senior50-50'), 
     SeniorDeveloper({'policy_lstm_sizes': [256,256]}, LanguageModel, name='senior256-256'),
     JuniorDeveloper(indpb=1/3, name='junior1by3'),
     JuniorDeveloper(indpb=1/6, name='junior1by6'),
     JuniorDeveloper(indpb=1/12, name='junior1by12'),
     Tester()],
    [SeniorDeveloper({'policy_lstm_sizes': [10]}, LanguageModel, name='senior10'), 
     SeniorDeveloper({'policy_lstm_sizes': [50]}, LanguageModel, name='senior50'), 
     SeniorDeveloper({'policy_lstm_sizes': [256]}, LanguageModel, name='senior256'),
     SeniorDeveloper({'policy_lstm_sizes': [10,10]}, LanguageModel, name='senior10-10'), 
     SeniorDeveloper({'policy_lstm_sizes': [50,50]}, LanguageModel, name='senior50-50'), 
     SeniorDeveloper({'policy_lstm_sizes': [256,256]}, LanguageModel, name='senior256-256'),
     JuniorDeveloper(indpb=1/3, name='junior1by3'),
     JuniorDeveloper(indpb=1/6, name='junior1by6'),
     JuniorDeveloper(indpb=1/12, name='junior1by12')]
]