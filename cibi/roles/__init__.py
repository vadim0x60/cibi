from cibi.roles import crossover, mutation, generation, delegation, refactoring

LeadDeveloper = delegation.LeadDeveloper
Writer = generation.Writer
Mixer = crossover.Mixer
Editor = mutation.Editor
Nitpicker = refactoring.Nitpicker

#from cibi.lm import LanguageModel

# When trying out new settings, we recommend adding a new team to this 
# file for reproducibility
# teams = [
#     [Writer({}, LanguageModel), Tester()],
#     [Writer({}, LanguageModel), JuniorDeveloper(eps=0), Tester()],
#     [JuniorDeveloper(eps=0), Tester()],
#     [Writer({}, LanguageModel), JuniorDeveloper(), Tester()],
#     [JuniorDeveloper(), Tester()],
#     [Writer({'policy_lstm_sizes': [10]}, LanguageModel, name='senior10'), 
#      Writer({'policy_lstm_sizes': [50]}, LanguageModel, name='senior50'), 
#      Writer({'policy_lstm_sizes': [256]}, LanguageModel, name='senior256'),
#      Writer({'policy_lstm_sizes': [10,10]}, LanguageModel, name='senior10-10'), 
#      Writer({'policy_lstm_sizes': [50,50]}, LanguageModel, name='senior50-50'), 
#      Writer({'policy_lstm_sizes': [256,256]}, LanguageModel, name='senior256-256'),
#      JuniorDeveloper(indpb=1/3, name='junior1by3'),
#      JuniorDeveloper(indpb=1/6, name='junior1by6'),
#      JuniorDeveloper(indpb=1/12, name='junior1by12'),
#      Tester()],
#     [Writer({'policy_lstm_sizes': [10]}, LanguageModel, name='senior10'), 
#      Writer({'policy_lstm_sizes': [50]}, LanguageModel, name='senior50'), 
#      Writer({'policy_lstm_sizes': [256]}, LanguageModel, name='senior256'),
#      Writer({'policy_lstm_sizes': [10,10]}, LanguageModel, name='senior10-10'), 
#      Writer({'policy_lstm_sizes': [50,50]}, LanguageModel, name='senior50-50'), 
#      Writer({'policy_lstm_sizes': [256,256]}, LanguageModel, name='senior256-256'),
#      JuniorDeveloper(indpb=1/3, name='junior1by3'),
#      JuniorDeveloper(indpb=1/6, name='junior1by6'),
#      JuniorDeveloper(indpb=1/12, name='junior1by12')]
# ]