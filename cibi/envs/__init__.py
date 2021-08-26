import gym

gym.register(id='Taxi-v3-mod-v0',
             entry_point='cibi.envs.taxi_wrapper:TaxiEnvMultiDiscrete',
             nondeterministic=False)

# Optional healthcare packages
try:  
    import auto_als
except ImportError:
    pass

try:
    from heartpole import HeartPole
except ImportError:
    pass