from setuptools import setup

dependencies = [
      'numpy==1.*',
      'scipy==1.*',
      'tensorflow==1.*',
      'six==1.*',
      'gym[box2d]==0.*',
      'click==8.*',
      'dill==0.*',
      'sortedcontainers==2.*',
      'sortedcollections==1.*',
      'deap==1.*',
      'pyyaml==5.*',
      'pandas==1.*',
      'heartpole==1.*',
      'evestop==0.*',
      'auto-als',
      'bandits @ git+https://github.com/vadim0x60/bandits.git',
      'importlib_metadata==2.*',
      'importlib_resources==5.*'
]

setup(name='cibi',
      version='5.3',
      description='Cibi: lifelong reinforcement learning via program generation and scrum',
      author='Vadim Liventsev',
      author_email='v.liventsev@tue.nl',
      url='https://github.com/vadim0x60/cibi',
      packages=['cibi', 'cibi.codebases'],
      install_requires=dependencies,
      include_package_data=True,
      license='Apache 2.0',
      entry_points='''
            [console_scripts]
            cibi-train=cibi.train:run_experiments_cmd
            cibi-run=cibi.run:run
      ''',
     )