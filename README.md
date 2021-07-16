# Cibi

[Cibi](https://en.wikipedia.org/wiki/Cibi) is a war dance that Fijian rugby team performs before it gets into a [scrum](https://en.wikipedia.org/wiki/Scrum_(rugby)) and defeats its opponents.

This package doesn't perform any war dances (yet), but it can defeat most [OpenAI Gym](https://gym.openai.com/) environments using [neurogenetic programming](https://arxiv.org/abs/2102.04231) and [BF++](https://arxiv.org/abs/2101.09571): an extended version of [Brainfuck language](https://en.wikipedia.org/wiki/Brainfuck)

Some code is blatantly copied and pasted from [Google's Brain Coder](https://github.com/tensorflow/models/tree/master/research/brain_coder) which is permitted by [Apache License](LICENSE)

This package has 2 applications:
- Run BF++ programs in a reinforcement learning environment
- Synthesize programs that (hopefully) maximize rewards of the agent in a reinforcement learning environment

This is done with `cibi/run.py` and `cibi/train.py` respectively

## Requirements

The main prerequisite is Python 3.0-3.7 (we use a version of Tensorsflow incompatible with Python 3.8 and above, feel free to rewrite everything in modern Tensorsflow and pull request)
Before using `run.py` and `train.py` commands, install the requirements and `cibi` itself:

```
pip3 install -r requirements.txt
pip3 install -e .
```

## Environments

Cibi supports any valid [OpenAI Gym](https://gym.openai.com) environment (it is, however, unlikely to perform well on environments with high-dimensional action/observation spaces like Atari games - it will have to write the entire screen pixel-by-pixel onto the memory tape).

For environments listed in [the registry](https://github.com/openai/gym/wiki/Table-of-environments) no download or installation is required. Custom environments can be added in `cibi/extensions.py`. Then you just need to specify the environment name in the form `CartPole-v0`

## Running programs

You can define the program as an argument

`cibi-run CartPole-v1 @!`

Or a run a codebase file with one or several newline-separated programs

`cibi-run CartPole-v1 -i programs.txt`

The program will be run once unless `--avg` and/or `--best` options are specified. All options:

```
Usage: run.py [OPTIONS] ENV_NAME [INPUT_CODE]

Options:
  --avg INTEGER                   Average results over multiple runs
  --best INTEGER                  Take the best result over multiple runs
  --force-fluid-discretization    Use fluid discretization even if an
                                  observation has lower and upper bounds
                                  defined
  --fluid-discretization-history INTEGER
                                  Length of the observation history kept for
                                  fluid discretization
  -i, --input-file TEXT           Load the programs from a specified file
  -o, --output-file TEXT          Save total rewards to a file
  --debug                         Show a visual rendering of the env and log
                                  full execution traces
  --help                          Show this message and exit.
```

## Training

To train a program synthesis model and generate programs for an environment, create a directory for the results and define an `experiment.yml` YAML configuration file in this directory. Minimal required configuration is

```
env: MountainCarContinuous-v0
team: 0
```

"Team 0" means generating programs with one LSTM. In the future, more configurations will be available including genetic programming. 

An example that specifies everything that can be specified: 

```
env: HeartPole
cibi-version: 4
max-sprints: 1000000
max-sprints-without-improvement: 10000
team: 0
allowed-commands: "@><^+-[].,!~01234"
seed: heartpole.txt
sprint_length: 100
stretch_sprints: false
max-episode-length: 10000
```

`max-sprints` sets the absolute number of sprints after which training will end. In this case, no more than 4000000 (`max-sprints`) programs will be written since our batch size is 4. `max-sprints-without-improvement` lets you set early stopping

`allowed-commands` lets you use a subset of BF++ instead of the full language

`seed` let's you use programs you already have to jumpstart the training (the seed file has to be in `EXPERIMENT_DIR` or in `codebases`)

`sprint_length` controls how often the programs are swapped out during training, `stretch_sprints` make sure that the program isn't swapped out before the episode ends. It's important that `stretch_sprints` is `true` for environments where reward is given at the end. It is `true` by default

`HeartPole` is an endless environment (it never returns `done=True`) so we specify `max-episode-length` manually.
If we did not specify this, all sprints would be done within one reinforcement learning episode.

Then run

```
cibi-train EXPERIMENT_DIR
```

If the training process was killed (intentionally or not), it can be resumed with the same command.

After the training finishes, there will be several files in `EXPERIMENT_DIR`, the most important one being `top.pickle` containing 256 best programs. It is a pandas dataframe to be loaded with `pandas.read_pickle()` with programs and some metadata including their `test_quality` - total episode reward averaged over 100 episodes. `programs.pickle` contains all programs written in the process of getting to the best ones, `log.log` is what you'd expect from a log file, `train` folder contains model checkpoints, and `summary.yml` is a short summary of experiment status:

```
cibi-version: '4.0'
longest-episode: 42
max-total-reward: 42.0
seconds-elapsed: 16467.135838000104
shortest-episode: 8
sprints-elapsed: 100000
```

A sample experiment can be found in the `sample experiment` folder. Experiments we ran for our paper can be found [in a separated repo](https://github.com/vadim0x60/cibi-experiments).

## Codebases

`codebases` folder contains a few programs we've written for `CartPole-v1`, `MountainCarContinuous-v0` and `Taxi-v3` environments to be used as seeds.

For example, for `Taxi-v3` we've implemented an algorithm that finds what the current destination should be and always moves in its direction:

```
# Memory map
[T T Pid Did id_pointer D D A_plan]

# Destination derivation
e2[e+>0>0c[e>0>4e^-e>>>>0]c[e>4>0e^-e>>>>0]c[e>4>3e^-e>>>>0]]

# Absolute destination to relative destination
a[e>-a-]b[e>>-b-]

# Move using the relative destination
e>>>4+e---[e>>>4e0]e>[>>0e>0]e>~[>>1e>0]e>>[>2e>>0]e>>~[>3e>>0]>!

# Move back to a
a
```

If it's stuck at a wall, it does nothing to unstuck itself - a fatal flaw fixed by the program synthesis system.
