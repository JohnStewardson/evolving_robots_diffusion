Overview:

## Evolving Soft Robot Morphologies with Generative Diffusion Models

This Git repository contains the code developed for my Bachelor's thesis
on evolving soft robot morphologies using generative diffusion models.
The thesis is included in the repository for reference (see
"BA_John_Stewardson.pdf"). You can read the thesis for a detailed
explanation of the research and methods used, and explore the code to
inspect the implementation of the models and experiments.



#### Note:

This repository is currently under development to enable pip-installation and add scripts using EvoGym environments without having to install that repository, but you can install it from source in the meantime, and explore the concepts on a simplified environment.

## Setup:

This setup was tested with Ubuntu Noble and Python 3.10, but should be compatible with any system also compatible with Evogym (see the github: [https://github.com/EvolutionGym/evogym](https://github.com/EvolutionGym/evogym))


1. (Recommended: Conda environment)

```conda create -n evogym_diffusion python=3.10 -yconda activate evogym_diffusion
conda create -n evogym_diffusion python=3.10 -y
conda activate evogym_diffusion
```

2. Installing pytorch (device-dependent, repository tested with this version:)

```
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

3. Build from source:

```
pip install -e .
```


## Example scripts:

**run_simple_generational_dm.py:**

Entire pipeline of the algorithm:

* Sampling structures
* Repeat:
  * Evaluating structures
  * Training diffusion model on survivors
  * Sampling new generation
* Plot results

**run_simple_ga.py** and **run_simple_cppn.py**:

* two alternative algorithms on the same problem (genetic algorithm and CPPN-NEAT)


**visualize_forward_process.py:**

Visualizes adding noise to a robot configuration for different timesteps


**train_diffusion.py:**

One step of training the model on the survivors and sampling. In order to illustrate how the mutation works, all survivors are the same robot.

**eval_overfitting.py:**

Similarly to train_diffusion.py, train on one survivor and sample to analyze how often the input robot is reconstructed.


### Source code:

pytorch neat: package dependency, from: [https://github.com/uber-research/PyTorch-NEAT](https://github.com/uber-research/PyTorch-NEAThttps:/)

evolving_robots_diffusion: developed package:

* diffusion_model/: code related to the diffusion model and its training
  * Additionally included is the code for hyperparameter optimization, that was used to tune the parameters in the thesis
* simple_env/:
  * code for the simplified environment to test algorithms without a RL loop
* optimizers/:
  * code for the generational diffusion algorithm (interface from diffusion model to the optimization problem)
  * code for the genetic algorithm and CPPN-NEAT for comparison in the simple environment
* plotting/:
  * plotting the results
