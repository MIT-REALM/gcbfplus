<div align="center">

# GCBF+

[![Paper](https://img.shields.io/badge/T--RO-Accepted-success)](https://mit-realm.github.io/gcbfplus-website/)

Jax Official Implementation of T-RO Paper: [Songyuan Zhang*](https://syzhang092218-source.github.io), [Oswin So*](https://oswinso.xyz/), [Kunal Garg](https://kunalgarg.mit.edu/), and [Chuchu Fan](https://chuchu.mit.edu): "[GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control](https://mit-realm.github.io/gcbfplus-website/)". 

[Dependencies](#Dependencies) •
[Installation](#Installation) •
[Run](#Run)

</div>

A much improved version of [GCBFv0](https://mit-realm.github.io/gcbf-website/)!

<div align="center">
    <img src="./media/cbf1.gif" alt="LidarSpread" width="24.55%"/>
    <img src="./media/DoubleIntegrator_512_2x.gif" alt="LidarLine" width="24.55%"/>
    <img src="./media/Obstacle2D_32.gif" alt="VMASReverseTransport" width="24.55%"/>
    <img src="./media/Obstacle2D_512_2x.gif" alt="VMASWheel" width="24.55%"/>
</div>

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n gcbfplus python=3.10
conda activate gcbfplus
cd gcbfplus
```

Then install jax following the [official instructions](https://github.com/google/jax#installation), and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install GCBF: 

```bash
pip install -e .
```

## Run

### Environments

We provide 3 2D environments including `SingleIntegrator`, `DoubleIntegrator`, and `DubinsCar`, and 2 3D environments including `LinearDrone` and `CrazyFlie`.

### Algorithms

We provide algorithms including GCBF+ (`gcbf+`), GCBF (`gcbf`), centralized CBF-QP (`centralized_cbf`), and decentralized CBF-QP (`dec_share_cbf`). Use `--algo` to specify the algorithm. 

### Hyper-parameters

To reproduce the results shown in our paper, one can refer to [`settings.yaml`](./settings.yaml).

### Train

To train the model (only GCBF+ and GCBF need training), use:

```bash
python train.py --algo gcbf+ --env DoubleIntegrator -n 8 --area-size 4 --loss-action-coef 1e-4 --n-env-train 16 --lr-actor 1e-5 --lr-cbf 1e-5 --horizon 32
```

In our paper, we use 8 agents with 1000 training steps. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`. We also provide the following flags:

- `-n`: number of agents
- `--env`: environment, including `SingleIntegrator`, `DoubleIntegrator`, `DubinsCar`, `LinearDrone`, and `CrazyFlie`
- `--algo`: algorithm, including `gcbf`, `gcbf+`
- `--seed`: random seed
- `--steps`: number of training steps
- `--name`: name of the experiment
- `--debug`: debug mode: no recording, no saving
- `--obs`: number of obstacles
- `--n-rays`: number of LiDAR rays
- `--area-size`: side length of the environment
- `--n-env-train`: number of environments for training
- `--n-env-test`: number of environments for testing
- `--log-dir`: path to save the training logs
- `--eval-interval`: interval of evaluation
- `--eval-epi`: number of episodes for evaluation
- `--save-interval`: interval of saving the model

In addition, use the following flags to specify the hyper-parameters:
- `--alpha`: GCBF alpha
- `--horizon`: GCBF+ look forward horizon
- `--lr-actor`: learning rate of the actor
- `--lr-cbf`: learning rate of the CBF
- `--loss-action-coef`: coefficient of the action loss
- `--loss-h-dot-coef`: coefficient of the h_dot loss
- `--loss-safe-coef`: coefficient of the safe loss
- `--loss-unsafe-coef`: coefficient of the unsafe loss
- `--buffer-size`: size of the replay buffer

### Test

To test the learned model, use:

```bash
python test.py --path <path-to-log> --epi 5 --area-size 4 -n 16 --obs 0
```

This should report the safety rate, goal reaching rate, and success rate of the learned model, and generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--path`: path to the log folder
- `--n-rays`: number of LiDAR rays
- `--alpha`: CBF alpha, used in centralized CBF-QP and decentralized CBF-QP
- `--max-travel`: maximum travel distance of agents
- `--cbf`: plot the CBF contour of this agent, only support 2D environments
- `--seed`: random seed
- `--debug`: debug mode
- `--cpu`: use CPU
- `--u-ref`: test the nominal controller
- `--env`: test environment (not needed if the log folder is specified)
- `--algo`: test algorithm (not needed if the log folder is specified)
- `--step`: test step (not needed if testing the last saved model)
- `--epi`: number of episodes to test
- `--offset`: offset of the random seeds
- `--no-video`: do not generate videos
- `--log`: log the results to a file
- `--dpi`: dpi of the video
- `--nojit-rollout`: do not use jit to speed up the rollout, used for large-scale tests

To test the nominal controller, use:

```bash
python test.py --env SingleIntegrator -n 16 --u-ref --epi 1 --area-size 4 --obs 0
```

To test the CBF-QPs, use:

```bash
python test.py --env SingleIntegrator -n 16 --algo dec_share_cbf --epi 1 --area-size 4 --obs 0 --alpha 1
```

### Pre-trained models

We provide the pre-trained models in the folder [`pretrained`](pretrained).

## Citation

```
@ARTICLE{zhang2025gcbf+,
      author={Zhang, Songyuan and So, Oswin and Garg, Kunal and Fan, Chuchu},
      journal={IEEE Transactions on Robotics}, 
      title={GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multi-Agent Control}, 
      year={2025},
      volume={},
      number={},
      pages={1-20},
      doi={10.1109/TRO.2025.3530348}
}
```

## Acknowledgement

The developers were partially supported by MITRE during the project.

© 2024 MIT

© 2024 The MITRE Corporation
