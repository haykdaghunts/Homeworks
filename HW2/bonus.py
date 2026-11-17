"""
BONUS: evidence for the improved implementation plan (see README, "Bonus").

A single run with one seed cannot rank bandit algorithms, because one unlucky
early reward can change the whole run. This script re-runs every configuration
on ``N_SEEDS`` seeds and compares the distribution of cumulative regret:

* Epsilon-Greedy with ``epsilon_t = 1/t`` (the literal assignment setting) and ``5/t`` (ours),
* Thompson Sampling with a tight ``N(0, 1)`` prior and with a weak ``N(0, 10^2)`` prior (ours),
* UCB1, a deterministic alternative with no tuning parameter.

Outputs ``img/bonus_multiseed.png`` and ``data/bonus_summary.csv``.
Run: ``python bonus.py`` (about 30 seconds).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from Bandit import (BANDIT_REWARD, DATA_DIR, IMG_DIR, NUMBER_OF_TRIALS,
                    REWARD_PRECISION, SEED, Bandit, EpsilonGreedy, ThompsonSampling)

N_SEEDS = 30


class UCB1(Bandit):
    """UCB1: play the arm with the largest upper confidence bound.

    Index of arm ``i`` at trial ``t``: ``mean_i + c * sigma * sqrt(2 ln t / n_i)``,
    where ``sigma`` is the known reward standard deviation. Each arm is played
    once before the index is used.

    :param p: true mean reward of each arm
    :type p: list
    :param c: exploration multiplier
    :type c: float
    :param n_trials: number of trials
    :type n_trials: int
    :param reward_precision: precision of the reward noise
    :type reward_precision: float
    :param seed: random seed
    :type seed: int
    """

    name = "UCB1"

    def __init__(self, p, c=1.0, n_trials=NUMBER_OF_TRIALS,
                 reward_precision=REWARD_PRECISION, seed=SEED):
        self._setup(p, n_trials, reward_precision, seed)
        self.c = c
        self.sigma = 1 / np.sqrt(reward_precision)
        self.p_estimate = np.zeros(self.k)
        self.t = 0

    def __repr__(self):
        return f"UCB1(bandits={self.p.tolist()}, c={self.c}, trials={self.n_trials})"

    def estimates(self):
        """Current sample mean of each arm.

        :return: one estimate per arm
        :rtype: numpy.ndarray
        """
        return self.p_estimate.copy()

    def pull(self):
        """Play an untried arm, otherwise the arm with the largest UCB index.

        :return: chosen arm and the observed reward
        :rtype: tuple
        """
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            arm = int(untried[0])
        else:
            bonus = self.c * self.sigma * np.sqrt(2 * np.log(self.t) / self.counts)
            arm = self._argmax(self.p_estimate + bonus)
        return arm, self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the sample mean of the played arm.

        :param arm: index of the played arm
        :type arm: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[arm] += 1
        self.p_estimate[arm] += (reward - self.p_estimate[arm]) / self.counts[arm]

    def experiment(self):
        """Run the UCB1 experiment.

        :return: one row per trial with columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        return self._run()

    def report(self, csv_path=None):
        """Log the cumulative reward and regret (no CSV for the bonus runs).

        :param csv_path: unused, kept for the Bandit interface
        :type csv_path: str
        :return: cumulative reward and cumulative regret
        :rtype: tuple
        """
        cum_reward, cum_regret = self.rewards.sum(), self.regrets.sum()
        logger.info(f"{self.name}: cumulative reward {cum_reward:,.2f}, cumulative regret {cum_regret:,.2f}")
        return cum_reward, cum_regret


CONFIGS = {
    "E-Greedy, eps=1/t": lambda s: EpsilonGreedy(BANDIT_REWARD, epsilon=1.0, seed=s),
    "E-Greedy, eps=5/t (ours)": lambda s: EpsilonGreedy(BANDIT_REWARD, epsilon=5.0, seed=s),
    "TS, prior N(0,1)": lambda s: ThompsonSampling(BANDIT_REWARD, prior_precision=1.0, seed=s),
    "TS, prior N(0,10^2) (ours)": lambda s: ThompsonSampling(BANDIT_REWARD, prior_precision=0.01, seed=s),
    "UCB1": lambda s: UCB1(BANDIT_REWARD, seed=s),
}


def run_all(n_seeds=N_SEEDS):
    """Run every configuration on ``n_seeds`` seeds.

    :param n_seeds: number of independent runs per configuration
    :type n_seeds: int
    :return: cumulative-regret curves (seeds x trials) per configuration
    :rtype: dict
    """
    curves = {}
    for label, make in CONFIGS.items():
        runs = []
        for s in range(n_seeds):
            alg = make(s)
            alg._run()
            runs.append(np.cumsum(alg.regrets))
        curves[label] = np.array(runs)
        final = curves[label][:, -1]
        logger.info(f"{label:<27} final regret: mean {final.mean():8.1f}, "
                    f"median {np.median(final):6.1f}, worst {final.max():8.1f}")
    return curves


def summarise(curves):
    """Summary table of the final cumulative regret and save it to CSV.

    :param curves: output of ``run_all``
    :type curves: dict
    :return: one row per configuration
    :rtype: pandas.DataFrame
    """
    rows = []
    for label, c in curves.items():
        final = c[:, -1]
        rows.append({"Configuration": label, "Mean regret": final.mean().round(1),
                     "Median regret": np.median(final).round(1), "Worst regret": final.max().round(1),
                     "Runs stuck (regret > 1000)": int((final > 1000).sum())})
    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "bonus_summary.csv"), index=False)
    logger.info("Final cumulative regret over seeds:\n" + df.to_string(index=False))
    return df


def plot(curves):
    """Mean cumulative regret over seeds (left) and final-regret spread (right).

    :param curves: output of ``run_all``
    :type curves: dict
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    trials = np.arange(1, NUMBER_OF_TRIALS + 1)
    for i, (label, c) in enumerate(curves.items()):
        axes[0].plot(trials, c.mean(axis=0), color=f"C{i}", label=label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title(f"Mean cumulative regret over {N_SEEDS} seeds")
    axes[0].set_xlabel("trial (log scale)")
    axes[0].set_ylabel("mean cumulative regret (log scale)")
    axes[0].legend(fontsize=8)

    axes[1].boxplot([c[:, -1] for c in curves.values()], showfliers=True)
    axes[1].set_xticks(range(1, len(curves) + 1))
    axes[1].set_xticklabels([l.replace(", ", "\n") for l in curves], fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_title(f"Final cumulative regret over {N_SEEDS} seeds")
    axes[1].set_ylabel("regret (log scale)")
    fig.tight_layout()
    path = os.path.join(IMG_DIR, "bonus_multiseed.png")
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot {os.path.relpath(path, os.path.dirname(IMG_DIR))}")


if __name__ == "__main__":
    curves = run_all()
    summarise(curves)
    plot(curves)
