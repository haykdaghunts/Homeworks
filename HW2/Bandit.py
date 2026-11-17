"""
A/B testing of four advertisement options with Epsilon-Greedy and Thompson Sampling.

Each advertisement is a bandit arm whose reward is drawn from a Normal
distribution with true mean ``BANDIT_REWARD[i]`` and a known precision
(``REWARD_PRECISION``, i.e. variance ``1 / REWARD_PRECISION``).
Two algorithms play ``NUMBER_OF_TRIALS`` rounds each:

* ``EpsilonGreedy``: explores with probability ``epsilon_t = min(1, epsilon / t)``
  (epsilon decays by 1/t) and otherwise plays the arm with the highest sample mean.
* ``ThompsonSampling``: Gaussian Thompson Sampling with known precision. Each arm
  has a Normal prior on its mean, updated with the conjugate Normal posterior.
  The arm with the largest posterior draw is played.

Running this file (``python Bandit.py``) runs both experiments, then

* saves the plots to ``img/``,
* stores every trial in ``data/bandit_rewards.csv`` with columns
  ``Bandit, Reward, Algorithm``,
* logs the cumulative reward and the cumulative regret of each algorithm.

All messages go through ``loguru`` instead of ``print()``.
"""
############################### LOGGER
import os
import sys
from abc import ABC, abstractmethod

import matplotlib

matplotlib.use("Agg")  # write figures to files, never block on a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

############################### SETTINGS
BANDIT_REWARD = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
EPSILON = 5.0           # epsilon_t = min(1, EPSILON / t): decays by 1/t
REWARD_PRECISION = 1.0  # known precision tau of every arm's reward (variance = 1 / tau)
PRIOR_MEAN = 0.0        # Thompson Sampling prior on each arm's mean ...
PRIOR_PRECISION = 0.01  # ... N(PRIOR_MEAN, 1 / PRIOR_PRECISION), i.e. prior sd 10 (weak)
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
IMG_DIR = os.path.join(HERE, "img")
CSV_PATH = os.path.join(DATA_DIR, "bandit_rewards.csv")


class Bandit(ABC):
    """Abstract multi-armed bandit experiment (template given with the assignment).

    The subclasses below receive the list of true arm means ``p`` and play
    the whole experiment against those arms. The concrete helper methods
    at the end of the class are additions shared by both algorithms.
    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

    ##==== Additions shared by both algorithms ====##

    def _setup(self, p, n_trials, reward_precision, seed):
        """Initialise the environment and the per-trial history.

        :param p: true mean reward of each arm
        :type p: list
        :param n_trials: number of trials to play
        :type n_trials: int
        :param reward_precision: precision (1 / variance) of the reward noise
        :type reward_precision: float
        :param seed: random seed, for reproducible runs
        :type seed: int
        """
        self.p = np.asarray(p, dtype=float)
        self.k = len(self.p)
        self.n_trials = n_trials
        self.reward_precision = reward_precision
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.k, dtype=int)
        self.chosen = np.zeros(n_trials, dtype=int)
        self.rewards = np.zeros(n_trials)
        self.estimates_history = np.zeros((n_trials, self.k))

    def _draw_reward(self, arm):
        """Draw a reward from an arm: N(p[arm], 1 / reward_precision).

        :param arm: index of the arm that is played
        :type arm: int
        :return: the observed reward
        :rtype: float
        """
        return self.rng.normal(self.p[arm], 1 / np.sqrt(self.reward_precision))

    def _argmax(self, values):
        """Index of the largest value; ties are broken at random.

        :param values: one value per arm
        :type values: numpy.ndarray
        :return: index of the chosen arm
        :rtype: int
        """
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def _run(self):
        """Play ``n_trials`` rounds: pull, observe the reward, update.

        :return: one row per trial with columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        for t in range(self.n_trials):
            self.t = t + 1
            arm, reward = self.pull()
            self.update(arm, reward)
            self.chosen[t] = arm
            self.rewards[t] = reward
            self.estimates_history[t] = self.estimates()
            if self.t % 5000 == 0:
                logger.debug(f"{self.name}: trial {self.t:,}, estimates {np.round(self.estimates(), 3)}")
        return self.to_frame()

    @property
    def regrets(self):
        """Per-trial (expected) regret: best arm mean minus the played arm's mean.

        :return: regret of each trial
        :rtype: numpy.ndarray
        """
        return self.p.max() - self.p[self.chosen]

    def to_frame(self):
        """Trial history in the format required for the CSV.

        ``Bandit`` is the 1-based arm number (Bandit 1 has true mean
        ``BANDIT_REWARD[0]``, and so on).

        :return: columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame({
            "Bandit": self.chosen + 1,
            "Reward": self.rewards,
            "Algorithm": self.name,
        })

    def _report(self, csv_path):
        """Store this algorithm's trials in the CSV and log its results.

        Rows of the same algorithm already in the file are replaced, so
        re-running the script never duplicates data.

        :param csv_path: path of the CSV file
        :type csv_path: str
        :return: cumulative reward and cumulative regret
        :rtype: tuple
        """
        df = self.to_frame()
        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path)
            df = pd.concat([old[old["Algorithm"] != self.name], df], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"{self.name}: {self.n_trials:,} trials stored in {os.path.relpath(csv_path, HERE)}")

        cum_reward = self.rewards.sum()
        cum_regret = self.regrets.sum()
        realised_regret = self.n_trials * self.p.max() - cum_reward
        logger.info(f"{self.name}: cumulative reward = {cum_reward:,.2f}")
        logger.info(f"{self.name}: cumulative regret = {cum_regret:,.2f} "
                    f"(realised: {realised_regret:,.2f})")
        logger.info(f"{self.name}: average reward = {cum_reward / self.n_trials:.4f} "
                    f"(best possible {self.p.max():.4f})")
        logger.info(f"{self.name}: average regret = {cum_regret / self.n_trials:.4f}")
        for i in range(self.k):
            logger.info(f"{self.name}:   Bandit {i + 1} (true mean {self.p[i]:.0f}) "
                        f"pulled {self.counts[i]:>6,} times, estimate {self.estimates()[i]:.4f}")
        return cum_reward, cum_regret

#--------------------------------------#


class Visualization():
    """Plots of the learning process and of the two algorithms side by side.

    :param algorithms: algorithms that have already run ``experiment()``
    :type algorithms: list
    :param out_dir: folder where the figures are saved
    :type out_dir: str
    """

    def __init__(self, algorithms, out_dir=IMG_DIR):
        self.algorithms = algorithms
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _save(self, fig, name):
        """Save a figure to ``out_dir`` and close it.

        :param fig: figure to save
        :type fig: matplotlib.figure.Figure
        :param name: file name
        :type name: str
        """
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot {os.path.relpath(path, HERE)}")

    def plot1(self):
        """Learning process of each algorithm, on linear and log trial axes.

        One figure per algorithm. The top row shows the running average
        reward against the best arm's mean. The bottom row shows how each
        bandit's estimated mean (sample mean for Epsilon-Greedy, posterior
        mean for Thompson Sampling) converges to its true mean.
        """
        for alg in self.algorithms:
            trials = np.arange(1, alg.n_trials + 1)
            avg_reward = np.cumsum(alg.rewards) / trials
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
            for col, scale in enumerate(["linear", "log"]):
                ax = axes[0, col]
                ax.plot(trials, avg_reward, color="C0", label="running average reward")
                ax.axhline(alg.p.max(), color="black", ls="--", lw=1, label="best bandit mean")
                ax.set_title(f"Average reward ({scale} scale)")
                ax.set_ylabel("average reward")
                ax.legend(loc="lower right")

                ax = axes[1, col]
                for i in range(alg.k):
                    ax.plot(trials, alg.estimates_history[:, i], color=f"C{i}",
                            label=f"Bandit {i + 1} estimate")
                    ax.axhline(alg.p[i], color=f"C{i}", ls="--", lw=1)
                ax.set_title(f"Estimated mean of each bandit ({scale} scale)")
                ax.set_xlabel("trial")
                ax.set_ylabel("estimated mean (dashed: true)")
                ax.legend(loc="lower right", fontsize=8)
                for a in axes[:, col]:
                    a.set_xscale(scale)
            fig.suptitle(f"Learning process: {alg.name}")
            fig.tight_layout()
            self._save(fig, f"learning_{alg.name.lower().replace(' ', '_').replace('-', '_')}.png")

    def plot2(self):
        """Compare the cumulative rewards and cumulative regrets of the algorithms."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, alg in enumerate(self.algorithms):
            trials = np.arange(1, alg.n_trials + 1)
            axes[0].plot(trials, np.cumsum(alg.rewards), color=f"C{i}", label=alg.name)
            axes[1].plot(trials, np.cumsum(alg.regrets), color=f"C{i}", label=alg.name)
        best = self.algorithms[0].p.max()
        trials = np.arange(1, self.algorithms[0].n_trials + 1)
        axes[0].plot(trials, best * trials, color="black", ls="--", lw=1, label="always best bandit")
        axes[0].set_title("Cumulative reward")
        axes[0].set_ylabel("cumulative reward")
        axes[1].set_title("Cumulative regret")
        axes[1].set_ylabel("cumulative regret")
        for ax in axes:
            ax.set_xlabel("trial")
            ax.legend()
        fig.suptitle("Epsilon-Greedy vs Thompson Sampling")
        fig.tight_layout()
        self._save(fig, "cumulative_reward_regret.png")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy with epsilon decaying as ``epsilon_t = min(1, epsilon / t)``.

    ``epsilon = 5`` is used instead of 1: with ``epsilon_t = 1 / t`` there are only
    about ln(20000) ~ 10 exploration trials, and in 3 of 30 seeds the algorithm
    locked onto a worse bandit for the whole run (see README).

    :param p: true mean reward of each arm
    :type p: list
    :param epsilon: initial exploration rate (``epsilon_1``)
    :type epsilon: float
    :param n_trials: number of trials
    :type n_trials: int
    :param reward_precision: precision of the reward noise
    :type reward_precision: float
    :param seed: random seed
    :type seed: int
    """

    name = "Epsilon-Greedy"

    def __init__(self, p, epsilon=EPSILON, n_trials=NUMBER_OF_TRIALS,
                 reward_precision=REWARD_PRECISION, seed=SEED):
        self._setup(p, n_trials, reward_precision, seed)
        self.epsilon = epsilon
        self.p_estimate = np.zeros(self.k)
        self.t = 0
        self.n_explore = 0

    def __repr__(self):
        return (f"EpsilonGreedy(bandits={self.p.tolist()}, epsilon_t={self.epsilon}/t, "
                f"trials={self.n_trials})")

    def estimates(self):
        """Current sample mean reward of each arm.

        :return: one estimate per arm
        :rtype: numpy.ndarray
        """
        return self.p_estimate.copy()

    def pull(self):
        """Choose an arm (explore with probability epsilon / t, else exploit) and play it.

        :return: chosen arm and the observed reward
        :rtype: tuple
        """
        eps_t = min(1.0, self.epsilon / self.t)
        if self.rng.random() < eps_t:
            arm = int(self.rng.integers(self.k))
            self.n_explore += 1
        else:
            arm = self._argmax(self.p_estimate)
        return arm, self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the sample mean of the played arm incrementally.

        :param arm: index of the played arm
        :type arm: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[arm] += 1
        self.p_estimate[arm] += (reward - self.p_estimate[arm]) / self.counts[arm]

    def experiment(self):
        """Run the Epsilon-Greedy experiment.

        :return: one row per trial with columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        logger.info(f"Running {self!r}")
        df = self._run()
        logger.info(f"{self.name}: explored in {self.n_explore} of {self.n_trials:,} trials "
                    f"(final epsilon {min(1.0, self.epsilon / self.n_trials):.2e})")
        return df

    def report(self, csv_path=CSV_PATH):
        """Store the trials in the CSV and log cumulative reward and regret.

        :param csv_path: path of the CSV file
        :type csv_path: str
        :return: cumulative reward and cumulative regret
        :rtype: tuple
        """
        return self._report(csv_path)

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Gaussian Thompson Sampling with known precision.

    Every arm's reward is ``N(mu_i, 1 / tau)`` with known ``tau``. The prior
    on ``mu_i`` is ``N(m_0, 1 / lambda_0)``. After ``n_i`` rewards with sum
    ``S_i`` the posterior is Normal with

    * precision ``lambda_i = lambda_0 + tau * n_i``
    * mean ``m_i = (lambda_0 * m_0 + tau * S_i) / lambda_i``

    Each trial draws one ``mu_i`` per arm from its posterior and plays the largest.
    A weak prior (sd 10) is used: a tight ``N(0, 1)`` prior is pessimistic for
    rewards around 1-4 and can delay trying the best bandit by thousands of trials.

    :param p: true mean reward of each arm
    :type p: list
    :param precision: known precision ``tau`` of the rewards, as assumed by the algorithm
    :type precision: float
    :param prior_mean: prior mean ``m_0``
    :type prior_mean: float
    :param prior_precision: prior precision ``lambda_0``
    :type prior_precision: float
    :param n_trials: number of trials
    :type n_trials: int
    :param reward_precision: true precision of the reward noise in the environment
    :type reward_precision: float
    :param seed: random seed
    :type seed: int
    """

    name = "Thompson Sampling"

    def __init__(self, p, precision=REWARD_PRECISION, prior_mean=PRIOR_MEAN,
                 prior_precision=PRIOR_PRECISION, n_trials=NUMBER_OF_TRIALS,
                 reward_precision=REWARD_PRECISION, seed=SEED):
        self._setup(p, n_trials, reward_precision, seed)
        self.tau = precision
        self.m0 = prior_mean
        self.lambda0 = prior_precision
        self.post_mean = np.full(self.k, prior_mean, dtype=float)
        self.post_precision = np.full(self.k, prior_precision, dtype=float)
        self.reward_sum = np.zeros(self.k)
        self.t = 0

    def __repr__(self):
        return (f"ThompsonSampling(bandits={self.p.tolist()}, tau={self.tau}, "
                f"prior=N({self.m0}, 1/{self.lambda0}), trials={self.n_trials})")

    def estimates(self):
        """Current posterior mean of each arm.

        :return: one estimate per arm
        :rtype: numpy.ndarray
        """
        return self.post_mean.copy()

    def pull(self):
        """Draw a mean for every arm from its posterior and play the largest.

        :return: chosen arm and the observed reward
        :rtype: tuple
        """
        samples = self.rng.normal(self.post_mean, 1 / np.sqrt(self.post_precision))
        arm = self._argmax(samples)
        return arm, self._draw_reward(arm)

    def update(self, arm, reward):
        """Conjugate Normal update of the played arm's posterior.

        :param arm: index of the played arm
        :type arm: int
        :param reward: observed reward
        :type reward: float
        """
        self.counts[arm] += 1
        self.reward_sum[arm] += reward
        self.post_precision[arm] = self.lambda0 + self.tau * self.counts[arm]
        self.post_mean[arm] = (self.lambda0 * self.m0 + self.tau * self.reward_sum[arm]) \
            / self.post_precision[arm]

    def experiment(self):
        """Run the Thompson Sampling experiment.

        :return: one row per trial with columns Bandit, Reward, Algorithm
        :rtype: pandas.DataFrame
        """
        logger.info(f"Running {self!r}")
        return self._run()

    def report(self, csv_path=CSV_PATH):
        """Store the trials in the CSV and log cumulative reward and regret.

        :param csv_path: path of the CSV file
        :type csv_path: str
        :return: cumulative reward and cumulative regret
        :rtype: tuple
        """
        return self._report(csv_path)


def comparison(algorithms, out_dir=IMG_DIR):
    """Compare the algorithms visually and in a summary table.

    The figure shows (left) the share of trials that played the best bandit,
    as a moving average over 500 trials on a log trial axis, and (right) how
    often each bandit was played. The table with the cumulative reward,
    cumulative regret and best-bandit share goes to the log.

    :param algorithms: algorithms that have already run ``experiment()``
    :type algorithms: list
    :param out_dir: folder where the figure is saved
    :type out_dir: str
    :return: one row of summary statistics per algorithm
    :rtype: pandas.DataFrame
    """
    window = 500
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.8 / len(algorithms)
    for j, alg in enumerate(algorithms):
        best = (alg.chosen == alg.p.argmax()).astype(float)
        moving = pd.Series(best).rolling(window, min_periods=1).mean()
        axes[0].plot(np.arange(1, alg.n_trials + 1), moving, color=f"C{j}", label=alg.name)
        axes[1].bar(np.arange(alg.k) + (j - (len(algorithms) - 1) / 2) * width, alg.counts,
                    width=width, color=f"C{j}", label=alg.name)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Share of trials on the best bandit ({window}-trial moving average)")
    axes[0].set_xlabel("trial (log scale)")
    axes[0].set_ylabel("share")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(algorithms[0].k))
    axes[1].set_xticklabels([f"Bandit {i + 1}\n(mean {m:g})" for i, m in enumerate(algorithms[0].p)])
    axes[1].set_title("Number of pulls per bandit")
    axes[1].set_ylabel("pulls (log scale)")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot {os.path.relpath(path, HERE)}")

    summary = pd.DataFrame([{
        "Algorithm": alg.name,
        "Cumulative reward": round(alg.rewards.sum(), 2),
        "Cumulative regret": round(alg.regrets.sum(), 2),
        "Best bandit share": round((alg.chosen == alg.p.argmax()).mean(), 4),
    } for alg in algorithms]).set_index("Algorithm")
    logger.info("Comparison of the algorithms:\n" + summary.to_string())
    return summary


if __name__=='__main__':
    eg = EpsilonGreedy(BANDIT_REWARD)
    ts = ThompsonSampling(BANDIT_REWARD)

    for alg in (eg, ts):
        alg.experiment()
        alg.report()

    viz = Visualization([eg, ts])
    viz.plot1()
    viz.plot2()
    comparison([eg, ts])
