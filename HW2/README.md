# DS 223 HW2 — A/B Testing with Epsilon-Greedy and Thompson Sampling

Four advertisement options (bandits) with true mean rewards `Bandit_Reward = [1, 2, 3, 4]`
are tested over `20,000` trials. Each pull returns a reward drawn from `N(mean, 1)`,
which has known precision τ = 1.

## How to run

```bash
cd HW2
pip install -r requirements.txt
python Bandit.py      # the assignment: both experiments, CSV, plots, logged results
python bonus.py       # BONUS: 30-seed comparison backing the improvement plan (~30 s)
```

Both scripts write their outputs next to themselves (`data/`, `img/`), so they can be run from any folder.
All output goes through `loguru` instead of `print()`.

## Files

| File | Content |
|---|---|
| `Bandit.py` | `Bandit` (the given abstract class, nothing removed), `EpsilonGreedy`, `ThompsonSampling`, `Visualization` (`plot1`, `plot2`), `comparison()` |
| `bonus.py` | Bonus: `UCB1` (another `Bandit` subclass) and the multi-seed comparison |
| `data/bandit_rewards.csv` | Every trial of both algorithms: `Bandit, Reward, Algorithm` (40,000 rows). `Bandit` is the 1-based arm number. |
| `data/bonus_summary.csv` | Bonus: regret statistics over 30 seeds |
| `img/learning_epsilon_greedy.png`, `img/learning_thompson_sampling.png` | `plot1()`: learning process of each algorithm, linear and log scale |
| `img/cumulative_reward_regret.png` | `plot2()`: cumulative reward and cumulative regret, E-Greedy vs Thompson Sampling |
| `img/comparison.png` | `comparison()`: share of trials on the best bandit and pulls per bandit |
| `img/bonus_multiseed.png` | Bonus: regret distributions over 30 seeds |

## Design

**Epsilon-Greedy.** At trial *t* the algorithm explores (picks a random bandit) with probability
ε_t = min(1, ε / t), so epsilon decays by 1/t. Otherwise it exploits the bandit with the highest sample mean.
I use **ε = 5**. With ε = 1 (ε_t = 1/t) there are only about ln(20,000) ≈ 10 exploration trials in total,
and in 4 of 30 seeds the algorithm locked onto a worse bandit (see the bonus table).

**Thompson Sampling with known precision.** Rewards are `N(μ_i, 1/τ)` with known **τ = 1**.
Each μ_i has a Normal prior `N(m₀, 1/λ₀)`. After *n_i* pulls with reward sum *S_i*, the posterior is
`N(m_i, 1/λ_i)`, where λ_i = λ₀ + τ·n_i and m_i = (λ₀·m₀ + τ·S_i) / λ_i.
Each trial draws one μ_i from every posterior and plays the largest.
I use a **weak prior N(0, 10²)** (λ₀ = 0.01). A tight N(0, 1) prior is pessimistic for rewards of 1–4:
an untried bandit almost never draws above 3, so the best bandit can be missed for thousands of trials.

**Regret.** The regret of a trial is `max(mean) − mean(chosen bandit)` (expected regret).
The log also shows the realised regret, `N · max(mean) − total reward`.

## Results (seed 42)

| | Cumulative reward | Cumulative regret | Trials on best bandit |
|---|---|---|---|
| Epsilon-Greedy (ε = 5/t) | 79,964.69 | 86.0 | 99.81% |
| Thompson Sampling (τ = 1, prior N(0, 10²)) | 79,937.44 | 37.0 | 99.88% |

Both algorithms find Bandit 4. Thompson Sampling has less than half the regret, because it explores
in proportion to how uncertain each bandit is. Epsilon-Greedy explores at random and keeps picking
the clearly worst bandits. The cumulative rewards are close to each other because reward noise
(sd 1 over 20,000 trials) is of the same size as the regret gap. Regret is the clearer measure.

## BONUS — Better implementation plan

**1. Evaluate over many seeds, not one run.** One seed cannot rank bandit algorithms. A single unlucky
early reward decides the whole run. `bonus.py` repeats every configuration with 30 seeds:

| Configuration | Mean regret | Median | Worst | Runs stuck (regret > 1000) |
|---|---|---|---|---|
| E-Greedy, ε = 1/t | 1019.7 | 21.5 | 20013.0 | 4 |
| E-Greedy, ε = 5/t (ours) | 70.8 | 70.5 | 127.0 | 0 |
| TS, prior N(0, 1) | 342.2 | 64.0 | 2866.0 | 2 |
| TS, prior N(0, 10²) (ours) | 32.1 | 31.0 | 51.0 | 0 |
| UCB1 | 36.7 | 36.0 | 52.0 | 0 |

The literal ε = 1/t has the best *median* but the worst *mean*. In one run it played a worse bandit
for all 20,000 trials. A single lucky seed would hide that risk. This is also why the parameters
in `Bandit.py` were picked from these runs and not from one run.

**2. Make the algorithms robust by default.**
- Play every bandit once before the algorithm starts choosing, or use a weak or optimistic prior,
  so that no bandit is dismissed from zero data.
- Tune the scale of ε to the problem. ε_t = min(1, c/(d²t)) (Auer et al., 2002, with *d* a lower
  bound on the gap between bandits) is the version with a regret guarantee. A bare 1/t is not.
- Use UCB1 (in `bonus.py`) or Thompson Sampling. Both explore by uncertainty, have no exploration
  rate to tune, and reach logarithmic regret. They match each other here (~32–37 vs ~71 for ε-greedy).

**3. Model the real problem.**
- Ad rewards are usually clicks or conversions. Bernoulli bandits with a Beta prior fit them better than Normal rewards.
- If the reward variance is unknown, use Thompson Sampling with a Normal-Gamma prior, so precision is
  learned rather than assumed.
- Real ads have delayed and batched feedback, and CTR drifts over time. Use batch updates, and use
  discounted or sliding-window estimates for non-stationary ads.
- If user features are available (device, time, segment), use a contextual bandit such as LinUCB or
  linear Thompson Sampling to choose a different best ad per segment.

**4. Software structure.**
- Separate the *environment* (true means, reward noise) from the *policy* (`pull`/`update`), so the
  same simulated traffic (same random stream) can be replayed for every algorithm. That gives a paired,
  lower-variance comparison.
- Put the settings (means, trials, ε, τ, prior, seeds) in one config.
- Add unit tests: the posterior update against the closed form, ε_t schedule, and CSV schema.
- Vectorise over seeds with NumPy for fast large-scale evaluation.
- Report regret with confidence bands and a stopping rule (for example, stop when P(best bandit) > 95%).
