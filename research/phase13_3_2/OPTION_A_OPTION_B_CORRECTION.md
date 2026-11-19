# Statistical Correction: Formal Analysis of Poisson Subsampling Schedules (Option A vs. Option B)

**Phase:** 13.3.2  
**Task:** Mathematical Correction and Rigorous Statistical Formulation of DP Poisson Sampling  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Differential Privacy Specialist  

---

## 1. Executive Summary

In Phase 13.3 and Phase 13.3.1, a discussion was presented comparing two design options for configuring the Poisson subsampling rate $q$ and expected batch size $B_{\text{exp}}$ under an encounter training cohort of $N = 1,132$ records:
- **Option A:** Fix steps per epoch to $K = \lceil N / B_{\text{nom}} \rceil = 36$, setting $q = 1 / 36$ and $B_{\text{exp}} = qN = 31.4444$.
- **Option B:** Fix expected batch size to $B_{\text{exp}} = 32$, setting $q = 32 / 1,132 \approx 0.028269$.

The previous audit narrative contained a **statistically flawed explanation** of Option B, asserting that:
> *"Setting $q = 32/1,132$ with 35 steps per epoch undersamples the cohort by 12 patients per epoch ($1,132 - (35 \times 32) = 12$)."*

This audit formally corrects this explanation. We demonstrate why treating Poisson subsampling as a deterministic partition is mathematically incorrect, formulate the exact probability distributions governing Poisson subsampled DP-SGD, and contrast Option A and Option B under rigorous statistical privacy theory.

---

## 2. Mathematical Flaw in the Previous Explanation

The claim that 35 batches of expected size 32 "leaves behind 12 patients" conflates **deterministic batching without replacement** (standard epoch shuffling) with **Poisson subsampling** (independent Bernoulli trials):
1. In deterministic mini-batching, a fixed dataset $\mathcal{D}$ of size $N$ is permuted and sliced into disjoint subsets $B_1, \dots, B_K$. If $N$ is not divisible by $B_{\text{nom}}$, a final fractional batch or discarded remainder occurs.
2. In Poisson subsampling, there are **no fixed partitions, no fixed batch sizes, and no guarantees of disjointness across steps**. Each record $z_i \in \mathcal{D}$ is included in batch $B_t$ with independent probability $q$, completely independent of other records and independent of past iterations.
3. Therefore, no specific "12 patients" are systematically excluded. Every patient in the dataset has an identical, positive probability of being sampled in every step.

---

## 3. Formal Statistical Formulation of Poisson Subsampling

### 3.1 Definition of the Sampling Mechanism
Let $\mathcal{D} = \{z_1, z_2, \dots, z_N\}$ be the training dataset of $N$ encounters. At each step $t \in \{1, 2, \dots, T\}$, the mini-batch $B_t$ is constructed via independent Bernoulli trials:
$$\mathbb{I}(z_i \in B_t) \sim \text{Bernoulli}(q), \quad \forall i \in \{1, \dots, N\}$$

The actual batch size $|B_t|$ is a random variable following a binomial distribution:
$$|B_t| \sim \text{Binomial}(N, q)$$
with expectation and variance:
$$\mathbb{E}[|B_t|] = qN, \quad \text{Var}(|B_t|) = Nq(1 - q)$$

### 3.2 Total Gradient Calculations
Over a total training horizon of $T$ optimization steps, the total number of sample-gradient evaluations $M_{\text{total}} = \sum_{t=1}^T |B_t|$ has expectation:
$$\mathbb{E}[M_{\text{total}}] = \sum_{t=1}^T \mathbb{E}[|B_t|] = T \cdot q \cdot N$$

### 3.3 Individual Sample Coverage Probability
Because inclusions are independent across steps, the probability that an individual record $z_i$ is **not selected** in a single step is $1 - q$.  
Over $T$ independent steps, the probability that record $z_i$ is **never sampled** is:
$$\mathbb{P}(z_i \notin \bigcup_{t=1}^T B_t) = (1 - q)^T$$

Consequently, the probability that an individual encounter is sampled **at least once** during training is:
$$\mathbb{P}(z_i \text{ is sampled } \ge 1 \text{ time}) = 1 - (1 - q)^T$$

The expected number of times a specific record $z_i$ is processed over $T$ steps is:
$$\mathbb{E}[\text{count}(z_i)] = T \cdot q$$

---

## 4. Quantitative Comparison: Option A vs. Option B

For the landmark training cohort, $N = 1,132$ encounters and the training schedule spans $E = 5$ epochs.

### 4.1 Option A (Implemented in Phase 13.3.1)
- **Schedule:** Steps per epoch $K_A = \lceil N / B_{\text{nom}} \rceil = \lceil 1,132 / 32 \rceil = 36$ steps/epoch.
- **Total Steps:** $T_A = 5 \times 36 = 180$ steps.
- **Sampling Rate:** $q_A = \frac{1}{K_A} = \frac{1}{36} = 0.0277777778$.
- **Expected Batch Size:** $B_{\text{exp}, A} = q_A N = \frac{1,132}{36} = 31.444444...$
- **Expected Total Gradient Evaluations:**
  $$\mathbb{E}[M_A] = 180 \times \left(\frac{1}{36}\right) \times 1,132 = 5 \times 1,132 = 5,660$$
  *Property:* Exactly matches 5 complete presentations of the dataset in expectation.
- **Expected Presentations per Record:** $\mathbb{E}[\text{count}(z_i)] = 180 \times \frac{1}{36} = 5.0000$.
- **Sample Coverage Probability:**
  $$\mathbb{P}(\text{sampled } \ge 1) = 1 - \left(1 - \frac{1}{36}\right)^{180} = 1 - \left(\frac{35}{36}\right)^{180} \approx 1 - 0.006209 = \mathbf{99.3791\%}$$

### 4.2 Option B (Integer Nominal Normalization)
- **Sampling Rate:** Set directly from nominal batch size:
  $$q_B = \frac{B_{\text{nom}}}{N} = \frac{32}{1,132} \approx 0.02826855$$
- **Expected Batch Size:** $B_{\text{exp}, B} = q_B N = 32.0$ (exact integer).
- **Step Count Dilemma:** Because $1,132 / 32 = 35.375$ is non-integer, the number of steps per epoch must be discretized:

#### Case B.1: Truncation to 35 steps/epoch ($T = 175$ steps over 5 epochs)
- Total gradient evaluations: $\mathbb{E}[M_{B, 35}] = 175 \times 32 = 5,600$.
- Gradient deficit: Evaluates $5,660 - 5,600 = 60$ fewer sample gradients over 5 epochs (an expectation gap of $1.06\%$ fewer computations across the cohort, **not** 12 specific excluded patients).
- Expected presentations per record: $\mathbb{E}[\text{count}(z_i)] = 175 \times \frac{32}{1,132} \approx 4.9470$.
- Sample coverage probability:
  $$\mathbb{P}(\text{sampled } \ge 1) = 1 - \left(1 - \frac{32}{1,132}\right)^{175} \approx 1 - 0.006596 = \mathbf{99.3404\%}$$

#### Case B.2: Ceiling to 36 steps/epoch ($T = 180$ steps over 5 epochs)
- Total gradient evaluations: $\mathbb{E}[M_{B, 36}] = 180 \times 32 = 5,760$.
- Gradient surplus: Evaluates $5,760 - 5,660 = 100$ additional sample gradients ($+1.77\%$).
- Expected presentations per record: $\mathbb{E}[\text{count}(z_i)] = 180 \times \frac{32}{1,132} \approx 5.0883$.
- Sample coverage probability:
  $$\mathbb{P}(\text{sampled } \ge 1) = 1 - \left(1 - \frac{32}{1,132}\right)^{180} \approx 1 - 0.005697 = \mathbf{99.4303\%}$$

---

## 5. Comparative Synthesis

| Parameter / Metric | Option A ($q = 1/36$) | Option B.1 ($q = 32/1132, K=35$) | Option B.2 ($q = 32/1132, K=36$) |
| :--- | :---: | :---: | :---: |
| **Sampling Rate ($q$)** | $1/36 \approx 0.027778$ | $32/1,132 \approx 0.028269$ | $32/1,132 \approx 0.028269$ |
| **Expected Batch Size ($B_{\text{exp}}$)** | **$31.444444...$ (Float)** | **$32.0$ (Integer)** | **$32.0$ (Integer)** |
| **Steps per Epoch ($K$)** | 36 | 35 | 36 |
| **Total Steps ($T$)** | 180 | 175 | 180 |
| **Expected Cohort Presentations** | **5.0000 (Exact)** | 4.9470 ($-1.06\%$) | 5.0883 ($+1.77\%$) |
| **Expected Total Gradients** | **5,660** | 5,600 ($-60$) | 5,760 ($+100$) |
| **Probability Sampled $\ge 1$ Time** | **99.3791%** | **99.3404%** | **99.4303%** |
| **Opacus DataLoader Alignment** | **Native ($1 / \text{len}(\text{loader})$)** | Non-standard wrapper | Non-standard wrapper |

---

## 6. Formal Conclusion

1. **Retraction of Earlier Narrative:** The statement that Option B "excludes 12 patients per epoch" is statistically incorrect and is formally superseded by this document.
2. **Mathematical Equivalence of Privacy Guarantees:** Both Option A and Option B provide valid differential privacy guarantees under Renyi Differential Privacy (RDP) accounting, provided the accountant uses the exact sampling rate $q$ corresponding to that option.
3. **Justification of Option A:** Option A is preferred because setting $q = 1 / K$ guarantees that the expected number of sample exposures over $E$ epochs is identically $E \cdot N$, natively matches standard PyTorch/Opacus `DPDataLoader` iteration mechanics, and eliminates arbitrary integer truncation or oversampling.
