# SUMMARY.md: Executive Synthesis of Findings, Claim Adjustments, and Limitations

---

## 1. Headline Result in One Plain Sentence

> **Under a strictly invariant ~2.0M parameter edge budget ceiling, dividing neural capacity into intermediate compact sub-networks ($K = 2\text{--}4$) with orthogonal architectural inductive biases and dynamic Top-2 sparse routing outperforms monolithic parameter scaling in 12-lead ECG diagnosis, whereas over-fragmenting capacity ($K = 16$) erodes individual sub-network representation quality below clinical viability.**

---

## 2. Claim Status: Survived, Weakened, and Dropped

### Claims That Survived Intact (Confirmed by Data & Statistics)
1. **Multiplicity Superiority for Intermediate $K$ ($K = 2, 4$ vs. $K = 1$)**:
   * *Status*: **SURVIVED**.
   * *Evidence*: $K=2$ ($0.9085 \pm 0.0007$) and $K=4$ ($0.9084 \pm 0.0009$) statistically outperform the monolithic Single Large ($K=1$, $0.9059 \pm 0.0010$) with Holm-adjusted $p = 1.09 \times 10^{-3}$ and DeLong $p = 3.63 \times 10^{-4}$.
2. **Capacity Over-Fragmentation Boundary ($K = 16$)**:
   * *Status*: **SURVIVED**.
   * *Evidence*: Partitioning 2.0M parameters across 16 experts degrades performance to AUROC $0.9048 \pm 0.0005$ (worse than the monolithic baseline) while sequential latency triples to $6.78$~ms.
3. **Sparse Adaptive Routing Efficiency**:
   * *Status*: **SURVIVED**.
   * *Evidence*: Dynamic Top-2 routing matches dense ensemble discrimination ($0.9113$ vs. $0.9118$, Holm $p = 5.07 \times 10^{-5}$ vs. Single Large) while slashing inference compute from $414.9$ to $297.8$ MFLOPs (a verified **$28.2\%$ reduction**).
4. **Selective Deferral Risk Suppression**:
   * *Status*: **SURVIVED**.
   * *Evidence*: Deferring the $10\%$ most uncertain tracings to clinical review ($c=90\%$) achieves a statistically significant **$4.18\% \pm 2.02\%$ relative risk reduction** ($p = 9.34 \times 10^{-3}$).
5. **Catastrophic Gating Collapse Without Diversity Entropy**:
   * *Status*: **SURVIVED**.
   * *Evidence*: Factorial ablation proves that unregularized routers collapse into single experts ($0.9049 \pm 0.0017$, $p = 0.0480$ vs uniform baseline $0.9108$), whereas $\mathcal{L}_{\text{div}}$ prevents starvation and recovers AUROC to $0.9088 \pm 0.0015$.

---

### Claims That Were Weakened (Nuanced by Multiple Testing & Adjustments)
1. **Statistical Significance of Architectural Diversity Over Homogeneous Replication**:
   * *Original Pilot Claim*: Heterogeneous inductive biases statistically outperform homogeneous ResNet replication ($p < 0.05$).
   * *Weakened Realized Claim*: **Consistent empirical trend and variance reduction**.
   * *Reason*: While raw paired $t$-tests yield $p = 0.0164$ for AUROC and $p = 0.0144$ for AUPRC, family-wise Holm adjustment across the 4 endpoints yields **$p = 0.0577$**. The claim was weakened from "statistically significant superiority at $\alpha=0.05$" to a consistent numerical improvement with lower representational similarity (linear CKA $0.9008$ vs. $0.9458$).

---

### Claims That Were Dropped (Removed in Strict Compliance with Hard Rules)
1. **"External Multi-Center Domain Shift Validation"**:
   * *Dropped*: Re-evaluating on the $s \ne 0$ subset ($N=248$) of PTB-XL Fold 10 was originally described as external validation. This was dropped because it evaluates the same parent repository under alternative acquisition devices.
   * *Replacement*: Renamed to **"Site-Subgroup Sensitivity Analysis"**, with true cross-hospital multi-center validation explicitly designated as an open limitation.
2. **"Novelty of Mixture-of-Experts or Ensembles"**:
   * *Dropped*: In strict compliance with Proposal §2 (Rule 6), all novelty claims regarding the MoE architecture, gating, or ensembling were discarded. Novelty is strictly restricted to the controlled empirical scaling evaluation under fixed budgets.
3. **"Unconstrained 32M Server-Scale Generalizability"**:
   * *Dropped*: Claims extending findings to 32M foundation models were dropped. In compliance with Rule 5, the study is formally scoped to the edge telemetry budget ($\sim 2.0\text{M}$ parameters), and foundation-scale scaling is listed as a future research question.
