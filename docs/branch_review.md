# Review of the old branches before physXAI 1.0.0 (2026-11)

Before the release of 1.0.0, all open feature branches were reviewed against the refactored code base
(`refactoring_v1.0.0`). No code was taken over: useful ideas were re-implemented in the 1.0 architecture,
methodological errors were not carried over. **All branches below can be deleted.**

Ancestor check: `git merge-base --is-ancestor origin/<branch> origin/refactoring_v1.0.0`.

| Branch | In 1.0? | Result |
|---|---|---|
| `26-integrate-modular-anns` | yes (ancestor) | fully contained; the refactoring started from it |
| `14-integrate-developments-from-gitlab` | no | covered; one bug fixed in 1.0 |
| `58-preprocessingmultistep-avoid-data-leakage-with-overlapping-training-and-test-windows` | no | superseded by `BlockSplit` |
| `simplify-executables-structure` | no | not adopted |
| `45-add-interpolation-methods-for-input-output` | no | idea kept for later (see below) |
| `18-integrate-physically-consistent-multi-step-models` | no | PCNN re-implemented after the paper |
| `67-pinns` | no | ideas re-implemented (PINN loss, trainable residuals) |

## 26-integrate-modular-anns
Its tip is an ancestor of `refactoring_v1.0.0`. The modular expressions, `closed_loop` and `init_layer` were replaced
on purpose (see CHANGELOG, *Removed*); `constrained_regression.py` was kept and fixed.

## 14-integrate-developments-from-gitlab
- `MultiStepModel.evaluate(..., clipped_warmup_length)`: in 1.0 the warmup is never part of loss or metrics.
- `test_size=0`: the multi step part was covered. For single step data 1.0 still returned empty test arrays and
  predicted on them; **fixed in 1.0** (missing splits are `None` and skipped everywhere, tested).
- The branch cut `td.y_*` in place (twice on a second evaluation).

## 58-…-avoid-data-leakage-with-overlapping-training-and-test-windows
`_get_stratified_block_indices` shuffled block ids (no stratification despite the name, global random state), had no
embargo between neighbouring blocks of different splits and concatenated the training blocks in time order. 1.0's
`BlockSplit` (stratified blocks, embargo, seeded generator, windows never cross a split boundary) covers and
exceeds it.

## simplify-executables-structure
A `TestCase` helper inside the library (`physXAI/utils`), run names from the caller's stack frame, shared mutable
`Feature` objects between examples and `from configuration import *`, built on the removed `Logger` and constructed
features. 1.0's examples are short (`Run`, preprocessing, model, `run.save`) and deliberately self-contained.
Not adopted.

## 45-add-interpolation-methods-for-input-output
**Idea worth keeping:** sample controls and disturbances as **interval means** over the time step (zero order hold
of the control), states as point values. Not adopted as is: float modulo on the time index (fails for datetime),
O(N²) loops, silent NaN handling, global mutable defaults, and the output `shift` was removed from the config, which
AgentLib-MPC relies on. Design for a later version:

- `aggregation={'u': 'mean', ...}` in `PreprocessingData`, default `'sample'`; time weighted mean per step (zero
  order hold integral), NaN if less than ~80 % of the interval is covered.
- Inputs labelled at the start of their interval `[t_k, t_k+1)`, so `shift=1` and the AgentLib-MPC contract stay.
- Error if a state or recursive feature is aggregated (averaging T over the next interval leaks the target).
- Lags and `Change` derived after the aggregation; forecasts of disturbances in the MPC then also interval means.

## 18-integrate-physically-consistent-multi-step-models
Methodological errors found:
- **Data leakage:** the example fed `T_amb − T` with the *measured* room temperature over the whole 48 step horizon,
  so the multi step error was too optimistic and the model not usable in an MPC.
- No feedback of the predicted temperature (pure integrator without temperature dependent losses).
- Sign constraints on z-scored inputs with a free bias: "w ≥ 0" had no physical meaning; the nonlinear part was
  unconstrained.
- Double normalization in the sub networks; monotonicity constraints mapped by position, not by name.

**Re-implemented:** `PCNNModel` after Di Natale et al. (2022), with the energy accumulator and the unforced
dynamics as states, the predicted temperature fed back, provable consistency, warmup and export like an RNN.
**Not adopted (idea for later):** a generic monotone RNN (sign patterns in one SimpleRNN; the guarantee only holds
in open loop, the PCNN covers the closed loop case).

## 67-pinns
12 commits of a student project (~10k lines), building specific (hard coded columns and RC values), largely
duplicated, without tests, based on the `Logger`. Methodological findings:
- 1R1C residual with an explicit Euler step over the whole time step (inaccurate, sign flip for short time
  constants), non-identifiable parameter pairs, sigmoid parametrization frozen at its initial value, physics using
  inputs the network does not see, inconsistent units, RC parameters learned only through the physics term
  (co-adaptation with the network).
- PhysNet latent wall temperature: with only the air node equation the latent output can satisfy the physics
  exactly, the physics loss constrains nothing.
- TABS loss with a future measurement as model input; pairs of consecutive samples built without time check.

**Re-implemented:**
- PINN loss: terms normalized by the variance of the training targets, no averaging over terms, per-term metrics,
  early stopping on the validation data loss; one model output (exportable).
- Trainable physics residuals (`PhysicsResidual`, `px.positive`, `px.Prediction`), exact zero order hold
  discretization in the example, calibration of the parameters on the data before the network is trained.

**Ideas for later:** consistency loss on a latent state (e.g. wall temperature) in window or recurrent training,
`stratify_by` for `BlockSplit` (e.g. weekends), adaptive weighting of loss terms (not used: the weight encodes the
trust in the physics).
