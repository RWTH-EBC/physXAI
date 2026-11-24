# Review of the old branches before physXAI 1.0.0 (2026-11)

Before the release of 1.0.0, all open feature branches were reviewed against the refactored code base
(`refactoring_v1.0.0`). No code was taken over: useful ideas were re-implemented in the 1.0 architecture,
methodological errors were not carried over. **All branches below can be deleted.** Branches deleted since the
review are no longer listed.

Ancestor check: `git merge-base --is-ancestor origin/<branch> origin/refactoring_v1.0.0`.

| Branch | In 1.0? | Result |
|---|---|---|
| `26-integrate-modular-anns` | yes (ancestor) | fully contained; the refactoring started from it |
| `45-add-interpolation-methods-for-input-output` | no | idea kept for later (see below) |
| `67-pinns` | no | ideas re-implemented (PINN loss, trainable residuals) |

## 26-integrate-modular-anns
Its tip is an ancestor of `refactoring_v1.0.0`. The modular expressions, `closed_loop` and `init_layer` were replaced
on purpose (see CHANGELOG, *Removed*); the constrained regression was kept and rebuilt as `ModularANN(solver='qp')`.

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
