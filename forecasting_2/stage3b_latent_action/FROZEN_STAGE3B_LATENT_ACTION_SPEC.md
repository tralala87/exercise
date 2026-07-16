# Frozen Stage 3B LATENT-ACTION contrastive-pretraining gate

Frozen on 2026-07-16 before Stage 3B development or confirmation outcomes.

## Scientific question

At exactly matched active parameters, initialization, task streams, optimizer updates, and forecasting heads, does explicit action-trajectory contrastive pretraining with mechanism-conditioned latent slots make useful alternative forecast paths identifiable enough to safely close material action-oracle headroom?

## Causal arms

1. `forecast_only` — forecasting objective only; capacity-matched critic remains inert.
2. `detached_contrastive` — the identical critic and action encoder train, but critic gradients cannot update temporal tokens, mechanism slots, the pooled process state, or the default forecast.
3. `joint_contrastive` — true counterfactual action labels, ranking targets, trajectory/slot alignment, and advantage losses reshape the shared temporal representation.
4. `shuffled_joint` — identical joint architecture and compute, with counterfactual labels permuted across tasks.

All arms begin from bitwise-identical state dictionaries and receive the same minibatches in the same order.

## Architecture intervention

The temporal encoder produces six learned mechanism slots from typed level, difference, absolute-change, observation-mask, missing-age, and local-mean channels. Each fixed forecast action is represented by its complete candidate trajectory, its difference from the neural/compiled default, two legal rolling-origin historical backtests, an action identity embedding, and forecast geometry. The action query attends over mechanism slots before a shared critic predicts:

- five realized action/default MASE-ratio classes;
- signed clipped log-regret;
- a listwise action score;
- a latent action-competence representation.

The treatment objective combines probabilistic forecast loss, categorical and binary win/harm losses, signed-regret regression, listwise best-action contrast, best/worst ranking, trajectory-slot alignment, and slot diversity. Critic gradients are warmed in and bounded by a frozen coefficient.

## Data and firewall

Training worlds cover supported level, trend, seasonality, autoregression and pairs; held-out compositions; irregular/missing observations; long horizons; changepoints; saturation; chirps; and amplitude modulation. Candidate backtests are computed entirely inside observed history. Final futures provide training labels and scoring only. Family, source, seed, task identity, and future values are forbidden model features.

Development and calibration use disjoint namespaces. The confirmation namespace is never generated or scored unless every development gate passes.

## Frozen development gates

All are required:

1. useful-win AUC >= 0.78;
2. useful-win AUC gain over detached >= 0.06;
3. severe-harm AUC >= 0.80;
4. recall >= 45% on tasks with an alternative at least 20% better;
5. realized switch win rate >= 68%;
6. severe wrong-switch rate < 5%;
7. intervention rate in [5%, 20%];
8. long/irregular/misspecified task-p95 MASE reduction >= 10%;
9. supported-ID regression <= 1.5%;
10. compositional-OOD regression <= 1.5%;
11. close >= 20% of default-to-hindsight-oracle MASE gap;
12. default forecast regression versus forecast-only <= 1%;
13. useful-win AUC gain over shuffled control >= 0.04;
14. latent linear-probe AUC gain over detached features >= 0.05;
15. exact matched parameters, initialization, data, updates, replay, and future-information firewall.

The machine-readable implementation splits the intervention-rate range into lower and upper gates, yielding 16 recorded gate rows.

## Decision rule

A complete development pass opens the frozen confirmation namespace. Any development failure keeps confirmation sealed and authorizes no further post-hoc selector or threshold ladder. The next repair must alter the executable mechanism/action basis or return to process-grammar expansion.
