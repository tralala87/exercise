# Stage 2I canonical source relay

This pull-request-only workflow performs the frozen canonical source acquisition outside the model runtime. It uses only the exact M3/M4 URLs in `STAGE2I_PRE_VALUE_MANIFEST.json`, validates the manifest hash before downloading, records every source hash, and either emits the deterministic 700-task source suite or a fail-closed acquisition diagnostic.

No model checkpoint, forecast policy, threshold, or outcome is present in this relay. A source failure never triggers mirror substitution.
