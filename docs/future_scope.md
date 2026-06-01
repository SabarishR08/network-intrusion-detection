# Future Scope

## Short-Term Enhancements

- Add attack subtype classification instead of binary output
- Add feature selection and explainability plots for analysts
- Add alert thresholds and severity levels
- Build a reusable REST API on top of the trained pipeline

## SOC Integration

The next stage of this project can forward anomaly predictions into a Security Operations Center workflow. Examples include queueing suspicious rows, generating analyst alerts, and correlating traffic anomalies with host-based indicators.

## Real-Time IDS Possibilities

- Stream packet metadata from a network tap
- Score traffic in batches every few seconds
- Save alerts into a security log store
- Surface the output in a monitoring dashboard

## IBM / LangFlow Extension

The ML detector can remain the core engine while a LangFlow workflow routes predictions into an IBM Granite explanation agent. That agent can summarize the reason a row was flagged and suggest response actions.
