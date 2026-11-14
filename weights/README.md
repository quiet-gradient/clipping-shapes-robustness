# Evaluation weights

Download the `.pt` assets from the `weights-v1` GitHub release into
`weights/files/`. The root README gives verification and loading commands.

- `manifest.json`: 52 original GTSRB assets (public initialization and 51 evaluated models).
- `additional_manifest.json`: 82 CIFAR assets (public initialization, 54 original test models, 24 additional test models and three dev-selection models), plus six additional GTSRB fixed-norm models.

The manifests map each weight file to its run, seed, cohort and source/export
SHA-256 digests. All exports contain model tensors and loading metadata;
optimizer, RNG and private-training records are excluded. They support
inference and attack evaluation rather than training resume. CIFAR's public
initialization has 100 output classes; its fine-tuned models have 10.

Non-private baselines and zero-noise controls do not have private-training
guarantees. Accounting for individual private runs does not certify the joint
release of the model collection.
