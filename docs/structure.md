# Project Structure

```
core/                  Training engine, problem-agnostic (NumPy only)
  activations.py       Step, Identity, Tanh, Sigmoid, ReLU, LeakyReLU, Softmax
  initializers.py      Uniform, Normal, Xavier, He
  layers.py            Dense layer: forward / backward
  losses.py            MSE, BinaryCrossEntropy, CategoricalCrossEntropy
  optimizers.py        GD, Momentum, RMSProp, Adam, AdaGrad
  network.py           Network: MLP with predict / backward / fit / evaluate
  history.py           Per-epoch training log, CSV export
  callbacks.py         PrintProgress, EarlyStopping, AdaptiveEta, StopAtTarget,
                       ExponentialDecay, StepDecay
  metrics.py           Wrong pixels after binarizing: per sample, max, mean
  pca.py               PCA (NumPy SVD): fit / transform / inverse_transform
  augmentation.py      GaussianNoise, RandomShift, BitFlip, Compose (train batches only)
  autoencoder.py       Autoencoder(encoder, decoder): encode / decode / fit
  vae.py               VariationalAutoencoder(Autoencoder): stochastic latent layer + KL term
ej1/                   Exercise 1a: basic autoencoder on font.h (spec: docs/context/ej1-autoencoder.md)
                       and 1b: denoising autoencoder (spec: docs/context/ej1b-denoising.md)
  datos.py             cargar_font(): X (32, 35) in {0, 1} and the 32 labels
  config.json          Sweep: base config, baselines, per-step grids, seeds, max epochs
  barrido.py           Sequential sweep; writes only to resultados/ej1/
  graficos.py          Figures; reads only from resultados/ej1/, writes resultados/ej1/figuras/
  config_dae.json      1b: base config, noise levels, noisy val/test seeds, experiment grids
  dae.py               1b experiments; writes only to resultados/ej1b/
  graficos_dae.py      1b figures; reads only from resultados/ej1b/, writes resultados/ej1b/figuras/
ej2/                   Exercise 2: VAE on four datasets (spec: docs/context/ej2-vae.md)
  datos.py             Loads processed .npz; splits and the flag scaler (train only)
  banderas.py          Tolerant flags CSV parser + render_flag (figures only)
  config.json          Sweep grid and per-dataset base configuration
  modelos.py           Image VAE, tabular VAE and deterministic AE builders
  vae_exp.py           Compact sweep and final training; writes only to resultados/ej2/
  graficos.py          Figures; reads only from resultados/ej2/, writes resultados/ej2/<dataset>/figuras/
tests/                 pytest suite, one file per core module (test_ej1_*.py for ej1/)
data/                  Datasets (font.h, flags_tricolor.csv, raw images and chineseMNIST.csv)
data/procesadas/       Processed .npz, low-resolution PNGs and F1 manifests (committed)
resultados/ej1/        Sweep output: one CSV per step, elecciones.json, final/ (weights, history)
resultados/ej1b/       1b output: one CSV per experiment, elecciones.json, final/ (weights, test.csv)
resultados/ej2/        VAE output: dedup manifests, sweep CSVs, final/ and latent_2/16 (weights, figures)
utils/                 Standalone data scripts (excluded from ruff)
docs/context/          Assignment, course notes and exercise specs (source of truth)
check-cicd.py          Local CI: creates .venv, installs requirements, runs all checks
.github/workflows/     Same checks on every push and pull request
```

`core/` was ported from the TP3 engine. It contains nothing specific to an exercise.

## Conventions

- Arrays are `(n_samples, n_features)`; weights are `W (n_in, n_out)` and bias `b (1, n_out)`.
- All randomness comes from an explicit `np.random.Generator` passed to the `Network`.
- Activations, initializers, losses and optimizers can be built by name
  (`get_activation("tanh", beta=2.0)`) or passed as objects.
- Losses average over samples, so gradients already carry the `1/n` factor.
- Code, comments and test names are in Spanish.

## Autoencoder

```python
rng = np.random.default_rng(seed)
encoder = Network([n_features, 16, 2], output_activation="identity", rng=rng)
decoder = Network([2, 16, n_features], output_activation="sigmoid", rng=rng)
ae = Autoencoder(encoder, decoder)

ae.fit(X, optimizer=Adam(), loss=MSE(), epochs=5000)  # target defaults to X
Z = ae.encode(X)  # (n, 2)
X_rec = ae.decode(Z)  # (n, n_features)
```

`Autoencoder` is a `Network` whose layers are the encoder's followed by the decoder's, so
every `Network.fit` argument (batch size, validation, callbacks, metrics, l2, augment) applies.

## Variational autoencoder

```python
encoder = Network([n_features, 16, 2 * 2], output_activation="identity", rng=rng)  # [mu, log var]
decoder = Network([2, 16, n_features], output_activation="sigmoid", rng=rng)
vae = VariationalAutoencoder(encoder, decoder, kl_weight=1.0)

vae.fit(X, optimizer=Adam(), loss=MSE(), epochs=5000)  # train_loss is L + kl_weight * mean KL
mu = vae.encode(X)  # mean of q(z/x), no sampling
new = vae.sample(10)  # z ~ N(0, I) through the decoder
```

`predict` is stochastic (samples z); the noise-free reconstruction is `decode(encode(X))`.
The KL term is averaged over samples (not over samples and features), the standard β-VAE
objective.

## Commands

- Full local CI: `python3 check-cicd.py`
- Lint and format: `ruff check . && ruff format --check .`
- Tests: `python3 -m pytest -q`
- Exercise 1a sweep: `python3 -m ej1.barrido` (resumable: steps whose CSV exists are skipped)
- Exercise 1a figures: `python3 -m ej1.graficos`
- Exercise 1b experiments: `python3 -m ej1.dae` (resumable, like the 1a sweep)
- Exercise 1b figures: `python3 -m ej1.graficos_dae`
- Exercise 2 data: `python3 utils/preparar_ej2.py --todo` (dedup first: `--dedup-f1`)
- Exercise 2 sweep: `python3 -m ej2.vae_exp --dataset nba --fase barrido`
- Exercise 2 final models: `python3 -m ej2.vae_exp --dataset nba --fase final`
- Exercise 2 figures: `python3 -m ej2.graficos`
