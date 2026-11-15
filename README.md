# PACE-Stab

PACE-Stab predicts protein stability changes (ΔΔG) caused by single amino-acid
mutations. It combines a perturbation-centered P-expert and a context-centered
C-expert, averaging three random seeds (42, 2024, and 3407) from each family:

```text
PACE-Stab = (P42 + P2024 + P3407 + C42 + C2024 + C3407) / 6
```

Negative predicted ΔΔG indicates stabilization.

## Installation

```bash
conda env create -f environment.yml
conda activate pace-stab
```

The environment uses Python 3.10, PyTorch 1.12.1, and fair-esm 2.0.0.
Run the commands below from the repository root.

## Model weights

PACE-trained checkpoints are stored in `checkpoints/P*/` and `checkpoints/C*/`.
Keep `checkpoints/training_prior/` alongside them; it contains the small site
fusion heads needed to initialize the P-expert.

ESM2 and Pythia foundation weights are downloaded separately:

```bash
python scripts/download_pretrained.py --list   # show official download URLs
python scripts/download_pretrained.py          # download and verify weights
```

The script saves ESM2 and its contact-regression weights to `checkpoints/esm/`,
and Pythia-p/Pythia-c to `checkpoints/pythia/`.
See [the download configuration](configs/pretrained.json) for individual URLs.
Existing weights can be selected using `PACE_ESM2`, `PACE_PYTHIA_P`, and
`PACE_PYTHIA_C`; keep the ESM contact-regression file beside its ESM2 checkpoint.

## Prediction

The example provides two mutations and one PDB structure:

```bash
python scripts/build_features.py --input data/inference_example/mutations.csv --output cache/example --device cuda:0
python predict.py --cache cache/example --output results/example.csv --device cuda:0
```

Use `--device cpu` to run on CPU, or add `--member P42` to `predict.py` to use
one ensemble member. Input format and structure preparation are described in
[data/README.md](data/README.md#input-format).

## Data

The release includes the final filtered MegaScale splits, external benchmarks,
G6PD/SNase inputs, and the PDB structures used for these samples.
Dataset counts, extraction instructions, label conventions, and original
references are listed in [data/README.md](data/README.md).

We use the SPURS data protocol and distributed benchmark resources, with
Pythia-provided AlphaFold structures for S669. These projects are processing
and distribution sources; the original datasets are credited separately in
[the data-source table](data/README.md#data-sources-and-references).

## Training

Extract the included MegaScale structures, build feature caches, and train:

```bash
tar -xzf data/megascale/structures.tar.gz -C data/megascale
python scripts/prepare_training_cache.py --device cuda:0
python pace/p_expert/train.py --seed 42 --no-amp --device cuda:0
python pace/c_expert/train.py --seed 42 --device cuda:0
```

Repeat with seeds 2024 and 3407 to train all six members. Settings are in
[the P-expert configuration](pace/p_expert/config.yaml) and
[the C-expert configuration](pace/c_expert/config.yaml).
Feature generation requires substantial compute and tens of GiB of local
storage. Generated caches and results are ignored by Git.

## Evaluation

For example, after predicting S461 using the commands in the
[data guide](data/README.md#using-an-included-dataset):

```bash
python evaluate.py --predictions results/s461.csv --labels data/benchmarks/s461/samples.csv.gz --output results/s461_metrics.json
```

The evaluator aligns `observation_uid` and `target_ddg` and reports Spearman,
Pearson, RMSE, and MAE. ΔTm and G6PD functional endpoints use different label
conventions; see the data guide. Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

## References and acknowledgements

We acknowledge the models, software, and data-processing resources provided by:

- [ESM2](https://github.com/facebookresearch/esm): Lin et al.,
  [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://doi.org/10.1126/science.ade2574), Science (2023).
- [Pythia](https://github.com/Wublab/Pythia): Sun et al.,
  *Structure-based self-supervised learning enables ultrafast protein stability prediction upon mutation*, The Innovation (2025).
- [SPURS](https://github.com/luo-group/SPURS): Li and Luo,
  [Generalizable and scalable protein stability prediction with rewired protein generative models](https://doi.org/10.1038/s41467-025-67609-4), Nature Communications (2025).

Please also cite the original publications for the datasets you use, listed
in [data/README.md](data/README.md#data-sources-and-references).
The bundled Pythia code retains its [upstream license](third_party/pythia_structure/LICENSE).
