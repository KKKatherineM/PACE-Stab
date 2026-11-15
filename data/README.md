# Released datasets

The CSVs contain the **final filtered populations used in evaluation**, rather
than the complete unfiltered source datasets. No feature matrices, prediction
outputs, or intermediate experiment directories are included.

## Contents

| Directory | Samples | Population |
| --- | ---: | --- |
| `benchmarks/s461` | 461 | Frozen regular benchmark |
| `benchmarks/s783` | 783 | Frozen regular benchmark |
| `benchmarks/s2648` | 2,648 | Frozen regular benchmark |
| `benchmarks/fireprot_hf` | 2,578 | Accepted FireProt population |
| `benchmarks/ssym_direct` | 342 | Direct mutations |
| `benchmarks/ssym_inverse` | 342 | Inverse mutations, using their actual mutant structures |
| `benchmarks/s8754` | 8,184 | Final exact-matched population; excludes the historical 8,192-row parent |
| `benchmarks/s571` | 519 | Final common unique inputs, representing 531 experimental observations |
| `benchmarks/s4346` | 3,500 | Final common unique inputs, representing 3,963 experimental observations |
| `applications/snase` | 559 | Frozen Figure 6 direct-stability observations |
| `applications/g6pd` | 8,804 | Frozen structure-compatible G6PD missense inputs |

The previously released MegaScale train/validation/test and S669 tables remain
in `manifests/`. The S669 structures remain in `s669_structures/` (Pythia-provided
AlphaFold models). MegaScale PDBs are included in `megascale/structures.tar.gz`:
212 training, 29 validation, and 28 test structures, 269 unique files in total.
These are the exact PDB bytes referenced by the original split feature
manifests, selected using the released final train/validation/test CSVs.
From the repository root, run:

```bash
tar -xzf data/megascale/structures.tar.gz -C data/megascale
python scripts/prepare_training_cache.py --device cuda:0
```

The training-cache script defaults to `data/megascale/structures/`; an explicit
`--megascale-pdb-dir` can still override it. No feature caches are bundled.

## Input format

Prediction CSVs contain `observation_uid`, `protein_id`, `position0`, `wt_aa`,
`mut_aa`, `wt_sequence`, `mut_sequence`, `structure_path`, and `chain_id`.
Positions are zero-based. Structure paths are relative to the repository root.
Mutation identity and structure mapping are validated. Labels are not used
for feature generation or inference. Both `.csv` and `.csv.gz` are supported.

## Data sources and references

The table distinguishes dataset publications from the files and processing
used by PACE-Stab. A download from SPURS or Pythia does not make those projects
the original producers of the experimental measurements. For a curated
collection, cite its dataset publication and retain the underlying experimental
references supplied by that collection.

| Dataset | Dataset publication / upstream collection | Version used by PACE-Stab |
| --- | --- | --- |
| MegaScale | [Tsuboyama et al., Nature (2023)](https://doi.org/10.1038/s41586-023-06328-6) | SPURS split and filtering protocol; 190,427 / 25,909 / 28,172 train / validation / test observations and their 269 PDBs |
| S669 | [Pancotti et al., Briefings in Bioinformatics (2022)](https://doi.org/10.1093/bib/bbab555) | 669 benchmark records; 94 Pythia-provided AlphaFold structures, rather than the experimental-structure version |
| S461 | [Hernández et al., Bioinformatics (2023), KORPM](https://doi.org/10.1093/bioinformatics/btad011); curated from S669 | SPURS-distributed GeoStab cleaned files; 461 observations |
| S2648 | [Dehouck et al., Bioinformatics (2009), PoPMuSiC-2.0](https://doi.org/10.1093/bioinformatics/btp445) | SPURS-distributed cleaned files; 2,648 observations |
| S783 | [GeoStab data collection](https://github.com/Gonglab-THU/GeoStab#data) and [Xu, Liu and Gong, Nature Computational Science (2024)](https://doi.org/10.1038/s43588-024-00716-2) | SPURS-distributed GeoStab cleaned S783; 783 observations |
| S8754 | [GeoStab, Xu et al. (2024)](https://doi.org/10.1038/s43588-024-00716-2), collecting experimental stability records | SPURS-distributed GeoStab files; final 8,184-observation exact-matched subset |
| S571 / S4346 | [GeoStab ΔTm collection and publication](https://doi.org/10.1038/s43588-024-00716-2) | Final source-backed common inputs: 519 / 3,500; PDB bytes from the actual inference manifests, including the local ProStab data copy |
| FireProt | [FireProtDB: database of manually curated protein stability data](https://doi.org/10.1093/nar/gkaa981) | SPURS-distributed FireProt data; 2,578 accepted observations; this is a frozen subset, not the latest full FireProtDB release |
| SSym direct / inverse | [Pucci et al., Quantification of biases in predictions of protein stability changes upon mutations (2018)](https://academic.oup.com/bioinformatics/article/34/21/3659/4987874) | SPURS-distributed SSym files; 342 observations in each direction and the corresponding structures |
| SNase | [FireProt 2.0](https://doi.org/10.1093/bib/bbad425); case observations as distributed in [IFUM Figure 3b source data](https://doi.org/10.1038/s41467-026-68637-4) | 559 retained worksheet observations; PDB 1EY0 chain A; experimental labels converted once to the PACE ΔΔG convention |
| G6PD | [Geck et al., Evidence for G6PD variant classification from multiplexed functional assays (2026)](https://doi.org/10.1186/s13059-026-04105-y), supplementary Table S3 | 8,804 structure-compatible inputs; endpoint labels joined by exact HGVS; PDB 7SNG chain A |

Distribution and processing references: [SPURS repository](https://github.com/luo-group/SPURS),
[SPURS data archive](https://huggingface.co/cyclization9/SPURS/blob/main/data.tar.gz),
[Pythia repository](https://github.com/Wublab/Pythia), and
[GeoStab repository](https://github.com/Gonglab-THU/GeoStab).
Experimental structure identifiers refer to entries in the
[Protein Data Bank](https://www.rcsb.org/). The released PDB files preserve the
actual input versions; do not silently replace them with newly downloaded or
repaired structures.

## Using an included dataset

Each new dataset directory contains:

- `samples.csv.gz`: mutation identity, sequences, structure mapping, and labels.
  Pandas and the release CLIs read this compressed CSV directly.
- `structures.tar.gz`: only the PDBs referenced by that CSV, with original bytes
  preserved. Identical PDB content is stored once within a dataset.
- For S571/S4346, `observations.csv.gz`: the final retained experimental records,
  their pH, labels, and mapping to `model_input_uid`.

From the repository root, for example:

```bash
tar -xzf data/benchmarks/s461/structures.tar.gz -C data/benchmarks/s461
python scripts/build_features.py --input data/benchmarks/s461/samples.csv.gz --output cache/s461 --device cuda:0
python predict.py --cache cache/s461 --output results/s461.csv --device cuda:0
python evaluate.py --predictions results/s461.csv --labels data/benchmarks/s461/samples.csv.gz --output results/s461_metrics.json
```

Extract each archive into **its own dataset directory**, as above. CSV
`structure_path` values are relative to the repository root and point to the
extracted `structures/` directory. No original experiment directory is needed.
Extracted PDB copies are ignored by Git; keep the compressed archive in the
release. To view a CSV without compression, use `gzip -dc samples.csv.gz`.

## Labels and exact sample identity

- ΔΔG benchmarks and SNase use `target_ddg`: negative means stabilizing. Preserve
  the stored sign. Experimental replicates remain separate observations;
  do not deduplicate mutation strings or renumber observation identifiers.
- S8754 uses the final 8,184-UID exact-matched subset. Its sequences,
  structures and labels are selected from the frozen structure-available
  parent by exact UID, without reapplying a performance-based filter.
- S571/S4346 use `target_dtm`, **not ΔΔG**. In `samples.csv.gz`, each label is the
  mean experimental ΔTm across the retained records sharing `model_input_uid`;
  `observation_multiplicity` records their count. The observation-level labels
  remain in `observations.csv.gz`. S4346 follows `S4346-3500-unique-v1`.
  For ΔTm transfer, correlate **negative predicted ΔΔG** with experimental ΔTm;
  do not run the ΔΔG RMSE/MAE evaluator on these endpoints.
- G6PD retains the original `abund_score`, `activity_score`, `functional_score`,
  `ClinVar_class`, `WHO_class_2022`, `phenotype`, and
  `activity_RBC_literature_avg` fields from the frozen Table S3 label join.
  Exact HGVS matching and duplicate-conflict handling follow the original
  evaluation: conflicting or absent endpoint values remain missing. Thus
  endpoint-specific sample counts can be smaller than 8,804. Functional scores
  are not ΔΔG labels; higher scores mean more normal abundance/activity/function.
- `position0` indexes the supplied WT sequence; `embedding_position` records the
  structure mapping used in the original inputs. It is not necessarily the
  PDB author residue number. Preserve the supplied chain and sequence.
- FireProt's `source_row_index` is the original source-table row, whereas
  `source_record_index` identifies the frozen accepted population. Both are
  preserved; they should not be substituted for one another.

Packaged sample identities and labels were checked against the final evaluation
populations. Archived PDBs were checked byte-for-byte against their referenced
source files, and every CSV structure reference is covered.
