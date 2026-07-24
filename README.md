# Modern Restoration of Z-Lin for CIFAR-10

This repository provides a TensorFlow 2.19 restoration of the Z-Lin CIFAR-10 training pipeline originally released by Zhouhan Lin. It accompanies the DEARING 2026 workshop paper:

> *Restoring the Z-Lin Permutation-Invariant CIFAR-10 Benchmark: A Reproducibility Audit under Modern Framework Semantics*

## Purpose

The repository supports the reproducibility audit reported in the accompanying paper. Its primary objective is to reconstruct the released executable benchmark specification in a modern Python and TensorFlow environment.

The implementation includes:

- contrast normalization and PCA whitening;
- the released 800-component PCA representation;
- alternating zero-bias autoencoder and linear bottleneck layers;
- layer-wise unsupervised pre-training;
- multinomial logistic-regression classifier fitting;
- supervised end-to-end fine-tuning;
- optional geometric data augmentation;
- controlled variants of dropout and optimizer semantics.

The original paper describes PCA whitening with 99% variance retention, whereas the released reference scripts use a fixed representation of 800 PCA components. The reported reproduction runs follow the released executable-code specification. The 99%-variance rule was evaluated as a specification fork and did not change the regime-level recoverability conclusion.

## Relation to the Original Implementation

This repository is not the original Z-Lin implementation. It is a modern restoration intended to make the historical pipeline executable and inspectable under current software frameworks.

Compared with the historical codebase, this implementation:

- removes the dependency on NeuroBricks;
- replaces the legacy Theano and Python 2 stack with TensorFlow 2.19 and Python 3;
- preserves the staged training structure where it can be reconstructed;
- distinguishes paper-text fidelity, executable-code fidelity, and controlled modernization;
- documents benchmark-relevant paper-code and framework-level divergences;
- provides a standalone implementation for execution and inspection.

Framework-specific behavior is not assumed to be identical to the original Theano environment. In particular, dropout semantics, random-number streams, update ordering, and numerical execution may differ.

## Repository Structure

The main entry point is:

```text
main.py
```

The script trains a permutation-invariant, fully connected model on CIFAR-10 and writes intermediate and final training artifacts.

## Installation

Create a clean Python environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The minimal pinned dependencies are:

```text
tensorflow==2.19.0
numpy==2.1.3
scipy==1.15.2
scikit-learn==1.6.1
```

## Running the Pipeline

Run the main script from the repository root:

```bash
python main.py
```

Depending on the selected configuration, execution may require substantial memory and compute time because the historical architectures contain large fully connected layers and use staged pre-training.

## Outputs

The implementation may write artifacts including:

- layer-wise pre-training checkpoints;
- linear bottleneck checkpoints;
- classifier initialization results;
- fine-tuned model weights;
- final model parameter snapshots;
- preprocessing and PCA artifacts.

Output paths and enabled training stages are defined in the implementation and experiment configuration.

## Interpretation of Results

The repository distinguishes between three classes of results:

1. **Paper-text fidelity:** settings reconstructed from the methodological description in the original paper.
2. **Executable-code fidelity:** settings reconstructed from the released reference implementation.
3. **Controlled modernization:** intentional changes required for, or enabled by, the modern TensorFlow environment.

Results from controlled modern variants should not be interpreted as exact reproductions of the historical benchmark.

The accompanying audit finds that recoverability is regime-dependent. The augmented Z-Lin configuration is recovered under the executable-code specification, whereas the deeper non-augmented configuration remains below the historical reference.

The comparison between these historical regimes does not isolate the effect of data augmentation because the augmented and non-augmented benchmarks use different reference architectures.

## Original Work

Zhouhan Lin, Roland Memisevic, and Kishore Konda:

> *How Far Can We Go Without Convolution: Improving Fully-Connected Networks*  
> International Conference on Learning Representations, 2016  
> arXiv:1511.02580

## Disclaimer

This repository is provided as a research artifact for academic reproducibility and benchmark-restoration work.

Results may depend on:

- package and framework versions;
- preprocessing and PCA choices;
- random seeds;
- dropout semantics;
- hardware and numerical kernels;
- optimizer and update semantics;
- the ordering of staged training operations.

Results obtained under other environments or modified configurations may therefore require independent validation.

## Citation

When using this repository, please cite both the original Z-Lin paper and the accompanying DEARING 2026 workshop paper.

A BibTeX entry for the workshop paper will be added when the final proceedings metadata becomes available.
````
