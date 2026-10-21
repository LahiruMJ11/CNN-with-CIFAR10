# CNN with CIFAR-10

PyTorch implementation and experimental study of backpropagation, optimizers, regularization, data augmentation, and residual CNNs on CIFAR-10. This repository contains the work for **EN4554 Assignment 2: Convolutional Neural Networks**.

The complete report, theory, implementation, plots, tables, interpretations, and saved outputs are contained in one reproducible Jupyter notebook:

**[Open the completed notebook](./CNN%20with%20CIFAR10.ipynb)**

## Final result

The validation-selected final model is a depth-32 CIFAR residual network trained with momentum SGD.

| Property | Result |
|---|---:|
| Trainable parameters | **464,154** |
| Parameter limit | 500,000 |
| Training budget | 20 epochs |
| Selected checkpoint | Epoch 19 |
| Best validation accuracy | **89.14%** |
| Final test accuracy | **88.75%** |
| Final test cross-entropy | 0.3375 |
| Test images | 10,000, evaluated once |

The test set was created only after the architecture, optimizer, hyperparameters, and checkpoint had been selected from validation results. It was evaluated once and was not used to revise the model.

## Assignment coverage

The notebook completes all required tasks and the optional final CNN task.

- **Task 1 — Backpropagation:** manual Linear, ReLU, softmax cross-entropy, and MLP forward/backward passes; gradient verification; manual SGD training; sigmoid/ReLU gradient-flow comparison.
- **Task 2 — Optimizers:** custom momentum SGD and bias-corrected Adam subclasses; parity checks against PyTorch; optimizer trajectories; CNN learning-rate comparison.
- **Task 3 — Regularization and residual connections:** baseline, weight decay, dropout, flip and crop, manual Cutout, manual Mixup, combined regularization, and controlled plain/residual depth comparisons.
- **Task 4 — Final CNN:** a sub-500k-parameter model, four design-choice ablations, validation-only model selection, and one final test evaluation.

## Data and reproducibility

CIFAR-10 contains 50,000 standard training images and 10,000 test images. The standard training portion is split reproducibly into:

- 45,000 training images
- 5,000 validation images
- Random seed: `42`

Channel normalization statistics are calculated only from the 45,000-image training partition:

| Channel | Mean | Population standard deviation |
|---|---:|---:|
| Red | 0.491656 | 0.247081 |
| Green | 0.482334 | 0.243509 |
| Blue | 0.446679 | 0.261581 |

Python, NumPy, and PyTorch are seeded. The split indices, minibatch order, and experiment initializations are controlled. Device selection follows CUDA, Apple MPS, then CPU, so the notebook runs on a Colab T4 and on a laptop. The saved results in this repository were produced with Apple MPS and PyTorch 2.14.1.

## Experimental results

### Task 1: Explicit backpropagation

The manual implementation uses PyTorch tensors as numerical arrays but does not use `torch.nn` layers, `loss.backward()`, or `torch.autograd` to calculate the MLP gradients.

#### Gradient verification

The manual gradients were compared independently with PyTorch autograd and centred finite differences using double precision.

| Check | Largest relative error | All `torch.allclose` checks |
|---|---:|:---:|
| Autograd reference | 8.63 × 10⁻¹⁶ | ✅ |
| Centred finite differences | 2.90 × 10⁻⁹ | ✅ |

These errors are at floating-point and finite-difference precision, supporting the correctness of the explicit backward equations.

#### Manual MLP training

The manually differentiated MLP contains 789,258 trainable values and was trained with manually applied SGD updates on flattened `3 × 32 × 32` images.

| Metric | Result |
|---|---:|
| Initial epoch loss | 1.7600 |
| Final epoch loss | 1.0534 |
| Best validation accuracy | **49.34%** at epoch 3 |
| Final validation accuracy | 46.76% |

#### Ten-layer gradient flow

At initialization, the first-layer sigmoid gradient norm was only `1.5 × 10⁻⁵`, compared with `0.270886` for ReLU. The sigmoid gradients shrink strongly toward the input because repeated sigmoid derivatives are at most 0.25. Kaiming-initialized ReLU maintains substantially healthier gradient magnitudes across depth.

### Task 2: Optimizers

#### Custom optimizer verification

`MomentumSGD` and `BiasCorrectedAdam` are implemented as `torch.optim.Optimizer` subclasses. Over six deterministic update steps, their parameters matched the corresponding PyTorch optimizers exactly. Adam's largest optimizer-state difference was `4.16 × 10⁻¹⁷`; every parameter and state check passed.

#### Ill-conditioned quadratic

The optimizers were compared on

$$
f(x,y)=\frac{1}{2}(x^2+50y^2).
$$

| Optimizer | Learning rate | Final objective after 60 updates |
|---|---:|---:|
| SGD | 0.035 | 0.111267 |
| SGD + momentum | 0.035 | 0.031483 |
| Adam | 0.120 | **0.016996** |

Plain SGD oscillates along the steep direction and progresses slowly along the shallow direction. Momentum reduces this zig-zag motion, while Adam's coordinate-wise scaling adapts effectively to the different curvatures.

#### CNN optimizer and learning-rate comparison

Each optimizer was tested at three learning rates for eight epochs using the same small CNN and controlled initialization.

| Optimizer | Tested learning rates | Selected rate | Selected validation accuracy | Accuracy range |
|---|---|---:|---:|---:|
| SGD | 0.001, 0.01, 0.1 | 0.1 | 72.12% | 38.10 pp |
| SGD + momentum | 0.001, 0.01, 0.1 | 0.01 | **73.24%** | 15.46 pp |
| Adam | 0.0001, 0.0005, 0.001 | 0.001 | 72.20% | 10.88 pp |

Plain SGD was the most learning-rate sensitive over the tested grid. Adam was the least sensitive, while momentum SGD achieved the highest validation accuracy at its best rate.

### Task 3: Regularization and residual connections

#### Regularization on 5,000 training images

The comparison uses a small batch-normalized CNN. Cutout and Mixup are implemented directly in the notebook rather than using library implementations.

| Configuration | Train accuracy | Validation accuracy | Train–validation gap |
|---|---:|---:|---:|
| Baseline | 100.00% | 60.82% | 39.18 pp |
| Weight decay | 100.00% | 61.64% | 38.36 pp |
| Dropout | 85.10% | 59.62% | 25.48 pp |
| **Flip + crop** | 76.06% | **65.64%** | 10.42 pp |
| Cutout | 98.36% | 60.56% | 37.80 pp |
| Mixup | 99.50% | 57.24% | 42.26 pp |
| All combined | 57.78% | 54.02% | **3.76 pp** |

Flip and crop produced the highest validation accuracy and greatly reduced the generalization gap. The combined configuration had the smallest gap but also low training and validation accuracy, indicating underfitting under the fixed training schedule.

#### Plain and residual CIFAR networks

Both families use depths `6n + 2`, three stages with 16/32/64 channels, batch normalization, Kaiming initialization, and global average pooling. The shortcut is the only difference between each matched pair.

| Depth | Parameters | Plain final loss | Residual final loss |
|---:|---:|---:|---:|
| 8 | 75,290 | **0.4599** | 0.4999 |
| 20 | 269,722 | 0.4081 | **0.2849** |
| 56 | 853,018 | 1.4420 | **0.3913** |

The depth-8 networks optimize similarly, but the residual advantage grows with depth. The 56-layer plain network remains at high loss, whereas its residual counterpart trains successfully. This demonstrates how identity shortcuts improve gradient propagation and optimization in deeper networks.

### Task 4: Final CNN and ablation study

The final candidate is a depth-32 CIFAR network because `32 = 6(5) + 2`. It uses three 16/32/64-channel stages, batch normalization, Kaiming initialization, global average pooling, and parameter-free residual shortcuts.

The fixed protocol uses all 45,000 training images, batch size 128, cross-entropy, and 20 epochs. Augmentation consists of a random `32 × 32` crop after four-pixel reflection padding and a horizontal flip with probability 0.5. Weight decay of `5 × 10⁻⁴` is applied to convolutional and linear weights only.

| Configuration | Best validation accuracy | Best epoch | Final training loss | Change from full candidate |
|---|---:|---:|---:|---:|
| Full candidate: AdamW | 87.58% | 20 | 0.2908 | — |
| No shortcuts | 79.28% | 18 | 0.5779 | −8.30 pp |
| No augmentation | 79.20% | 16 | 0.0036 | −8.38 pp |
| No weight decay | 87.32% | 20 | 0.2908 | −0.26 pp |
| **Momentum SGD** | **89.14%** | **19** | **0.2183** | **+1.56 pp** |

Removing shortcuts slowed optimization and reduced validation accuracy. Removing augmentation drove the training loss almost to zero while validation accuracy stalled near 79%, showing severe overfitting. Weight decay gave a small improvement in this run. Momentum SGD outperformed the initial AdamW candidate and was therefore selected before the test set was loaded.

## Final-model training configuration

| Setting | Value |
|---|---|
| Architecture | Depth-32 residual CIFAR network |
| Stage widths | 16 / 32 / 64 |
| Classifier | Global average pooling + linear layer |
| Initialization | Kaiming initialization |
| Optimizer | SGD |
| Initial learning rate | 0.1 |
| Momentum | 0.9 |
| Weight decay | 5 × 10⁻⁴ on weight tensors only |
| Schedule | Cosine annealing to 10⁻⁴ |
| Augmentation | Reflection-pad crop + horizontal flip |
| Batch size | 128 |
| Epochs | 20 |
| Checkpoint rule | Highest validation accuracy; earlier epoch breaks ties |

## Running the notebook

### Local environment

Python 3.10 or newer is recommended. A minimal Conda setup is:

```bash
conda create -n myenv python=3.11 -y
conda activate myenv
python -m pip install torch torchvision numpy pandas matplotlib jupyter
jupyter notebook "CNN with CIFAR10.ipynb"
```

Open the notebook and use **Run All** for a complete fresh execution. `torchvision.datasets.CIFAR10(download=True)` downloads CIFAR-10 into `./data` when it is absent. The dataset directory is excluded from Git because it is generated locally.

The complete notebook runs many controlled training experiments and can take several hours. Runtime depends on the device and software version.

### Google Colab

1. Upload or open `CNN with CIFAR10.ipynb` in Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the notebook from the first cell in order.

PyTorch, torchvision, NumPy, pandas, and Matplotlib are normally available in Colab. The device-selection cell automatically chooses CUDA when the T4 is active.

### Test-set policy when reproducing

The notebook contains only one `CIFAR10(train=False)` construction, in the final Task 4 evaluation cell. During new model development, keep that cell unexecuted until validation-based model selection is complete. A fresh full run evaluates it once at the end.

## Repository structure

```text
.
├── CNN with CIFAR10.ipynb       # Complete implementation, report, outputs, and discussion
├── README.md                    # Project overview and reproducibility guide
├── data/                        # Local CIFAR-10 dataset (excluded from Git)
│   ├── cifar-10-python.tar.gz   # Downloaded Python archive
│   └── cifar-10-batches-py/     # Extracted CIFAR-10 batches
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       └── test_batch
```

The `data/` directory is shown because it is part of the local project layout used by the notebook. It is ignored by Git to avoid committing the approximately 170 MB CIFAR-10 archive and extracted dataset. On a fresh clone, the notebook creates this directory automatically through `CIFAR10(download=True)`. Jupyter checkpoint files are also ignored.

## AI-tool acknowledgement

ChatGPT/Codex was used as a coding and editing assistant during this project. It helped with implementation review, notebook organization, experiment execution, result verification, and documentation. The final notebook, experimental decisions, outputs, and explanations were reviewed as part of the assignment work, and the author is responsible for understanding and defending the submitted code.
