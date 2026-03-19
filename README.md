# TabPFN Embeddings for Sirbu Dataset

This repository contains scripts to load the Sirbu dataset, preprocess it for Cox modeling, and extract/visualize high-dimensional embeddings using TabPFN.

## Setup Requirements

We use `uv` for extremely fast python dependency resolving and installation. If you don't have `uv` installed yet, [install it first](https://github.com/astral-sh/uv?tab=readme-ov-file#installation):

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Create a Virtual Environment

Initialize a `.venv` directory in this folder using `uv`:

```bash
uv venv
```

Activate the virtual environment:

```bash
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

With the environment activated, use `uv pip` to install the requirements:

```bash
uv sync

# or manually install via toml if you don't have sync configured:
uv pip install -e .
```

*(Note: Ensure you are using the full TabPFN installation `tabpfn>=6.4.1`, as embeddings cannot be extracted using the lightweight `tabpfn-client`)*.

---

## Generating & Visualizing Embeddings

Once your environment is set up and your data folder (`Dataset Sirbu/`) exists within the directory, you can run the test script:

```bash
python test_tabpfn.py
```

### What this does:
1. **Loads & Imputes** the `Dataset Sirbu`.
2. Extracts **Total mortality** event outcomes and drops non-event features.
3. Automatically splits your arrays into a supervised 80/20 train/test mapping.
4. Leverages **TabPFN** to extract deep representations of your features across a K-fold stable distribution.
5. Projects the high-dimensional (`192` dimensional) TabPFN variables into a 2-dimensional scatterplot using **t-SNE**.
6. Drops the styled visual output into `results/tabpfn_mortality_tsne.pdf`.
