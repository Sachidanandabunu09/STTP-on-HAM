# Class Imbalance HAM10000 Dataset

A deep learning project addressing class imbalance in skin cancer detection using the HAM10000 dataset with the STTP_Net architecture.

## Project Structure

```
.
├── STTP_Net_HAM10000_6.ipynb    # Main training notebook
├── data/
│   ├── raw/
│   │   ├── metadata.csv          # Image metadata
│   │   └── images/               # HAM10000 images (not in repo)
│   └── splits/
│       ├── train.csv             # Training set indices
│       ├── val.csv               # Validation set indices
│       └── test.csv              # Test set indices
├── checkpoints/                  # Model checkpoints (not in repo)
├── experiments/                  # Experimental results
│   ├── exp01_baseline/
│   ├── exp02_baseline/
│   └── exp03_medical_aug/
└── .gitignore
```

## Dataset Setup

### Getting the HAM10000 Dataset

The dataset images are not included in this repository due to size constraints. You'll need to download them:

1. **Download from Kaggle**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. **Extract to**: `data/raw/images/`

### Directory Structure After Download

```
data/
├── raw/
│   ├── metadata.csv
│   └── images/
│       ├── ISIC_0024306.jpg
│       ├── ISIC_0024307.jpg
│       └── ... (10,015 images total)
├── splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
```

## Prerequisites

```bash
pip install -r requirements.txt
```

### Core Dependencies
- PyTorch >= 1.9.0
- NumPy
- Pandas
- Scikit-learn
- Pillow (PIL)
- Matplotlib
- Jupyter

## Usage

1. **Prepare the dataset**:
   - Download HAM10000 from Kaggle
   - Extract to `data/raw/images/`

2. **Run the notebook**:
   ```bash
   jupyter notebook STTP_Net_HAM10000_6.ipynb
   ```

3. **View results**:
   - Check `experiments/` folder for results and metrics
   - Results include `results_summary.json` and saved model checkpoints

## Experiments

- **exp01_baseline**: Baseline STTP_Net model
- **exp02_baseline**: Alternative baseline configuration
- **exp03_medical_aug**: Model with medical imaging augmentation

## Notes

- Model checkpoints (`.pt` files) are excluded from git due to size
- Image dataset is excluded from git (download separately)
- CSV split files are tracked for reproducibility

## License

This project uses the HAM10000 dataset which is publicly available. Please cite appropriately when using.

## References

- HAM10000 Dataset: https://github.com/christianversloot/HAM10000
- Original Paper: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 (2018)
