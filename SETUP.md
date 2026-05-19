# Setup Guide

## Prerequisites

- Python 3.7 or higher
- Git
- ~4GB free disk space for the dataset
- CUDA-capable GPU (recommended, but CPU works too)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/class-imbalance-ham10000.git
cd date-16
```

### 2. Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the Dataset

The HAM10000 dataset is not included in this repository due to size constraints.

#### Option A: Download from Kaggle (Recommended)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
3. Click "Download" to get `skin-cancer-mnist-ham10000.zip` (~1.4GB)
4. Extract the zip file to `data/raw/images/`

#### Option B: Manual Setup

If you have the images already:
```bash
# Copy images to the correct location
copy /path/to/images/* data/raw/images/
```

### 5. Verify Installation

```bash
# Check if dataset is present
dir data/raw/images/

# Expected output: ~10,000 ISIC_*.jpg files
```

## Running the Project

### Start Jupyter Notebook

```bash
jupyter notebook
```

Then open `STTP_Net_HAM10000_6.ipynb` in your browser.

### GPU Setup (Optional but Recommended)

```bash
# Verify PyTorch can access GPU
python -c "import torch; print(torch.cuda.is_available())"
```

If using CUDA, ensure you have the correct NVIDIA drivers installed.

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Reinstall PyTorch
```bash
pip install --upgrade torch torchvision
```

### Issue: Memory errors during training
**Solution**: Reduce batch size in the notebook or use a GPU

### Issue: Images not found
**Solution**: 
- Verify images are in `data/raw/images/`
- Check filename format matches `ISIC_*.jpg`
- Run `dir data/raw/images/ | wc -l` to count files (should be ~10,000)

### Issue: Dataset splits not found
**Solution**: Ensure `data/splits/` contains `train.csv`, `val.csv`, `test.csv`

## Project Structure After Setup

```
date-16/
├── README.md
├── SETUP.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── STTP_Net_HAM10000_6.ipynb
├── data/
│   ├── raw/
│   │   ├── metadata.csv
│   │   └── images/          ← Download dataset here
│   │       ├── ISIC_0024306.jpg
│   │       ├── ISIC_0024307.jpg
│   │       └── ... (10,015 images)
│   └── splits/
│       ├── train.csv        ← Training indices
│       ├── val.csv          ← Validation indices
│       └── test.csv         ← Test indices
├── experiments/
│   ├── exp01_baseline/
│   │   ├── config.json
│   │   └── results_summary.json
│   ├── exp02_baseline/
│   │   ├── config.json
│   │   └── results_summary.json
│   └── exp03_medical_aug/
│       ├── config.json
│       └── results_summary.json
└── checkpoints/             ← Models saved here

```

## Next Steps

1. ✅ Clone repository
2. ✅ Set up virtual environment
3. ✅ Install dependencies
4. ✅ Download dataset
5. 🚀 Open and run the Jupyter notebook
6. 📊 Review results in `experiments/` folder

## Support

For issues or questions, please check:
- Dataset documentation: https://github.com/christianversloot/HAM10000
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Kaggle dataset page: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
