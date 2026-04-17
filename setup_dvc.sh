#!/bin/bash
# Run once to set up DVC, S3, and push both dataset versions
# Usage: bash setup_dvc.sh

BUCKET="2022bcs0057"
ROLL_NO="2022BCS0057"

echo "===== Lab 8 DVC Setup - $ROLL_NO ====="

# 1. Create S3 bucket (name = roll no as per lab spec)
echo "[1] Creating S3 bucket: $BUCKET"
aws s3 mb s3://$BUCKET --region us-east-1 || echo "Bucket exists, continuing..."

# 2. Install dependencies
pip install dvc dvc-s3 pandas scikit-learn -q

# 3. Init DVC
echo "[2] Initializing DVC"
dvc init 2>/dev/null || true
git add .dvc/ .dvcignore 2>/dev/null
git commit -m "Initialize DVC [$ROLL_NO]" 2>/dev/null || true

# 4. Configure S3 remote
echo "[3] Configuring S3 remote"
dvc remote add -d s3remote_2022BCS0057 s3://$BUCKET/dvc-store -f
git add .dvc/config
git commit -m "Configure S3 remote [$ROLL_NO]" 2>/dev/null || true

# 5. Download California Housing dataset
echo "[4] Downloading California Housing dataset"
pip install kaggle -q 2>/dev/null
python3 -c "
import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np

housing = fetch_california_housing(as_frame=True)
df = housing.frame
df.columns = ['longitude','latitude','housing_median_age','total_rooms',
              'total_bedrooms','population','households','median_income','median_house_value']
df['median_house_value'] = df['median_house_value'] * 100000

# VERSION 1 - first 5000 rows
import os; os.makedirs('data', exist_ok=True)
df.iloc[:5000].to_csv('data/housing.csv', index=False)
df.to_csv('data/housing_full.csv', index=False)
print(f'V1: {len(df.iloc[:5000])} rows | V2: {len(df)} rows')
"

# 6. VERSION 1 - track and push
echo "[5] Pushing Version 1 (5000 rows)"
dvc add data/housing.csv
git add data/housing.csv.dvc data/.gitignore
git commit -m "Dataset v1: partial California Housing (5000 rows) [$ROLL_NO]"
dvc push

# 7. VERSION 2 - replace with full dataset
echo "[6] Pushing Version 2 (full dataset)"
cp data/housing_full.csv data/housing.csv
dvc add data/housing.csv
git add data/housing.csv.dvc
git commit -m "Dataset v2: full California Housing (20640 rows) [$ROLL_NO]"
dvc push

echo ""
echo "✅ Done!"
echo "   Bucket : s3://$BUCKET/dvc-store"
echo "   V1     : 5000 rows"
echo "   V2     : 20640 rows"
echo ""
echo "Now push to GitHub: git push origin main"
