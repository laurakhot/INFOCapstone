# 1. Install dependencies
pip install sagemaker boto3 scikit-learn joblib

# 2. Export your model
python train_and_export.py

# 3. Package and upload to S3
python package_and_upload.py

# 4. Deploy and test
python deploy.py


Edit inference.py
      ↓
Re-run package_and_upload.py   ← re-tars and pushes new .tar.gz to S3
      ↓
Re-run deploy.py               ← creates new endpoint config pointing to new artifact


| What | Where | What to change |
|---|---|---|
| Your training code | `train_and_export.py` | Replace the placeholder `make_classification` block |
| S3 bucket | `package_and_upload.py` + `deploy.py` | `BUCKET_NAME` and `S3_MODEL_URI` |
| IAM role | `deploy.py` | `SAGEMAKER_ROLE` |
| Memory size | `deploy.py` | Start at `2048`, increase if you hit OOM errors |
| sklearn version | `deploy.py` | Match `framework_version` to your local sklearn version (`sklearn.__version__`) |
| Test payload | `deploy.py` | Update `test_payload` to match your model's feature count/shape |