import boto3
import tarfile
import os
import shutil

# -------------------------------------------------------
# TODO - update CONFIGS
# -------------------------------------------------------
BUCKET_NAME = "your-s3-bucket-name"
S3_PREFIX = "random-forest-serverless"
REGION = "us-east-1"
# -------------------------------------------------------

def package_model():
    """Package model.joblib + inference.py into model.tar.gz"""
    staging_dir = "staging"
    os.makedirs(staging_dir, exist_ok=True)

    # Copy artifacts into staging
    shutil.copy("model/model.joblib", f"{staging_dir}/model.joblib")
    shutil.copy("inference.py", f"{staging_dir}/inference.py")

    # Create the tar.gz (SageMaker requires this exact format)
    tar_path = "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(f"{staging_dir}/model.joblib", arcname="model.joblib")
        tar.add(f"{staging_dir}/inference.py", arcname="inference.py")

    shutil.rmtree(staging_dir)
    print(f"✅ Packaged model to {tar_path}")
    return tar_path


def upload_to_s3(tar_path):
    """Upload model.tar.gz to S3 and return the S3 URI"""
    s3 = boto3.client("s3", region_name=REGION)
    s3_key = f"{S3_PREFIX}/model.tar.gz"

    print(f"⬆️  Uploading to s3://{BUCKET_NAME}/{s3_key} ...")
    s3.upload_file(tar_path, BUCKET_NAME, s3_key)

    s3_uri = f"s3://{BUCKET_NAME}/{s3_key}"
    print(f"✅ Uploaded to {s3_uri}")
    return s3_uri


if __name__ == "__main__":
    tar_path = package_model()
    s3_uri = upload_to_s3(tar_path)
    print(f"\n📦 S3 URI for deployment:\n   {s3_uri}")