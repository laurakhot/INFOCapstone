import boto3
import csv
import gzip
import io
import json
import urllib.parse
from alerts.sender import send_alert
from alerts.templates import METRIC_LOWER_IS_WORSE

REGION = "us-east-2"

THRESHOLDS_BUCKET = 'huskysupport-bucket'
THRESHOLDS_KEY = 'thresholds/spike_thresholds.json'

s3 = boto3.client('s3')

thresholds = {}

def get_thresholds():
    global thresholds
    response = s3.get_object(Bucket=THRESHOLDS_BUCKET, Key=THRESHOLDS_KEY)
    thresholds = json.loads(response['Body'].read().decode('utf-8'))
    print(f"Loaded {len(thresholds)} thresholds")

def check_thresholds(bucket, file_key):
    response = s3.get_object(Bucket=bucket, Key=file_key)
    raw = response['Body'].read()
    if file_key.endswith('.gz'):
        raw = gzip.decompress(raw)
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))

    problematic_users = []

    for row in reader:
        triggered_features = {}

        for feature, threshold in thresholds.items():
            if feature not in row or not row.get("auth_username") or not row.get("model_name"):
                continue
            try:
                if feature in METRIC_LOWER_IS_WORSE:
                    triggered = float(row[feature]) < threshold
                else:
                    triggered = float(row[feature]) > threshold
                if triggered:
                    triggered_features[feature] = float(row[feature])
                    print(row["auth_username"])
            except (ValueError, TypeError):
                continue

        if triggered_features:
            problematic_users.append({
                "username": row["auth_username"],
                "device": row["model_name"],
                "features": triggered_features
            })

    return {"problematic_users": problematic_users}

def lambda_handler(event, context):
    """
    Triggered by S3 upload to huskysupport-bucket/telemetry/
    Reads the uploaded CSV, compares each row's features against thresholds,
    and returns all users with at least one triggered feature.
    """
    record = event["Records"][0]
    bucket_name = record["s3"]["bucket"]["name"]
    file_key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
    file_size = record["s3"]["object"]["size"]

    print(f"New file detected: s3://{bucket_name}/{file_key} ({file_size} bytes)")

    if bucket_name != "huskysupport-bucket":
        print(f"Unexpected bucket: {bucket_name}, skipping.")
        return {"statusCode": 200, "body": "Skipped — wrong bucket"}

    if not file_key.startswith("telemetry/"):
        print(f"Unexpected path: {file_key}, skipping.")
        return {"statusCode": 200, "body": "Skipped — wrong folder"}

    get_thresholds()
    result = check_thresholds(bucket_name, file_key)

    print(f"Problematic users found: {len(result['problematic_users'])}")

    for user in result["problematic_users"]:
        try:
            send_alert(
                username=user["username"],
                device=user["device"],
                features=user["features"],
            )
        except Exception as e:
            print(f"Alert failed for {user['username']}: {e}")

    return {
        "statusCode": 200,
        "body": json.dumps(result)
    }
