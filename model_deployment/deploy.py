import boto3
import json
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

# -------------------------------------------------------
# CONFIG — update these
# -------------------------------------------------------
S3_MODEL_URI = "s3://your-s3-bucket-name/random-forest-serverless/model.tar.gz"
SAGEMAKER_ROLE = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"
ENDPOINT_NAME = "random-forest-serverless-endpoint"
REGION = "us-east-2"
MEMORY_SIZE_MB = 4096
MAX_CONCURRENCY = 5
# -------------------------------------------------------


def endpoint_exists():
    """Check if the endpoint already exists in SageMaker."""
    client = boto3.client("sagemaker", region_name=REGION)
    try:
        client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return True
    except client.exceptions.ClientError:
        return False


def get_sklearn_model():
    """Build and return the SKLearnModel object."""
    return SKLearnModel(
        model_data=S3_MODEL_URI,
        role=SAGEMAKER_ROLE,
        entry_point="inference.py",
        framework_version="1.2-1",
        region_name=REGION,
    )


def get_serverless_config():
    """Build and return the serverless inference config."""
    return ServerlessInferenceConfig(
        memory_size_in_mb=MEMORY_SIZE_MB,
        max_concurrency=MAX_CONCURRENCY,
    )


def deploy_new():
    """Fresh deploy — no existing endpoint."""
    print(f"🚀 Deploying new endpoint: {ENDPOINT_NAME}")

    predictor = get_sklearn_model().deploy(
        serverless_inference_config=get_serverless_config(),
        endpoint_name=ENDPOINT_NAME,
    )

    print(f"✅ Endpoint deployed: {ENDPOINT_NAME}")
    return predictor


def redeploy_update():
    """Update existing endpoint in place — no downtime."""
    print(f"🔄 Endpoint already exists — updating in place: {ENDPOINT_NAME}")

    predictor = get_sklearn_model().deploy(
        serverless_inference_config=get_serverless_config(),
        endpoint_name=ENDPOINT_NAME,
        update_endpoint=True,       # ← swaps the model with no downtime
    )

    print(f"✅ Endpoint updated: {ENDPOINT_NAME}")
    return predictor


def redeploy_fresh():
    """Delete the existing endpoint and redeploy from scratch."""
    print(f"🗑️  Deleting existing endpoint: {ENDPOINT_NAME}")

    client = boto3.client("sagemaker", region_name=REGION)
    client.delete_endpoint(EndpointName=ENDPOINT_NAME)

    # Wait for deletion to complete before redeploying
    waiter = client.get_waiter("endpoint_deleted")
    print("⏳ Waiting for endpoint deletion...")
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print("✅ Endpoint deleted")

    return deploy_new()


def deploy(strategy="update"):
    """
    Main deploy entry point.

    strategy options:
      "update" — update in place if exists, deploy fresh if not (recommended)
      "fresh"  — always delete and redeploy from scratch
      "new"    — only deploy if endpoint doesn't exist, error if it does
    """
    exists = endpoint_exists()

    if strategy == "update":
        return redeploy_update() if exists else deploy_new()

    elif strategy == "fresh":
        return redeploy_fresh() if exists else deploy_new()

    elif strategy == "new":
        if exists:
            raise ValueError(
                f"Endpoint '{ENDPOINT_NAME}' already exists. "
                "Use strategy='update' or strategy='fresh' to redeploy."
            )
        return deploy_new()

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'update', 'fresh', or 'new'.")


def test_endpoint(predictor):
    """Send a test prediction to the live endpoint."""
    print("\n🧪 Testing endpoint...")

    test_payload = {
        "instances": [
            [0.5, 1.2, -0.3, 0.8, 1.5, -1.1, 0.2, 0.9, -0.5, 1.0],
            [1.1, -0.4, 0.7, -0.2, 0.3,  0.6, 1.4, -0.8, 0.1, 0.5],
        ]
    }

    response = predictor.predict(
        json.dumps(test_payload),
        initial_args={"ContentType": "application/json", "Accept": "application/json"}
    )

    result = json.loads(response)
    print("✅ Predictions:", result["predictions"])
    print("   Probabilities:", result["probabilities"])
    return result


def delete_endpoint():
    """Clean up — delete the endpoint to stop incurring charges."""
    if endpoint_exists():
        client = boto3.client("sagemaker", region_name=REGION)
        client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"🗑️  Endpoint {ENDPOINT_NAME} deleted.")
    else:
        print(f"⚠️  Endpoint {ENDPOINT_NAME} doesn't exist, nothing to delete.")


if __name__ == "__main__":
    # Change strategy here:
    #   "update" — update in place, no downtime        (recommended for prod)
    #   "fresh"  — delete and redeploy from scratch    (use if update fails)
    #   "new"    — only works if no endpoint exists    (use for first deploy)
    predictor = deploy(strategy="update")
    test_endpoint(predictor)

    # Uncomment to delete after testing:
    # delete_endpoint()