"""
Deploy registered models as serving endpoints in Databricks.

Automatically deploys all models in SOURCE_SUFFIXES (gemini3 and gpt5).
Uses MLflow to manage model registry and Databricks SDK to create serving endpoints.

Configuration via environment variables:
  DEPLOY_FORCE_RECREATE: Set to 'true' to delete and recreate existing endpoints

Example .env file:
  DEPLOY_FORCE_RECREATE=false
"""
import os
from typing import Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
SOURCE_SUFFIXES = {
    "gemini3": "gemini_output",
    "gpt5": "gpt_output",
}


def get_registered_model_name(source_suffix: str) -> str:
    """Get the registered model name, supporting both standard and UC registry."""
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    model_name = f"{MODEL_NAME.replace('/', '_').replace('-', '_').replace('.', '')}_{source_suffix}"
    if registry_uri == "databricks-uc":
        catalog = os.environ["MLFLOW_UC_CATALOG"]
        schema = os.environ["MLFLOW_UC_SCHEMA"]
        return f"{catalog}.{schema}.{model_name}"
    return model_name


def get_model_uri(model_name: str) -> str:
    """Get the model URI for serving from the registered model name."""
    # For models registered in MLflow, the URI format is:
    # - Standard registry: models:/model_name/version
    # - Unity Catalog: models:/catalog.schema.model_name/version
    return f"models:/{model_name}/latest"


def get_latest_model_version(model_name: str) -> Optional[int]:
    """Get the latest version number of a registered model."""
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        # Sort by version number (descending) and get the first one
        latest_version = max(versions, key=lambda v: int(v.version))
        return int(latest_version.version)
    except Exception as e:
        print(f"Error fetching model version: {e}")
        return None


def verify_model_exists(model_name: str) -> bool:
    """Verify that the registered model exists."""
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return len(versions) > 0
    except Exception as e:
        print(f"Error verifying model exists: {e}")
        return False


def endpoint_exists(model_name: str) -> bool:
    """Check if a model has been registered for serving."""
    client = mlflow.MlflowClient()
    try:
        model = client.get_registered_model(model_name)
        return model is not None
    except Exception:
        return False


def cleanup_old_model_versions(model_name: str, keep_count: int = 3) -> None:
    """Archive old model versions, keeping only the most recent ones.
    
    Note: Unity Catalog doesn't support stage transitions, so cleanup is a no-op for UC.
    
    Args:
        model_name: Registered model name
        keep_count: Number of recent versions to keep (default: 3)
    """
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    
    # Skip cleanup for Unity Catalog (UC doesn't support stage transitions)
    if registry_uri == "databricks-uc":
        print(f"  Skipping cleanup for Unity Catalog (not supported)")
        return
    
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if len(versions) <= keep_count:
            print(f"  Only {len(versions)} versions found, no cleanup needed.")
            return
        
        # Sort by version number (descending)
        sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        
        # Archive versions beyond keep_count
        for version in sorted_versions[keep_count:]:
            print(f"  Archiving model version: {model_name} v{version.version}")
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
            except Exception as e:
                print(f"  Warning: Failed to archive version {version.version}: {e}")
    except Exception as e:
        print(f"Warning: Could not cleanup old versions: {e}")


def deploy_model_to_production(
    model_name: str,
    model_version: int,
) -> None:
    """Deploy a model version to Production using appropriate method for registry type.
    
    For Unity Catalog: Uses aliases
    For standard registry: Uses stages
    
    Args:
        model_name: Registered model name
        model_version: Model version to deploy
    """
    client = mlflow.MlflowClient()
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    
    print(f"Deploying model '{model_name}' v{model_version} to Production...")

    try:
        if registry_uri == "databricks-uc":
            # For Unity Catalog, use aliases
            print(f"  Using alias 'prod' for Unity Catalog deployment...")
            client.set_registered_model_alias(
                name=model_name,
                alias="prod",
                version=model_version
            )
            print(f"✓ Model deployed to Production!")
            print(f"  Model: {model_name}")
            print(f"  Version: {model_version}")
            print(f"  Alias: prod")
            print(f"  Model URI: models:/{model_name}@prod")
        else:
            # For standard registry, use stages
            print(f"  Using 'Production' stage for standard registry...")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production"
            )
            print(f"✓ Model deployed to Production!")
            print(f"  Model: {model_name}")
            print(f"  Version: {model_version}")
            print(f"  Stage: Production")
            print(f"  Model URI: models:/{model_name}/Production")
    except Exception as e:
        print(f"Error deploying model: {e}")
        raise


def endpoint_exists(ws_client: WorkspaceClient, endpoint_name: str) -> bool:
    """Check if a serving endpoint already exists."""
    try:
        ws_client.serving_endpoints.get(name=endpoint_name)
        return True
    except Exception:
        return False


def create_serving_endpoint(
    ws_client: WorkspaceClient,
    endpoint_name: str,
    model_name: str,
    model_version: int,
    force_recreate: bool = False,
) -> str:
    """Create or update a serving endpoint for the model using Databricks SDK.
    
    Args:
        ws_client: Databricks workspace client
        endpoint_name: Name for the serving endpoint
        model_name: Registered model name
        model_version: Model version to serve
        force_recreate: Whether to delete and recreate if exists
        
    Returns:
        The created/updated endpoint name
    """
    # Sanitize endpoint name (lowercase, replace special chars with dashes)
    sanitized_name = endpoint_name.lower().replace(".", "-").replace("/", "-").replace("_", "-")
    print(f"Creating serving endpoint '{sanitized_name}' for model '{model_name}' v{model_version}...")

    if endpoint_exists(ws_client, sanitized_name):
        if force_recreate:
            print(f"Endpoint '{sanitized_name}' already exists, deleting...")
            ws_client.serving_endpoints.delete(name=sanitized_name)
            # Wait for deletion to complete
            import time
            time.sleep(5)
        else:
            print(f"Endpoint '{sanitized_name}' already exists.")
            return sanitized_name

    # Create the endpoint configuration
    endpoint_config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version=model_version,
                workload_size="Small",  # Can be adjusted: Small, Medium, Large
                scale_to_zero_enabled=True,
            )
        ],
    )

    # Create the endpoint
    response = ws_client.serving_endpoints.create_and_wait(
        name=sanitized_name,
        config=endpoint_config
    )
    print(f"✓ Serving endpoint created successfully!")
    print(f"  Endpoint name: {response.name}")
    print(f"  Endpoint URL: https://databricks-prod.cloud/serving-endpoints/{response.name}")
    
    return response.name


def deploy(
    source_suffix: str,
    force_recreate: bool = False,
) -> None:
    """Main deployment function using MLflow and Databricks SDK."""
    print(f"\n{'='*60}")
    print(f"Deploying registered model as Databricks serving endpoint")
    print(f"{'='*60}")

    # Setup MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri(os.environ.get("MLFLOW_REGISTRY_URI", "databricks"))

    # Get the registered model name
    model_name = get_registered_model_name(source_suffix)
    print(f"\n1. Verifying registered model: {model_name}")

    # Verify the model is registered
    if not verify_model_exists(model_name):
        raise ValueError(
            f"Model '{model_name}' is not registered in MLflow. "
            f"Please run training first to register the model."
        )

    # Get the latest version
    version = get_latest_model_version(model_name)
    if version is None:
        raise ValueError(f"Could not find any versions of model '{model_name}'")

    print(f"   ✓ Model found (latest version: {version})")

    # Cleanup old model versions (keep only 3 most recent versions)
    print(f"\n2. Cleaning up old model versions (keeping at most 3 recent versions)...")
    cleanup_old_model_versions(model_name, keep_count=1)

    # Deploy the model to Production stage/alias
    print(f"\n3. Deploying model to MLflow Production...")
    deploy_model_to_production(model_name, version)

    # Generate endpoint name as model_name + version
    endpoint_name = f"{model_name.replace(',', '').replace('.', '-').replace('-', '_')}_v{version}"

    # Create Databricks serving endpoint
    print(f"\n4. Creating Databricks serving endpoint...")
    ws_client = WorkspaceClient()
    
    endpoint_result = create_serving_endpoint(
        ws_client,
        endpoint_name,
        model_name,
        version,
        force_recreate=force_recreate,
    )

    print(f"\n✓ Deployment complete!")
    print(f"  Model: {model_name}")
    print(f"  Version: {version}")
    print(f"  Serving Endpoint: {endpoint_result}")
    print(f"  The model is now available in Databricks for inference.")


def main():
    """Main function - reads configuration from environment variables and deploys all models."""
    # Load environment variables
    load_dotenv()

    # Get optional configuration from environment variables
    force_recreate = os.environ.get("DEPLOY_FORCE_RECREATE", "false").lower() == "true"

    # Validate environment setup
    required_env_vars = ["MLFLOW_TRACKING_URI"]
    for var in required_env_vars:
        if not os.environ.get(var):
            raise ValueError(f"Required environment variable '{var}' is not set")

    # Check for UC catalog if using Unity Catalog
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI", "databricks")
    if registry_uri == "databricks-uc":
        for var in ["MLFLOW_UC_CATALOG", "MLFLOW_UC_SCHEMA"]:
            if not os.environ.get(var):
                raise ValueError(f"Required for Unity Catalog: '{var}' is not set")

    print(f"Configuration:")
    print(f"  Force Recreate: {force_recreate}")
    print(f"  Deploying models for all sources: {', '.join(SOURCE_SUFFIXES.keys())}")

    # Deploy all source suffixes
    for source_suffix in SOURCE_SUFFIXES.keys():
        print(f"\n{'='*60}")
        print(f"Deploying model for source: {source_suffix}")
        print(f"{'='*60}")
        try:
            deploy(
                source_suffix=source_suffix,
                force_recreate=force_recreate,
            )
        except Exception as e:
            print(f"Error deploying {source_suffix}: {e}")
            raise


if __name__ == "__main__":
    main()
