"""One-time provisioning of the Google Cloud resources used by the app.

Creates (idempotently, by display name):
  1. a Cloud Storage bucket for the contracts (optionally uploading ./contracts)
  2. a Vertex AI Vector Search index with STREAM_UPDATE, DOT_PRODUCT_DISTANCE and
     UNIT_L2_NORM (so returned distances are cosine similarities)
  3. a public index endpoint, with the index deployed on it

and prints the values to put in ``.env``.

Prerequisites (see README): APIs enabled, a Firestore database created, and
Application Default Credentials (``gcloud auth application-default login`` or a
service account) with permission to manage these resources.

Usage:
    python scripts/setup_gcp.py --project my-project --region us-central1 \\
        --bucket my-project-contracts --upload-samples

Deploying the index takes ~20-40 minutes. A deployed index is billed per node
hour even when idle; undeploy it (or delete the endpoint) when you are done.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("setup_gcp")

ROOT = Path(__file__).resolve().parent.parent


def ensure_bucket(project: str, region: str, bucket_name: str, upload_dir: Path | None, prefix: str) -> None:
    from google.cloud import storage
    from google.api_core.exceptions import Conflict

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        try:
            bucket = client.create_bucket(bucket_name, location=region)
            bucket.iam_configuration.uniform_bucket_level_access_enabled = True
            bucket.patch()
            logger.info("Created bucket gs://%s", bucket_name)
        except Conflict:
            logger.info("Bucket gs://%s already exists", bucket_name)
    else:
        logger.info("Bucket gs://%s exists", bucket_name)
    if upload_dir:
        for path in sorted(upload_dir.iterdir()):
            if path.suffix.lower() in (".pdf", ".docx", ".txt"):
                bucket.blob(f"{prefix}{path.name}").upload_from_filename(str(path))
                logger.info("Uploaded %s -> gs://%s/%s%s", path.name, bucket_name, prefix, path.name)


def ensure_index(display_name: str, dim: int):
    from google.cloud import aiplatform
    from google.cloud.aiplatform.matching_engine import matching_engine_index_config as cfg

    existing = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{display_name}"')
    if existing:
        logger.info("Index %s exists: %s", display_name, existing[0].resource_name)
        return existing[0]
    logger.info("Creating index %s (dim=%d); this takes a few minutes...", display_name, dim)
    return aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        dimensions=dim,
        approximate_neighbors_count=50,
        distance_measure_type=cfg.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
        feature_norm_type=cfg.FeatureNormType.UNIT_L2_NORM,
        index_update_method="STREAM_UPDATE",
        shard_size="SHARD_SIZE_SMALL",
        description="Contract chat clause chunks (restrict namespace: doc_id)",
    )


def ensure_endpoint(display_name: str):
    from google.cloud import aiplatform

    existing = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{display_name}"')
    if existing:
        logger.info("Endpoint %s exists: %s", display_name, existing[0].resource_name)
        return existing[0]
    logger.info("Creating public index endpoint %s...", display_name)
    return aiplatform.MatchingEngineIndexEndpoint.create(display_name=display_name, public_endpoint_enabled=True)


def ensure_deployed(endpoint, index, deployed_index_id: str, machine_type: str) -> None:
    for d in endpoint.deployed_indexes:
        if d.id == deployed_index_id:
            logger.info("Index already deployed as %s", deployed_index_id)
            return
    logger.info("Deploying index as %s on %s (this takes ~20-40 minutes)...", deployed_index_id, machine_type)
    endpoint.deploy_index(
        index=index,
        deployed_index_id=deployed_index_id,
        machine_type=machine_type,
        min_replica_count=1,
        max_replica_count=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True)
    p.add_argument("--region", default="us-central1")
    p.add_argument("--bucket", required=True, help="Cloud Storage bucket for the contracts")
    p.add_argument("--prefix", default="contracts/", help="Object prefix inside the bucket")
    p.add_argument("--upload-samples", action="store_true", help="Upload ./contracts to the bucket")
    p.add_argument("--dim", type=int, default=768, help="Embedding dimension (must match EMBEDDING_DIM)")
    p.add_argument("--index-name", default="contract-chat-index")
    p.add_argument("--endpoint-name", default="contract-chat-endpoint")
    p.add_argument("--deployed-index-id", default="contract_chat_deployed")
    p.add_argument("--machine-type", default="e2-standard-2")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from google.cloud import aiplatform

    aiplatform.init(project=args.project, location=args.region)

    ensure_bucket(args.project, args.region, args.bucket, ROOT / "contracts" if args.upload_samples else None,
                  args.prefix)
    index = ensure_index(args.index_name, args.dim)
    endpoint = ensure_endpoint(args.endpoint_name)
    ensure_deployed(endpoint, index, args.deployed_index_id, args.machine_type)

    print("\n# ---- add to .env ----")
    print(f"GCP_PROJECT={args.project}")
    print(f"GCP_LOCATION={args.region}")
    print(f"GCS_BUCKET={args.bucket}")
    print(f"GCS_PREFIX={args.prefix}")
    print(f"EMBEDDING_DIM={args.dim}")
    print(f"VERTEX_INDEX_ID={index.resource_name}")
    print(f"VERTEX_INDEX_ENDPOINT_ID={endpoint.resource_name}")
    print(f"VERTEX_DEPLOYED_INDEX_ID={args.deployed_index_id}")


if __name__ == "__main__":
    main()
