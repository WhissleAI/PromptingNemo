#!/usr/bin/env bash
set -euo pipefail

PROJECT="deepvoice-468015"
ZONE="us-central1-c"
CLUSTER_NAME="dataprep-cluster"
NAMESPACE="dataprep"

echo "=== Teardown GKE Data Prep ==="
echo "This will delete the cluster and all jobs. NFS data is preserved."
echo ""
read -p "Continue? (y/N) " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

echo "Deleting namespace (stops all jobs)..."
kubectl delete namespace "$NAMESPACE" --ignore-not-found

echo "Deleting NFS PV..."
kubectl delete pv nfs-training-pv --ignore-not-found

echo "Deleting cluster..."
gcloud container clusters delete "$CLUSTER_NAME" \
    --zone "$ZONE" \
    --project "$PROJECT" \
    --quiet

echo "Done. NFS data at 10.134.50.122:/training is untouched."
