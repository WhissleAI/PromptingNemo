#!/usr/bin/env bash
set -euo pipefail

PROJECT="deepvoice-468015"
REGION="us-central1"
ZONE="us-central1-c"
CLUSTER_NAME="dataprep-cluster"
NODE_POOL="dataprep-pool"
NAMESPACE="dataprep"
IMAGE="us-central1-docker.pkg.dev/${PROJECT}/cloud-run-source-deploy/dataprep:latest"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

HF_TOKEN="${HF_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Set HF_TOKEN env var before running."
    echo "  export HF_TOKEN=hf_your_token_here"
    exit 1
fi

GEMINI_API_KEY="${GEMINI_API_KEY:-}"
if [ -z "$GEMINI_API_KEY" ]; then
    echo "WARNING: GEMINI_API_KEY not set — annotation jobs will skip entity/intent tagging."
fi

echo "=== Step 1: Create GKE cluster (if needed) ==="
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
    echo "Creating cluster ${CLUSTER_NAME} in ${ZONE}..."
    gcloud container clusters create "$CLUSTER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --num-nodes 0 \
        --machine-type n2-standard-2 \
        --disk-size 50 \
        --no-enable-autoupgrade \
        --network default \
        --subnetwork default
    echo "Cluster created."
else
    echo "Cluster ${CLUSTER_NAME} already exists."
fi

echo "=== Step 2: Create node pool (if needed) ==="
if ! gcloud container node-pools describe "$NODE_POOL" --cluster "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
    echo "Creating node pool ${NODE_POOL}..."
    gcloud container node-pools create "$NODE_POOL" \
        --cluster "$CLUSTER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --machine-type n2-standard-16 \
        --num-nodes 2 \
        --disk-size 100 \
        --disk-type pd-ssd \
        --no-enable-autoupgrade \
        --scopes storage-full,compute-rw
    echo "Node pool created (2x n2-standard-16, on-demand, no preemption)."
else
    echo "Node pool ${NODE_POOL} already exists."
fi

echo "=== Step 2b: Create GPU node pool (if needed) ==="
GPU_POOL="gpu-pool"
if ! gcloud container node-pools describe "$GPU_POOL" --cluster "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
    echo "Creating GPU node pool ${GPU_POOL}..."
    gcloud container node-pools create "$GPU_POOL" \
        --cluster "$CLUSTER_NAME" \
        --project "$PROJECT" \
        --zone "$ZONE" \
        --machine-type n1-standard-8 \
        --num-nodes 0 \
        --enable-autoscaling \
        --min-nodes 0 \
        --max-nodes 1 \
        --accelerator type=nvidia-tesla-t4,count=1 \
        --disk-size 100 \
        --disk-type pd-ssd \
        --no-enable-autoupgrade \
        --provisioning-model SPOT \
        --scopes storage-full,compute-rw
    echo "GPU node pool created (0-1x n1-standard-8 + T4, spot)."
else
    echo "GPU node pool ${GPU_POOL} already exists."
fi

echo "=== Step 3: Get cluster credentials ==="
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$ZONE" --project "$PROJECT"

echo "=== Step 4: Create namespace ==="
kubectl create namespace "$NAMESPACE" 2>/dev/null || true

echo "=== Step 5: Create secrets ==="
kubectl create secret generic hf-token \
    --namespace "$NAMESPACE" \
    --from-literal=token="$HF_TOKEN" \
    --dry-run=client -o yaml | kubectl apply -f -

if [ -n "$GEMINI_API_KEY" ]; then
    kubectl create secret generic gemini-key \
        --namespace "$NAMESPACE" \
        --from-literal=key="$GEMINI_API_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -
    echo "  HF token + Gemini key secrets applied."
else
    echo "  HF token secret applied. (Gemini key skipped)"
fi

echo "=== Step 6: Build and push container image ==="
DOCKER_DIR=$(mktemp -d)
cp "$SCRIPT_DIR/Dockerfile" "$DOCKER_DIR/"
mkdir -p "$DOCKER_DIR/scripts"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_multilingual_cv.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_hf_dataset.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_en_people.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_zh_datasets.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_madasr.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_indicvoices_raw.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/annotate_indic_meta.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/merge_gujarati_manifests.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/download_en_digits.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/merge_english_manifests.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/annotate_audio_tags.py" "$DOCKER_DIR/scripts/"
cp "$REPO_ROOT/scripts/asr/meta-asr/utils/annotate_with_gemini.py" "$DOCKER_DIR/scripts/"

echo "Building image..."
docker build --platform linux/amd64 -t "$IMAGE" "$DOCKER_DIR"
echo "Pushing image..."
docker push "$IMAGE"
rm -rf "$DOCKER_DIR"

echo "=== Step 7: Apply NFS PV/PVC ==="
kubectl apply -f "$SCRIPT_DIR/nfs-volume.yaml"

echo "=== Step 8: Launch jobs ==="
echo "Applying all download jobs..."
for job_file in "$SCRIPT_DIR"/job-*.yaml; do
    job_name=$(basename "$job_file" .yaml)
    echo "  Deploying ${job_name}..."
    kubectl delete job "${job_name/job-/dataprep-}" --namespace "$NAMESPACE" 2>/dev/null || true
    kubectl apply -f "$job_file"
done

echo ""
echo "=== All jobs launched ==="
echo "Monitor with:"
echo "  kubectl get jobs -n $NAMESPACE"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -n $NAMESPACE -l app=dataprep -f --max-log-requests=10"
echo ""
echo "Check specific job:"
echo "  kubectl logs -n $NAMESPACE job/dataprep-cv-euro -f"
echo "  kubectl logs -n $NAMESPACE job/dataprep-cv-slavic -f"
echo "  kubectl logs -n $NAMESPACE job/dataprep-indo -c aishell1 -f"
echo ""
echo "Estimated runtime: 24-48 hours for full download"
echo "Estimated cost: ~\$1.50/hr × 2 nodes × 48h ≈ \$144"
