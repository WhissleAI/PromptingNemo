#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="dataprep"

echo "=== GKE Data Prep Status ==="
echo ""

echo "--- Jobs ---"
kubectl get jobs -n "$NAMESPACE" -o wide 2>/dev/null || echo "No jobs found (cluster may not be configured)"
echo ""

echo "--- Pods ---"
kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null || echo "No pods found"
echo ""

echo "--- Pod Logs (last 5 lines per container) ---"
for pod in $(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
    containers=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.spec.containers[*].name}')
    for container in $containers; do
        echo "  [$pod/$container]:"
        kubectl logs "$pod" -n "$NAMESPACE" -c "$container" --tail=5 2>/dev/null | sed 's/^/    /'
        echo ""
    done
done

echo "--- NFS Disk Usage (from any running pod) ---"
POD=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$POD" ]; then
    CONTAINER=$(kubectl get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.spec.containers[0].name}')
    kubectl exec "$POD" -n "$NAMESPACE" -c "$CONTAINER" -- df -h /mnt/nfs 2>/dev/null || echo "  Could not check NFS"
    echo ""
    echo "--- Downloaded Data ---"
    kubectl exec "$POD" -n "$NAMESPACE" -c "$CONTAINER" -- bash -c '
        echo "CommonVoice EURO:"
        for d in /mnt/nfs/data/multilingual_v1/raw/commonvoice_17/euro/audio/*/; do
            lang=$(basename "$d")
            count=$(find "$d" -name "*.wav" 2>/dev/null | wc -l)
            echo "  $lang: $count WAVs"
        done
        echo ""
        echo "CommonVoice SLAVIC:"
        for d in /mnt/nfs/data/multilingual_v1/raw/commonvoice_17/slavic/audio/*/; do
            lang=$(basename "$d")
            count=$(find "$d" -name "*.wav" 2>/dev/null | wc -l)
            echo "  $lang: $count WAVs"
        done
        echo ""
        echo "Indo/ZH datasets:"
        for d in /mnt/nfs/data/multilingual_v1/raw/aishell1 /mnt/nfs/data/multilingual_v1/raw/indicvoices/hi /mnt/nfs/data/multilingual_v1/raw/indicvoices/bn /mnt/nfs/data/multilingual_v1/raw/indicvoices/mr /mnt/nfs/data/multilingual_v1/raw/indicvoices/pa /mnt/nfs/data/multilingual_v1/raw/global /mnt/nfs/data/multilingual_v1/raw/betrac; do
            name=$(echo "$d" | sed "s|/mnt/nfs/data/multilingual_v1/raw/||")
            if [ -d "$d" ]; then
                count=$(find "$d" -name "*.wav" 2>/dev/null | wc -l)
                echo "  $name: $count WAVs"
            else
                echo "  $name: not started"
            fi
        done
        echo ""
        echo "Gujarati datasets:"
        for d in /mnt/nfs/data/gujarati_v1/raw/indicvoices /mnt/nfs/data/gujarati_v1/raw/indicvoices_r /mnt/nfs/data/gujarati_v1/raw/kathbath /mnt/nfs/data/gujarati_v1/raw/fleurs; do
            name=$(basename "$d")
            if [ -d "$d" ]; then
                count=$(find "$d" -name "*.wav" 2>/dev/null | wc -l)
                manifest_count=0
                for mf in "$d"/*.json "$d"/*.jsonl; do
                    [ -f "$mf" ] && manifest_count=$((manifest_count + 1))
                done
                echo "  $name: $count WAVs, $manifest_count manifests"
            else
                echo "  $name: not started"
            fi
        done
        if [ -f /mnt/nfs/data/gujarati_v1/train.json ]; then
            train_count=$(wc -l < /mnt/nfs/data/gujarati_v1/train.json)
            valid_count=$(wc -l < /mnt/nfs/data/gujarati_v1/valid.json 2>/dev/null || echo 0)
            echo "  FINAL: train=$train_count valid=$valid_count"
        fi
        echo ""
        echo "English datasets:"
        for d in /mnt/nfs/data/multilingual_v1/raw/en_people /mnt/nfs/data/multilingual_v1/raw/commonvoice_17/english /mnt/nfs/data/english_v1/raw/digits/speech_commands /mnt/nfs/data/english_v1/raw/digits/fsdd; do
            name=$(echo "$d" | sed "s|/mnt/nfs/data/||")
            if [ -d "$d" ]; then
                count=$(find "$d" -name "*.wav" -o -name "*.flac" 2>/dev/null | wc -l)
                manifest_count=0
                for mf in "$d"/*.json "$d"/*.jsonl; do
                    [ -f "$mf" ] && manifest_count=$((manifest_count + 1))
                done
                echo "  $name: $count audio files, $manifest_count manifests"
            else
                echo "  $name: not started"
            fi
        done
        if [ -f /mnt/nfs/data/english_v1/train.json ]; then
            train_count=$(wc -l < /mnt/nfs/data/english_v1/train.json)
            valid_count=$(wc -l < /mnt/nfs/data/english_v1/valid.json 2>/dev/null || echo 0)
            echo "  FINAL: train=$train_count valid=$valid_count"
        fi
        echo ""
        echo "English training:"
        if [ -d /mnt/nfs/experiments/en-meta-v1 ]; then
            latest_ckpt=$(find /mnt/nfs/experiments/en-meta-v1 -name "*.ckpt" -o -name "*.nemo" 2>/dev/null | sort | tail -1)
            echo "  Latest checkpoint: ${latest_ckpt:-none}"
        else
            echo "  Not started"
        fi
    ' 2>/dev/null || echo "  Could not enumerate data dirs"
else
    echo "  No running pod to check NFS from"
fi
