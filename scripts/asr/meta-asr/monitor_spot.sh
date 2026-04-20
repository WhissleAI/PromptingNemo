#!/bin/bash
# Monitor spot instance and auto-restart after preemption
# Usage: nohup ./monitor_spot.sh &

INSTANCE="nemo-hindi-a100"
ZONE="us-central1-c"
PROJECT="deepvoice-468015"
CHECK_INTERVAL=300  # 5 minutes

echo "$(date): Monitoring $INSTANCE (check every ${CHECK_INTERVAL}s)"

while true; do
    STATUS=$(gcloud compute instances describe $INSTANCE \
        --zone=$ZONE --project=$PROJECT \
        --format='get(status)' 2>/dev/null)

    if [ "$STATUS" != "RUNNING" ]; then
        echo "$(date): Instance is $STATUS — restarting..."
        gcloud compute instances start $INSTANCE \
            --zone=$ZONE --project=$PROJECT 2>&1
        echo "$(date): Start command sent. Waiting 120s for boot..."
        sleep 120
    fi
    sleep $CHECK_INTERVAL
done
