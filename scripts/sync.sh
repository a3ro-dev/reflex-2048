#!/usr/bin/env bash
# push the repo to the pod: scripts/sync.sh
source "$(dirname "$0")/pod.env"
tar czf - --exclude=_ext --exclude=.git --exclude=data --exclude=runs . | \
  ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $POD_PORT root@$POD_HOST 'mkdir -p /root/reflex2048 && tar xzf - --no-same-owner -C /root/reflex2048'
