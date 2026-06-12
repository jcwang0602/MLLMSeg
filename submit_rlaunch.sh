rlaunch --memory=32000 --cpu=32 \
    --charged-group=mineru4sh_gpu --private-machine=yes \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --mount=gpfs://gpfs1/wangjingchao:/mnt/shared-storage-user/wangjingchao \
    -- sudo pip install -r requirements.txt
