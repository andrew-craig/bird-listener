
## Overview

This project is a stripped down adaptation of [BirdNetAnalyzer](https://github.com/birdnet-team/BirdNET-Analyzer) intended for running the classification model on a lightweight devices like the Raspberry Pi Zero 2 W. It does not support training.



## Installation
Copy config.template.yaml to config.yaml and input latitude and longitude

Download the models
> mkdir models
> cd models
> wget https://drive.google.com/file/d/1ixYBPbZK2Fh1niUQzadE2IWTFZlwATa3
> unzip -q V2.4.zip
> mv V2.4/* .
> rmdir V2.4

Create a virtual environment
> python -m venv venv

Activate the venv
> source venv/bin/activate

> pip install -r requirements.txt

Navigate to the systemd directory
> cd /etc/systemd/system

Create a new file `bird-listener.service`
> sudo nano bird-listener.service

Copy the below into the file. Check the username
    [Unit]
    Description=Service to listen for birds via a mic
    StartLimitIntervalSec=300
    StartLimitBurst=5

    [Service]
    User=operator
    Group=operator
    SupplementaryGroups=audio
    ExecStart=/home/operator/bird-listener/venv/bin/python /home/operator/bird-listener/recordAnalyse.py
    WorkingDirectory=/home/operator/bird-listener
    Restart=on-failure
    RestartSec=5s
    # Fewer glibc malloc arenas: less memory fragmentation from worker threads
    Environment=MALLOC_ARENA_MAX=2
    # Hardening: a mic listener needs no root and only writes to db/
    NoNewPrivileges=true
    ProtectSystem=strict
    ReadWritePaths=/home/operator/bird-listener/db
    PrivateTmp=true
    ProtectKernelTunables=true
    ProtectKernelModules=true
    ProtectControlGroups=true

    [Install]
    WantedBy=multi-user.target

The service runs as `operator` (not root) and can only write to `db/`, so create that directory and make sure the user can use the mic
> mkdir -p /home/operator/bird-listener/db

> sudo chown -R operator:operator /home/operator/bird-listener/db

> sudo usermod -aG audio operator

> sudo systemctl enable bird-listener.service

> sudo systemctl start bird-listener.service

Check that your audio is configured correctly, specifcally the default mic
> alsamixer
