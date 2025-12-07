
## Overview

This project is a stripped down adaptation of [BirdNetAnalyzer](https://github.com/birdnet-team/BirdNET-Analyzer) intended for running the classification model on a lightweight devices like the Raspberry Pi Zero 2 W. It does not support training.



## Installation
Copy config.template.yaml to config.yaml and input latitude and longitude

Install ffmpeg
> sudo apt-get install ffmpeg

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
    ExecStart=/home/operator/bird-listener/venv/bin/python /home/operator/bird-listener/recordAnalyse.py
    WorkingDirectory=/home/operator/bird-listener
    Restart=on-failure
    RestartSec=5s

    [Install]
    WantedBy=multi-user.target

> sudo systemctl enable bird-listener.service

> sudo systemctl start bird-listener.service

Check that your audio is configured correctly, specifcally the default mic
> alsamixer
