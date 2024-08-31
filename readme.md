

## Installation
Copy config.template.yaml to config.yaml and input latitude and longitude

Install ffmpeg
> sudo apt-get install ffmpeg

Clone the BirdNet-Analyzer repository
> git clone https://github.com/kahst/BirdNET-Analyzer.git
(this will take a while)

Copy the `record-analyse.py` script into the BirdNET-Analyzer directory
> scp /path/to/file 

Create a virtual environment
> python -m venv venv

Activate the venv
> source venv/bin/activate

> pip install tflite-runtime librosa asyncio uuid7 pydub

Is librosa still needed?

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
    ExecStart=/home/operator/bird-listener/venv/bin/python /home/operator/bird-listener/BirdNET-Analyzer/recordAnalyse.py   
    WorkingDirectory=/home/operator/bird-listener/BirdNET-Analyzer  
    Restart=on-failure   
    RestartSec=5s   
    
    [Install]
    WantedBy=multi-user.target

> sudo systemctl enable bird-listener.service

> sudo systemctl start bird-listener.service

Check that your audio is configured correctly, specifcally the default mic
> alsamixer