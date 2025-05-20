import asyncio
import datetime
import os
import alsaaudio
import numpy as np
import math
import operator
from birdnet_analyzer.analyze.utils import load_codes, predict
import birdnet_analyzer.utils as utils
import birdnet_analyzer.config as cfg
from birdnet_analyzer.species.utils import get_species_list
import sqlite3
import yaml
from zoneinfo import ZoneInfo
from uuid_extensions import uuid7str
from pathlib import Path
import pydub

def confirm_dir(dir: Path):
    if not dir.is_dir():
        os.mkdir(dir)
    return True

def startup(w_dir):
    recording_dir = w_dir / 'recordings'
    confirm_dir(recording_dir)

    db_dir = w_dir / 'db'
    confirm_dir(db_dir)

    script_dir = w_dir / 'BirdNET-Analyzer'

    cfg.MODEL_PATH = os.path.join(script_dir, cfg.MODEL_PATH)
    cfg.LABELS_FILE = os.path.join(script_dir, 'birdnet_analyzer/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.txt')
    cfg.TRANSLATED_LABELS_PATH = os.path.join(script_dir, cfg.TRANSLATED_LABELS_PATH)
    cfg.MDATA_MODEL_PATH = os.path.join(script_dir, cfg.MDATA_MODEL_PATH)
    cfg.CODES_FILE = os.path.join(script_dir, cfg.CODES_FILE)
    cfg.ERROR_LOG_FILE = os.path.join(script_dir, cfg.ERROR_LOG_FILE)
    cfg.TFLITE_THREADS = 1
    cfg.MIN_CONFIDENCE = 0.05
    cfg.SIGMOID_SENSITIVITY = 1.0
    cfg.CPU_THREADS = 1
    cfg.SAMPLE_RATE = 48000
    cfg.CODES = load_codes()
    cfg.LABELS = utils.read_lines(cfg.LABELS_FILE)
    cfg.TRANSLATED_LABELS = cfg.LABELS
    cfg.LOCATION_FILTER_THRESHOLD = 0.03
    cfg.WEEK = datetime.date.today().isocalendar()[1] # could change this to read from the recording file name


    conf = yaml.safe_load(open('config.yaml'))
    cfg.LATITUDE = conf['latitude']
    cfg.LONGITUDE = conf['longitude']
    cfg.SPECIES_LIST = get_species_list(cfg.LATITUDE, cfg.LONGITUDE, cfg.WEEK, cfg.LOCATION_FILTER_THRESHOLD)


    # add database configuraiton
    database = db_dir / 'bird-observations.db'
    table_name = 'observations'
    columns = 'id text, ts integer, scientific_name txt, common_name txt, confidence real'
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS {table_name} ({columns})""".format(table_name=table_name, columns=columns))
    connection.commit()
    connection.close()

def convert_frame_to_signal(frames):
    signal = np.frombuffer(b''.join(frames), dtype=np.int16)
    return signal

def save_audio(species_name, signal, start_ts, w_dir):
    a = pydub.AudioSegment(signal.tobytes(), frame_rate=cfg.SAMPLE_RATE, sample_width=2, channels=1)
    n = 'ts_' + str(start_ts) + '.flac'
    d = w_dir / 'recordings' / species_name
    confirm_dir(d)
    fn = d / n
    a.export(fn, format="flac")
    return True

def analyse_recording(start_ts, signal):

    # Convert standard signal to [-1,1] range
    scaled_sig = signal / 32768

    # Run the analysis
    pred = predict([scaled_sig])[0]

    # Assign scores to labels
    p_labels = zip(cfg.LABELS, pred)

    # Sort by score
    p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)

    # Filter for scores above the confidence threshold
    predictions = list(filter(lambda x: (x[1] > cfg.MIN_CONFIDENCE), p_sorted))


    # SAVE THE RESULTS
    if len(predictions) == 0:
        return []
    else:
        # pull the timestamp from the file name
        print('Detected {x} sound'.format(x=len(predictions)))

        ## save to the db
        database = '../db/bird-observations.db'
        table_name = 'observations'
        connection = sqlite3.connect(database)
        cursor = connection.cursor()

        # reformat the data
        data_to_insert = []
        filtered_detections = []
        for p in predictions:
            if p[1] > cfg.MIN_CONFIDENCE and (not cfg.SPECIES_LIST or p[0] in cfg.SPECIES_LIST):
                species_names = p[0].split('_')
                print(species_names[1] + ' detected')
                data_to_insert.append((uuid7str(), start_ts, species_names[0], species_names[1], p[1]))
                filtered_detections.append(p[0])
            else:
                pass

        if len(data_to_insert) > 0:
            # insert to table and commit change
            cursor.executemany('INSERT INTO {table_name} VALUES(?, ?, ?, ?, ?)'.format(table_name=table_name), data_to_insert)
            connection.commit()

            # close the connection
            connection.close()
        else:
            pass

        return filtered_detections

async def recording_worker(queue, input, num_periods=140):
    while True:
        start_ts = int(datetime.datetime.now(tz=ZoneInfo('UTC')).timestamp())
        frames = []
        for i in range(num_periods):
            l, data = input.read()
            frames.append(data)
            await asyncio.sleep(0.01)

        print("Completed recording loop")
        await queue.put((start_ts, frames))


async def analysis_worker(queue, max_queue_size, w_dir):
    while True:
        print('Consumer started')
        # Get a "work item" out of the queue.
        start_ts, frames = await queue.get()
        print('picked  up item off queue')
        if queue.qsize() <= max_queue_size:
            # analyse the file
            signal = convert_frame_to_signal(frames)
            detections = analyse_recording(start_ts, signal)

            # save the file
            # EXCLUSION_LIST = ['Trichoglossus moluccanus_Rainbow Lorikeet','Cacatua galerita_Sulphur-crested Cockatoo','Strepera graculina_Pied Currawong','Alisterus scapularis_Australian King-Parrot','Fulica atra_Eurasian Cootz','Gymnorhina tibicen_Australian Magpie']
            # if len(detections) == 1 and detections[0] not in EXCLUSION_LIST:
            #     save_audio(detections[0], signal, start_ts, w_dir)

            print('Analysed signal')

        await asyncio.sleep(0.5)
        # Notify the queue that the "work item" has been processed.
        queue.task_done()

async def main():
    w_dir = Path.cwd().parent
    startup(w_dir)
    max_queue_size=6
    rate = cfg.SAMPLE_RATE
    record_seconds = 3
    period_size = 1024
    queue = asyncio.Queue()

    input = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL,
		channels=1, rate=rate, format=alsaaudio.PCM_FORMAT_S16_LE,
		periodsize=period_size, device='default')
    period_size = input.info().get('period_size', period_size)
    s = math.floor(record_seconds * (rate / period_size))

    await asyncio.gather(asyncio.create_task(analysis_worker(queue, max_queue_size, w_dir)), recording_worker(queue, input, s))

    # Shut the stream
    input.close()


if __name__ == "__main__":
    asyncio.run(main())
