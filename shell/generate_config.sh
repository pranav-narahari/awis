#!/bin/bash

echo "creating config.json\n"
#need to change the directory
cat > ~/Documents//Projects/TES\ EI\ E2E/AWIS/scripts/General/config.json<< EOF 

{
  "DEFAULT_IP": "0.0.0.0",
  "DEFAULT_PORT": 32022,
  "DEFAULT_CAMERA": 3,
  "DEFAULT_MODEL": "AWIS_yolov5n_edgetpu.tflite",
  "DEFAULT_LABELS": "AWIS.yaml",
  "DEFAULT_CONFIDENCE": "0.90",
  "DEFAULT_INTERSECTION": "0.45",
  "AWS_ACCESS_KEY": "",
  "AWS_SECRET_KEY": "",
  "IMAGE_STORAGE_PATH": "/media/flash/ImageData",
  "VIDEO_STORAGE_PATH": "/media/flash/VideoData",
  "VIDEO_RECORD_FRAME_RATE": "4",
  "START_DETECTION_AT_LAUNCH": 1,
  "START_TRACKING_AT_LAUNCH": 1,
  "START_IMAGE_CAPTURE_AT_LAUNCH": 0,
  "START_MOVING_CAPTURE_AT_LAUNCH": 0,
  "MIN_MOVE_DIST_FOR_IMAGE_CAPTURE": 20,
  "DISABLE_FLASK_SERVER": 0
}


EOF

echo "--- CONFIG CREATED ---"