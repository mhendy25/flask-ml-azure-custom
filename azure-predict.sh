#!/usr/bin/env bash

PORT=443
echo "Port: $PORT"

# POST method predict
curl -d '{
   "MedInc": {"0": 8.3252},
   "AveRooms": {"0": 6.984},
   "AveBedrms": {"0": 1.023},
   "Population": {"0": 322.0},
   "AveOccup": {"0": 2.555},
   "Latitude": {"0": 37.88}
}'\
     -H "Content-Type: application/json" \
     -X POST https://medohendy-flask-ml-service.azurewebsites.net:$PORT/predict 
     #your application name <yourappname>goes here