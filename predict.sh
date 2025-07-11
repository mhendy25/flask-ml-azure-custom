#!/usr/bin/env bash

PORT=5001
echo "Port: $PORT"

# POST method predict with California Housing features
curl -d '{
   "MedInc": {"0": 8.3252},
   "AveRooms": {"0": 6.984},
   "AveBedrms": {"0": 1.023},
   "Population": {"0": 322.0},
   "AveOccup": {"0": 2.555},
   "Latitude": {"0": 37.88}
}'\
     -H "Content-Type: application/json" \
     -X POST http://localhost:$PORT/predict