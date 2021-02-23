#!/bin/sh
cd ../MirrAI_UI/

python3 ../stylesense/style_sense.py &
P1=$!
npm start &
P2=$!
wait $P1 $P2