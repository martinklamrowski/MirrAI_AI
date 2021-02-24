#!/bin/sh
cd ../MirrAI_UI/

python3 ../stylesense/style_sense_edgetpu.py &
P1=$!
npm start &
P2=$!
wait $P1 $P2