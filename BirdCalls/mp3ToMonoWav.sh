#!/bin/bash
rm $1/*.wav
for f in $1/*.mp3
do
	NAME=$(basename $f .mp3)
	echo $NAME
	ffmpeg -i $f -acodec pcm_s16le -ac 1 -ar 20000 $1/$NAME.wav &
done

wait
