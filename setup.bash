#!/usr/bin/env bash

###########################Downloading required files###############################

echo "Copying files..."

cp Code/classify_track_count.py ../.
cp Code/GUIApp.py ../.
cp Code/SortTracker.py ../.

cp Code/extras/*.cfg ../cfg
cp Code/extras/*.names ../data
cp Code/extras/*.data ../data

#########################################################################