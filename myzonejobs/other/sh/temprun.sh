#!/bin/bash

for date in 20220219 20220205 20220115 20211218 20211205 20211122 20211120 20211113 20211103; do
    src_dir="/media/xiaoran/XXRMatrix/MyZoneRecording/$date/converted"
    dst_dir="/media/xiaoran/XWestWorld/MyZoneRecordings/$date/converted"
    mkdir -p $dst_dir

    for video in $src_dir/*.mov; do
        echo $video
        cp -p $video $dst_dir/
        sleep 10
    done
done