#！/bin/bash

for ((i=0; i<=4; i++)) 
do
    echo $i >> timing.txt
    python3 main.py >> timing.txt
done