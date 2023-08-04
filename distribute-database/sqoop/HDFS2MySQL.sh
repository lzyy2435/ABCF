#!/bin/sh

echo "start load data from MySQL to HDFS....."
cd /opt/sqoop

while :
do
    day=$(date "+%Y-%m-%d %H:%M:%S") 
    echo "[$day]start load data to HDFS....."
    bin/sqoop export --connect jdbc:mysql://127.0.0.1:3306/MyDB --username root --password 123456 --table hd --num-mappers 1 --export-dir /fromMySQL/ --input-fields-terminated-by "\t"
    sleep $1
done



