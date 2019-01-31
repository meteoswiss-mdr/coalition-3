AddVal=100
RetVal=201
while [ $RetVal != 200 ]; do 
  python script_tds_processing.py
  RetVal=$?
  if [ $RetVal != 0 ]
  then
    echo Percentage finished: $(( RetVal - AddVal ))%
    now=$(date +"%T")
    echo Current time: $now
  fi
done
