AddVal=100
RetVal=201
while [ $RetVal != 200 ]; do 
  python /opt/users/jmz/monti-pytroll/scripts/NOSTRADAMUS_0_training_ds.py
  RetVal=$?
  if [ $RetVal != 0 ]
  then
    echo Percentage finished: $(( RetVal - AddVal ))%
    now=$(date +"%T")
    echo Current time: $now
  fi
done
