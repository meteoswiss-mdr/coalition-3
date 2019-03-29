# User arguments of script_tds_processing.py:
# 1. argument is standard output to file, if missing to console
# 2. argument (optional) is month, if a respective
#    Training_Dataset_Processing_Status_8.pkl exists

AddVal=100
RetVal=201
while [ $RetVal != 200 ]; do 
  python script_tds_processing.py std2file 8
  RetVal=$?
  if [ $RetVal != 0 ]
  then
    echo Percentage finished: $(( RetVal - AddVal ))%
    now=$(date +"%T")
    echo Current time: $now
  fi
done
