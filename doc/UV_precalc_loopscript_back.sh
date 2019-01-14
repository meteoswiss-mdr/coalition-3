input_start=$1
input_end=$2

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

d="$startdate"
while [ "$d" != "$enddate" ]; do 
  echo Working on $d
  python /opt/users/jmz/monti-pytroll/scripts/NOSTRADAMUS_1_input_prep_disparr_training.py $d
  d=$(date -I -d "$d - 1 day")
done
