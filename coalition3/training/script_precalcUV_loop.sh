type=$1
input_start=$2
input_end=$3

# Check type input
if [ "$1" == "operational" ] || [ "$1" == "training" ]; then
  echo Run in mode "$type"
else
  echo "  Failed: Type not understood, use either 'operational' or 'training'"
  exit 0
fi

# Check date input
if [ "$input_start" == "" ] || [ "$input_end" == "" ]; then
  echo "  Failed: At least one of the date parameters is empty"
  exit 0
elif [ "$input_start" == "$input_end" ]; then
  echo "  Failed: Same date (to pre-calculate first date set second date +1day)"
  exit 0
fi

# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd
startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

d="$startdate"
while [ "$d" != "$enddate" ]; do 
  echo Working on $d
  python script_precalcUV.py $d $type
  d=$(date -I -d "$d + 1 day")
done
