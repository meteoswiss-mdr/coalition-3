dot="$(cd "$(dirname "$0")"; pwd)"
#path="$dot/coalition3"
path="$dot/"
echo   Addin path $path to PYTHONPATH env variable in .bash_profile
echo export 'PYTHONPATH=$PYTHONPATH:'$path  >> ~/.bash_profile
source ~/.bash_profile
