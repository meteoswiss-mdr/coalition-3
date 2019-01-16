dot="$(cd "$(dirname "$0")"; pwd)"
path="$dot/code"
echo   Addin path $path to PYTHONPATH env variable in .bash_profile
echo export 'PYTHONPATH=$PYTHONPATH:'$path  >> ~/.bash_profile
