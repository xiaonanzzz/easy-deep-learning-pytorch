BASEDIR=$(dirname "$0")

docker build -t easydl:test316 -f $BASEDIR/Dockerfile .