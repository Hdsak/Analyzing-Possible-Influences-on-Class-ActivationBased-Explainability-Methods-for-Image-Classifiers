IN_PATH=$1
OUT_PATH=$2

for DATA in ACRIMA DIARETDB0 DIARETDB1 DRHAGIS HRFAU KDRD LAG   
do
    echo $DATA
    python ${DATA}_converter.py --in_path ${IN_PATH}/${DATA} --out_path ${OUT_PATH}/${DATA} --preprocessing kdrd
done

for DATA in ODIR REFUGE RIGA RIM STARE Takahashi
do
    echo $DATA
    python ${DATA}_converter.py --in_path ${IN_PATH}/${DATA} --out_path ${OUT_PATH} --preprocessing kdrd
done