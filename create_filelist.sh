#!/usr/bin/env sh
DATA=DRclass4sxd

echo "Create trainDR.txt..."
rm -rf trainDR.txt
for i in 0 1 2 3 4 
do
find $DATA/train -name $i*.jpeg | cut -d '/' -f4-5 | sed "s/$/ $i/">>trainDR.txt
done
echo "Create testDR.txt..."
rm -rf testDR.txt
for i in 0 1 2 3 4
do
find $DATA/test -name $i*.jpeg | cut -d '/' -f4-5 | sed "s/$/ $i/">>testDR.txt
done
echo "All done"
