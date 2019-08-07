#!/bin/bash
dt='CIFAR10'
sd=./out
bs=128
lr=0.001
ar='RESNET18'
lf='CROSSENTROPY'
op='ADAM'
lrs=0.2
lri=30000
iters=100000
bts=1.05
ql=2
wd=0.0005

## REF
mt='REF'
lr=0.1
op='SGD'
mm=0.90
echo python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --momentum $mm --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd
python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --momentum $mm --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd

## PMF
mt='PMF'
lr=0.001
lrs=0.2
bts=1.02
op='ADAM'
wd=0.0001
echo python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql
python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql

## PGD
mt='PGD'
lr=0.1
lrs=0.2
bts=1.02
op='SGD'
mm=0.90
wd=0.0001
echo python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql --momentum $mm
python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --beta-scale $bts --quant-levels $ql --momentum $mm

## PICM
mt='PICM'
lr=0.0001
lrs=0.2
op='ADAM'
wd=0.0001
echo python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --quant-levels $ql
python dnets.py --save-dir $sd --learning-rate $lr --architecture $ar --loss-function $lf --method $mt --optimizer $op --dataset $dt --batch-size $bs --lr-scale $lrs --lr-interval $lri --num-iters $iters --weight-decay $wd --quant-levels $ql

