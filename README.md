# MountainCar-v0 solved by DQN

## Files

---- mainfile

MountainCar

----netfile

MountainCar_origin_net
MountainCar_reshape_net
MountainCar_HER_net
MountainCar_RND_net

## Execute

```shell
python MountainCar.py -a [origin, reshape, HER, RND]
```

## Plot

```shell
tensorboard --logdir=runs/MountainCar_curve
```