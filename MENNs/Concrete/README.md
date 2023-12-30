# Concrete FHE-MENNs 

These MENNs are based on the Concrete library.

To train the backbone from scratch, run
```
python3 bnn_pynq_train.py --data ./data --experiments ./experiments 
```

To to train each exit and fine tune the backbone, run:
```
python3 bnn_pynq_train.py --data ./data --experiments ./experiments --resume ./experiments/experiment_name --exit 0
``` 

And to evaluate the model
```
python3 evaluate_menn.py --model ./experiments/resnet_modded/checkpoints/best.tar --rounding 6
```
 
