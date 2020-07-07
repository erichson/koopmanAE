# Consistent Koopman Autoencoders

Research code that demonstrates consistent Koopman autoencoders on the nonlinear pendulum with no friction.

Here is an example:
``` 
python driver.py --dataset pendulum --folder results_back_pendulum --bottleneck 6 --backward 1
```

You can also create a baseline model:
```
python driver.py --dataset pendulum --folder results_pendulum --bottleneck 6 --backward 0
```

Use the following code to plot the results:
```
python plot_pred_error.py
```

<img src="https://github.com/erichson/koopmanAE/blob/master/plot/pred_pendulum.png" width="800">


##  Reference
[Forecasting Sequential Data Using Consistent Koopman Autoencoders](https://arxiv.org/pdf/2003.02236.pdf)


