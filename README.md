# Consistent Koopman Autoencoders

Research code that demonstrates consistent Koopman autoencoders on the nonlinear pendulum with no friction.



##  Reference
[Forecasting Sequential Data Using Consistent Koopman Autoencoders](https://arxiv.org/pdf/2003.02236.pdf)

python driver.py --dataset pendulum --folder results_pendulum --bottleneck 6 --backward 0

python driver.py --dataset pendulum --folder results_back_pendulum --bottleneck 6 --backward 1
