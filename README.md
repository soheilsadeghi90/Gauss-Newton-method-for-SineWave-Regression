# Gauss-Newton-method-for-SineWave-Regression
A demo for sinusoidal regression using Gauss_Newton method
The file is a demo for regressing a Sine function over a noisy data. 

In general, linear optimization methods such as GD cannot get to optimal point for harmonic functions, because of their nonconvex nature

One successful approach for these problems is to use second-order methods. In this illustration, a very simple problem is designed to show 
how accurate a second-order can optimize a SineWave regression line over a harmonic dataset

The problem is adopted from:
https://math.stackexchange.com/questions/301194/given-a-data-set-how-do-you-do-a-sinusoidal-regression-on-paper-what-are-the-e

Note: you can play with parameters of the underlying harmonic function, but the optimization method will loss its stability. As a hint, 
for unstable cases, initalize parameters closer to the actual values. Function value plot is showing how intense the function nonconvexity is!
