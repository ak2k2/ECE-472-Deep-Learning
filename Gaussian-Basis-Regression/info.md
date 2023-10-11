# Assignment 1 📄
---
Approximate a noisy sine wave as a linear combination of Gaussian basis functions. Use Stochastic Gradient Descent (SGD) and automatic differentiation to compute and optimize weights and biases for Linear and BasisExpansion Model Classes.
---

## Results

My script produces a dynamic plot (1 frame / 10 iterations) of the evolution of the estimated sine wave. The constituent basis functions are plotted as overlays, and the weights and bias of the Linear model, as well as the parameters of the basis function, are displayed on the top left. 

https://github.com/ak2k2/ECE-472-Deep-Learning/assets/103453421/50c9dd59-4d07-4691-b70b-d4085e537a09

## Aliasing Error (Gibbs Phenomenon?)
```
"NUM_DATA_POINTS": 50,
"NUM_BASIS_FUNCTIONS": 20,
```

![Absolute error wrt. X over a denser testing set](https://github.com/ak2k2/ECE-472-Deep-Learning/assets/103453421/8ab7c80f-b740-4bce-9951-aa7bc398dcd4)

- 2023-09-05 02:37:02,968 - __main__ - INFO - Final loss: 0.0024988707154989243
- 2023-09-05 02:37:02,969 - __main__ - INFO - Final R^2: 0.9899427890777588

**Looks like Frequency vs Magnitude response of FIR/IIR filter...**

---

```
"NUM_DATA_POINTS": 500,
"NUM_BASIS_FUNCTIONS": 100,
```

![outside_range](https://github.com/ak2k2/ECE-472-Deep-Learning/assets/103453421/1844569d-28cc-4ab4-b8d8-6ce1b380be36)
