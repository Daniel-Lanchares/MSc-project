# Results

The current development of the project has various semi-open fronts:

- **A chirp mass estimator**, which was the first test of the technology. The chirp mass is directly related to the 
  signals frequency and its derivative, making it the easiest parameter to extract from the data. However, training 
  slowed down significantly after reaching about _logprob 5_. This was attributed to a single parameter model 
  presenting to many degeneracies. In other words, it couldn't understand why two events with the same chirp mass 
  looked completely different, as it wasn't taking into account any other parameters. Results so far have been 
  promising, being capable of reducing prior uncertainty (if asked to predict the chirp mass of an event it gives a 
  more concrete answer than 'between 0 and 80 solar masses', its prior) but unable to compete with official 
  estimations, which again, is to be expected considering those estimations tend to regress over the full 15 
  dimensional parameter space.

<!---
  ![comparison of GW150914](https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_chirp_mass_estimator/comparison_v0.4.2_GW150914.png)
  ![comparison of GW170823](https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_chirp_mass_estimator/comparison_v0.4.2_GW170823.png)
-->  

<img width="500"  src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_chirp_mass_estimator/comparison_v0.4.2_GW150914.png" alt="comparison of GW150914"/>
<img width="500"  src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_chirp_mass_estimator/comparison_v0.4.2_GW170823.png" alt="comparison of GW170823"/>

  Might revisit with more training data to see if it can be improved.


- **A 4 parameter model**. These being the chirp mass, effective spin, luminosity distance and NAP (**N**etwork 
  **A**ntenna **P**attern), related to sky position. Again, a loss wall was found at around _logprob 22_, so not 
  much training was performed. Currently, not being developed.


- **A 5 parameter model**. Where effective spin has been substituted for mass ratio to allow mass reconstruction and 
  the full sky position has been trained over. This is the model family with the best scores so far. This first one was 
  trained with a dataset of 20000 images for its first stage and a further 20000 for its single-epoch second stage. 
  Validation around _logprob 12.8_.

| parameters<br>(flow Spv1.3.1b)   |       median |       truth |   accuracy<br>(MSE) | precision_left<br>(1.0 $\sigma$ ) | precision_right<br>(1.0 $\sigma$ ) | units          |
|----------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                       |   46.3576    |   46.8531   |             6.95897 | 9.29351                           | 8.99562                            | $M_{\odot}$    |
| mass_ratio                       |    0.612215  |    0.613444 |             0.17615 | 0.225936                          | 0.212689                           | $ø$            |
| luminosity_distance              | 1367.16      | 1502.02     |           536.337   | 581.89                            | 693.561                            | $\mathrm{Mpc}$ |
| ra                               |    3.11678   |    3.17589  |             1.21031 | 1.41312                           | 1.43929                            | $\mathrm{rad}$ |
| dec                              |    0.0674703 |    0.016705 |             0.44408 | 0.616734                          | 0.593469                           | $\mathrm{rad}$ |

<img width="1000" alt="estimation of 999.00001" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_5p_special_model/Spv2.3.1b_corner.png">

  **Currently, best performing model is Spv1.4.2.B6 (_lp ~12.1_), which was trained in sub-epochs (to give more detail on memory). 
  This technique gave interesting results, but is difficult to integrate with variable learning rates**.

| parameters<br>(flow Spv1.4.2.B6)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|------------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                         |   46.38      |   46.8531   |            6.17356  |                           7.14091 |                           7.34712  | $M_{\odot}$    |
| mass_ratio                         |    0.592085  |    0.613444 |            0.168598 |                           0.20229 |                           0.232095 | $ø$            |
| luminosity_distance                | 1470.63      | 1502.02     |          512.427    |                         552.524   |                         652.895    | $\mathrm{Mpc}$ |
| ra                                 |    3.05953   |    3.17589  |            1.11721  |                           1.30878 |                           1.2566   | $\mathrm{rad}$ |
| dec                                |   -0.0498375 |    0.016705 |            0.404739 |                           0.48882 |                           0.533604 | $\mathrm{rad}$ |

<img width="1000" alt="estimation of 999.00001" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_5p_special_model/Spv1.4.2.B6_corner.png">

- **A 6 parameter model**. These being chirp mass, mass ratio, effective spin, luminosity distance, right ascension 
  and declination. The models showed promise training to _logprob ~14_, but have been difficult to train any further.
  
  Performed an overfitting test to know whether the distributions **can** be learnt, even if that model wouldn't 
  be able to predict anything new. To do this I started from a partially trained flow and fed it the same dataset of 
  around 1000 images over and over. It currently stands at _logprob ~7.5_, but it is fairly clear that the model can 
  learn the necessary distributions.

| dataset 32 | MSE from overfitting_test | units       |
|:-----------|--------------------------:|:------------|
| chirp_mass |                  5.207210 | $M_{\odot}$ |
| mass_ratio |                  0.118419 | ø           |
| chi_eff    |                  0.145676 | ø           |
| d_L        |                203.664000 | Mpc         |
| ra         |                  0.776586 | rad         |
| dec        |                  0.433297 | rad         |

<img width="1000" alt="estimation of 32.00005" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_6p_model/Overfitting_32.00005_logprob_8.20.png">

The apparent proneness to overfitting motivated me to increase number of parameters and dataset size. However, 
increases on the parameter-space dimension also imply more difficulty avoiding exploding gradients on the first batches
of training. **Right now I have a 7p highly overfitted model (_logprob ~3.7_ on train data) with 60k dataset**.
These models are created interlacing affine and rq-coupling transforms.

| parameters<br>(flow Spv2.11.0c)   |       median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|--------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   47.2296    |   46.432     |           2.49937   |                         2.37798   |                          2.34017   | $M_{\odot}$    |
| mass_ratio                        |    0.607361  |    0.595741  |           0.0757192 |                         0.0757643 |                          0.0785897 | $ø$            |
| chi_eff                           |    0.0343048 |    0.015586  |           0.0618838 |                         0.0594163 |                          0.0580426 | $ø$            |
| luminosity_distance               | 1477.59      | 1450.36      |         208.641     |                       219.281     |                        220.717     | $\mathrm{Mpc}$ |
| theta_jn                          |    1.50883   |    1.58521   |           0.436029  |                         0.460305  |                          0.531358  | $\mathrm{rad}$ |
| ra                                |    3.06231   |    3.10922   |           0.443229  |                         0.420114  |                          0.416669  | $\mathrm{rad}$ |
| dec                               |    0.0485658 |    0.0304754 |           0.161322  |                         0.148832  |                          0.152212  | $\mathrm{rad}$ |

<img width="1000" alt="estimation of 32.00001" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_7p_model/Spv2.11.0c_corner.png">

By rescaling some of the parameters to avoid large numbers training has become much more stable, enabling similar 
results as before but on validation data:

| parameters<br>(flow Spv2.12.0e)   |        median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|---------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   46.9626     |   46.8531    |            6.06843  |                          6.50902  |                           6.47826  | $M_{\odot}$    |
| mass_ratio                        |    0.59587    |    0.613444  |            0.165765 |                          0.186067 |                           0.214523 | $ø$            |
| chi_eff                           |    0.0245531  |    0.0106512 |            0.149719 |                          0.178095 |                           0.174689 | $ø$            |
| luminosity_distance               | 1473.97       | 1502.02      |          505.016    |                        557.215    |                         637.261    | $\mathrm{Mpc}$ |
| theta_jn                          |    1.59081    |    1.54645   |            0.627239 |                          0.814805 |                           0.784108 | $\mathrm{rad}$ |
| ra                                |    3.10284    |    3.17589   |            1.10779  |                          1.32065  |                           1.16288  | $\mathrm{rad}$ |
| dec                               |    0.00524479 |    0.016705  |            0.409863 |                          0.47796  |                           0.513514 | $\mathrm{rad}$ |

<img width="1000" alt="estimation of 32.00001" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_7p_model/Spv2.12.0e_corner.png">

A 10 parameter model has been attempted, but first training batches proved challenging until the introduction of rescaling. 
Currently at:

-Validation data
<img width="1000" alt="estimation of 999.00001" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_10p_model/Spv3.0.0_corner.png">

-Training data
<img width="1000" alt="estimation of 999.00001" src="https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_10p_model/Spv3.0.0_32.1_corner.png">