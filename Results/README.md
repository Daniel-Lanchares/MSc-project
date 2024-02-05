# Results

The current development of the project has three semi-open fronts:

- **A chirp mass estimator**, which was the first test of the technology. The chirp mass is directly related to the 
  signals frequency and its derivative, making it the easiest parameter to extract from the data. However, training 
  slowed down significantly after reaching about _logprob 5_. This was attributed to a single parameter model 
  presenting to many degeneracies. In other words, it couldn't understand why two events with the same chirp mass 
  looked completely different, as it wasn't taking into account any other parameters. Results so far have been 
  promising, being capable of reducing prior uncertainty (if asked to predict the chirp mass of an event it gives a 
  more concrete answer than 'between 0 and 80 solar masses', its prior) but unable to compete with official 
  estimations, which again, is to be expected considering those estimations tend to regress over the full 15 
  dimensional parameter space.

  ![comparison of GW150914](https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_chirp_mass_estimator/comparison_v0.4.2_GW150914.png)
  ![comparison of GW170823](https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_chirp_mass_estimator/comparison_v0.4.2_GW170823.png)
  
  Might revisit with more training data to see if it can be improved.


- **A 4 parameter model**. These being the chirp mass, effective spin, luminosity distance and NAP (**N**etwork 
  **A**ntenna **P**attern), related to sky position. Again, a loss wall was found at around _logprob 22_, so not 
  much training was performed


- **A 6 parameter model**. These being chirp mass, mass ratio, effective spin, luminosity distance, right ascension 
  and declination. The models showed promise training to _logprob ~14_, but have been difficult to train any further.
  
  Currently doing an overfitting test to know whether the distributions **can** be learnt, even if that model wouldn't 
  be able to predict anything new. To do this I started from a partially trained flow and fed it the same dataset of 
  around 1000 images over and over. It currently stands at _logprob ~7.5_.

| dataset 32 | MSE from overfitting_test | units       |
|:-----------|--------------------------:|:------------|
| chirp_mass |                  5.207210 | $M_{\odot}$ |
| mass_ratio |                  0.118419 | ø           |
| chi_eff    |                  0.145676 | ø           |
| d_L        |                203.664000 | Mpc         |
| ra         |                  0.776586 | rad         |
| dec        |                  0.433297 | rad         |

![estimation of 32.00005](https://raw.githubusercontent.com/Daniel-Lanchares/MSc-project/main/Results/Pictures_6p_model/Overfitting_32.00005_logprob_8.20.png)