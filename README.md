# RC-NF: Reservoir Computing with Normalizing Flow
 This is anopen-source codebase about the article " ".
***
* The folder "nolitsa" is used to calculate the maximum Lyapunov exponent. We refer to the [NoLiTSA](https://github.com/manu-mannattil/nolitsa "NoLiTSA") open-source library.
* The folder "src" involves the RC framework (`RC.py`), ESC framework (`ESC.py`), and NF framework (`flows.py` `models.py` `utls.py`).
  * RC and ESC involve the training, prediction, calculation of single-step errors, and generating new trajectories.
  * NF provides training models and generating samples, as well as different transformation maps. We refer to the open-source code libraries: [normalizing-flows](https://github.com/tonyduan/normalizing-flows "tonyduan/normalizing-flows") and [Temporal-normalizing-flows-for-SDEs](https://github.com/Yubin-Lu/Temporal-normalizing-flows-for-SDEs "Yubin-Lu/Temporal-normalizing-flows-for-SDEs")
***
* We list three experiments mentioned in the article:
  * We consider a 1-D Double-Well system with Brownian motion. `RC-NF-DW.ipynb`
  * We consider a 1-D El Niño Southern Oscillation simplified model with additive Gaussian noise. `RC-NF-ESNO.ipynb`
  * We consider a 3-D El Lorenz system with additive Gaussian noise. `RC-NF-Lorenz.ipynb`
***
* An algorithm flowchart of RC-NF is shown below.

 





