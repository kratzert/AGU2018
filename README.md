# Do internals of neural networks make sense in the context of hydrology?

This repository will contain the documented code to reproduce any results presented at the AGU 18 fall meeting.

## Authors
Frederik Kratzert¹, Mathew Herrnegger², Daniel Klotz¹, Sepp Hochreiter¹, Günter Klambauer¹

- ¹: Institute for Machine Learning, Johannes Kepler University, Linz, Austria
- ²: Institute for Hydrology and Water Management, University of Natural Resources and Life Sciences, Vienna, Austria


## Scientific Abstract

In recent years, neural networks gained a new wave of popularity in many application domains, such as computer vision or natural language processing. However in applied environmental sciences, like rainfall-runoff modelling in hydrology, neural networks tend to have a rather bad reputation. This can be attributed to their _black-box-ness_ and the difficulty or impossibility to understand network internals leading to a prediction.

In this study, we tackle this criticism and show how recent advances and methods in the area of interpretable machine learning allow for deeper analysis of network internals. As an example we use the Long Short-Term Memory network (LSTM) for the task of rainfall-runoff modelling. LSTMs are a special kind of recurrent neural network architecture and, as the name suggests, are able to learn short- and long-term dependencies between inputs and outputs due to special memory cells. This property is especially appreciable in hydrology, where outputs of the system tend to have a long memory. Kratzert et al. (2018) confirm this and have recently shown that LSTMs can achieve competitive results compared to the well established Sacramento Soil Moisture Accounting model, coupled with the Snow-17 snow module.

In this contribution, we explore the internals of trained networks for different basins from the publicly available CAMELS data set, also utilizing remote sensing data of soil moisture, snow cover and evapotranspiration fluxes. We show that LSTMs internally learn to represent patterns that match our understandings of the hydrological system. In snow-driven catchments e.g. the network develops special memory cells that mimic conceptual snow storages with annual dynamics, as known from process-based catchment models.

In general, we anticipate that our findings can increase the trust in the application of neural networks in environmental sciences. We however stress, that expert knowledge on the modeled system is inevitable and still remains an important quality against black-box-ness.

References:

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and Herrnegger, M.: Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks, Hydrol. Earth Syst. Sci., 22, 6005-6022, https://doi.org/10.5194/hess-22-6005-2018, 2018. 
