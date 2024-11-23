# Neural network for mass, energy, and direction reconstruction

This is a TensorFlow implementation of the "MultiModal" neural network for reconstructing event properties and corresponding uncertainty estimation. It consists of the following blocks (modals):

## Detectors sequence analyzer

Detectors are ordered in a 1D chain according to the reconstructed time of plane front arrival. Data processing goes as follows:
- Waveforms are encoded using combination convolutional and recurrent layers
- The obtained waveforms encoding are concatenated with detector-wise characteristics, such their location.
- The resulting sequence is passed thorugh LSTM layers yielding "sequence" encodings of events.

## Detectors grid analyzer

A grid of 6x6 detectors is analyzed by convolutional layers, yielding "grid" encodings of events.

## Predictions branch
Sequence and grid encodings are combined with reconstructed parameters obtained via standard TA reconstruction procedure. They are passed through fully conected layers yielding:
- Conventional mass parameter (currently, -1 for protons and +1 for iron).
- Log10 energy of an event in normilized units.
- 3D vector (not normilized) representing incoming direction.

## Uncertainty estimation branch
The joint encoding and predictions are passed through fully conected layers to estimate uncertainty of the predictions. The corresponding loss changes weights of this branch only. 

## Notes

- Training neural network for a single target value yields better metrics.