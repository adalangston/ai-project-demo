# Projects

## Source Separation

This project explores music source separation through two methods: classical HPSS using Librosa and deep learning-based Demucs stem extraction. Audio is split into harmonic and percussive layers or four stems (drums, bass, vocals, other). Separated tracks are visualized as spectrograms and manipulated through remixing to evaluate acoustic characteristics.

### Key Features:
- Used librosa.decompose.hpss() for traditional source separation.
- Applied Demucs to extract high-quality stems using pretrained models.
- Included dynamic amplitude scaling of stems and spectrogram comparisons in remixing experiments.
- Integrated audio playback with IPython for comparative analysis.
  
Libraries Used: librosa, matplotlib, numpy, IPython.display, torch (Demucs dependency), pathlib

## Prediction of German Definite Artcle

This project implements a single-layer neural network in PyTorch that predicts the correct form of the German definite article ("der", "die", "das", etc.) based on grammatical features: number (singular/plural), gender (masculine/feminine/neuter), and grammatical case (nominative, accusative, dative, genitive). The model learns to classify all 24 valid combinations with 100% accuracy in just a few training epochs.

 ### Key Features:
- Encodes linguistic input using custom one-hot vectors.
- Trains with PyTorch’s nn.Linear, CrossEntropyLoss, and torch.optim.Adam.
- Achieves full classification accuracy on 24 grammatical combinations in under 10 epochs.

Libraries Used: torch, torch.nn, torch.optim
