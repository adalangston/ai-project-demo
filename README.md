# Projects

## Source Separation

This project explores music source separation through two methods: classical HPSS using Librosa and deep learning-based Demucs stem extraction. Audio is split into harmonic and percussive layers or four stems (drums, bass, vocals, other). Separated tracks are visualized as spectrograms and manipulated through remixing to evaluate acoustic characteristics.

### Key Features:
- Encodes linguistic input using custom one-hot dictionaries.
- Trains with PyTorch’s nn.Linear, CrossEntropyLoss, and torch.optim.Adam.
- Achieves full classification accuracy on 24 grammatical combinations in under 10 epochs.

Libraries Used: torch, torch.nn, torch.optim

## Prediction of German Definite Artcle

This project implements a single-layer neural network in PyTorch that predicts the correct form of the German definite article ("der", "die", "das", etc.) based on grammatical features: number (singular/plural), gender (masculine/feminine/neuter), and grammatical case (nominative, accusative, dative, genitive). The model learns to classify all 24 valid combinations with 100% accuracy in just a few training epochs.

### Key Features:
- Used librosa.decompose.hpss() for traditional source separation.
- Applied Facebook AI’s Demucs to extract high-quality stems using pretrained models.
- Remixing experiments include dynamic amplitude scaling of stems and spectrogram comparisons.
- Integrated audio playback with IPython for comparative analysis.
  
Libraries Used: librosa, matplotlib, numpy, IPython.display, torch (Demucs dependency), pathlib
