# Neural Style Transfer

Implementation of Neural Style Transfer using TensorFlow 1.12 and VGG19.

## Tasks

| File | Description |
|------|-------------|
| `0-neural_style.py` | NST class with `scale_image` static method |
| `1-neural_style.py` | Adds `load_model` using VGG19 with average pooling |
| `2-neural_style.py` | Adds `gram_matrix` static method |
| `3-neural_style.py` | Adds `generate_features` for style/content extraction |
| `4-neural_style.py` | Adds `layer_style_cost` for single-layer style cost |
| `5-neural_style.py` | Adds `style_cost` across all style layers |
| `6-neural_style.py` | Adds `content_cost` |
| `7-neural_style.py` | Adds `total_cost` combining content and style costs |
| `8-neural_style.py` | Adds `compute_grads` using GradientTape |
| `9-neural_style.py` | Adds `generate_image` with Adam optimization |
| `10-neural_style.py` | Adds `variational_cost` for smoother output |

## Requirements

- Python 3.5
- NumPy 1.15
- TensorFlow 1.12
