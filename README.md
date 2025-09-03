# YOLO-Keras

This repository contains Keras implementations of YOLO models with weight conversion utilities from PyTorch.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IMvision12/yolo-keras.git
cd yolo-keras
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv env
.\env\Scripts\activate

# On Linux/Mac
python -m venv env
source env/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
pip install sentencepiece
```

## Converting YOLOv8 Weights

There are two ways to run the conversion script:

### Method 1: Using Python Module Syntax (Recommended)
```bash
# Make sure you're in the yolo-keras directory
python -m yolo.yolov8.convert
```

### Method 2: Using PYTHONPATH (Alternative)
```bash
# On Windows
set PYTHONPATH=C:\path\to\yolo-keras
python yolo/yolov8/convert.py

# On Linux/Mac
export PYTHONPATH=/path/to/yolo-keras
python yolo/yolov8/convert.py
```

Replace `/path/to/yolo-keras` with the actual path to your yolo-keras directory.

For example, if you cloned the repository to `C:\Users\username\yolo-keras`, you would use:
```bash
# On Windows
set PYTHONPATH=C:\Users\username\yolo-keras
python yolo/yolov8/convert.py
```

This will:
1. Load the YOLOv8n model
2. Convert the weights to Keras format
3. Save the converted weights as `yolov8n.weights.h5`

## Using the Converted Model

After conversion, you can use the Keras model with the converted weights for inference or further training.

## License

See [LICENSE](LICENSE) for details.
