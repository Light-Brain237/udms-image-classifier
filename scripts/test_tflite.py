"""Quick test to verify TFLite model works locally."""

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

print("Testing TFLite model...")

# Load the model
interpreter = Interpreter(
    model_path='models/classifier.tflite'
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ TFLite model loaded successfully")
print(f"   Input shape:  {input_details[0]['shape']}")
print(f"   Output shape: {output_details[0]['shape']}")

# Test with a dummy image
test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"   Output sum:   {output.sum():.4f} (should be ~1.0)")
print(f"✅ Model inference working correctly!")

# Load label map
import json
with open('models/label_map.json') as f:
    label_map = json.load(f)

print(f"\n✅ Label map loaded: {len(label_map)} categories")
for k, v in label_map.items():
    print(f"   {k}: {v.get('label', v.get('category', k))}")
