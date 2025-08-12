import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.pipeline import Measurements

m=Measurements()

frontal = r"Test\test_images\front.png"
lateral = r"Test\test_images\lateral.png"

predictions = m.predict(frontal, lateral)

print("Predicted measurements:", predictions)