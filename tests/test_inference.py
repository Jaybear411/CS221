#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import json

# Add parent directory to path for importing from utility modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference.predict import preprocess_image

class TestInference(unittest.TestCase):
    """Basic tests for the inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image (a circle)
        self.image = Image.new('RGB', (64, 64), color='white')
        draw = ImageDraw.Draw(self.image)
        draw.ellipse((10, 10, 54, 54), fill=None, outline='black', width=2)
    
    def test_preprocess_image(self):
        """Test the image preprocessing function."""
        # Preprocess the test image
        processed = preprocess_image(self.image)
        
        # Check that the output has the correct shape and type
        self.assertEqual(processed.shape, (64, 64))
        self.assertEqual(processed.dtype, np.float64)
        
        # Check that the values are normalized to [0, 1]
        self.assertTrue(np.all(processed >= 0))
        self.assertTrue(np.all(processed <= 1))
        
        # Check that the image is not blank (should have some black pixels)
        self.assertLess(np.min(processed), 1.0)
    
    def test_image_saving_loading(self):
        """Test saving and loading an image."""
        # Save the test image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_path = temp.name
        
        self.image.save(temp_path)
        
        # Make sure the file exists
        self.assertTrue(os.path.exists(temp_path))
        
        # Load the image and preprocess it
        loaded_image = Image.open(temp_path)
        processed = preprocess_image(loaded_image)
        
        # Check that the image has expected properties
        self.assertEqual(processed.shape, (64, 64))
        
        # Clean up
        os.remove(temp_path)

if __name__ == '__main__':
    unittest.main() 