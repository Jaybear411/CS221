#!/usr/bin/env python3
import argparse
import pyttsx3
import json
import os

def speak_text(text, voice=None, rate=150):
    """Speak the given text using the pyttsx3 engine."""
    engine = pyttsx3.init()
    
    # Set properties
    engine.setProperty('rate', rate)  # Speed of speech
    
    # Set voice if specified
    if voice is not None:
        voices = engine.getProperty('voices')
        for v in voices:
            if voice.lower() in v.name.lower():
                engine.setProperty('voice', v.id)
                break
    
    # Speak the text
    engine.say(text)
    engine.runAndWait()

def format_confidence(confidence):
    """Format confidence as a percentage string."""
    return f"{int(confidence * 100)} percent"

def format_message(label, confidence):
    """Format the message to be spoken."""
    return f"{label}, {format_confidence(confidence)} confident."

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech module for sketch classification")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to JSON file with prediction data or a JSON string")
    parser.add_argument("--voice", type=str, default=None, 
                       help="Voice to use (female, male, etc.)")
    parser.add_argument("--rate", type=int, default=150, 
                       help="Speech rate (words per minute)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, 
                       help="Confidence threshold for speaking")
    
    args = parser.parse_args()
    
    # Parse input (either a file path or a JSON string)
    if os.path.isfile(args.input):
        with open(args.input, 'r') as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(args.input)
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON input: {args.input}")
            return
    
    # Extract label and confidence
    label = data.get('label')
    confidence = data.get('confidence')
    
    if label is None or confidence is None:
        print("Error: Input JSON must contain 'label' and 'confidence' fields")
        return
    
    # Check if confidence is above threshold
    if confidence < args.confidence_threshold:
        print(f"Confidence {confidence} is below threshold {args.confidence_threshold}, not speaking")
        return
    
    # Format message and speak
    message = format_message(label, confidence)
    print(f"Speaking: {message}")
    speak_text(message, args.voice, args.rate)

if __name__ == "__main__":
    main() 