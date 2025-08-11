#!/usr/bin/env python3

import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import base64
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the model for inference"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load species mapping
    with open(os.path.join(model_dir, 'species_to_label.json'), 'r') as f:
        species_to_label = json.load(f)
    
    num_classes = len(species_to_label)
    
    # Create model architecture
    model = models.efficientnet_b1(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    # Load models for each fold
    fold_models = []
    for fold in range(3):  # Assuming 3 folds
        fold_model = models.efficientnet_b1(pretrained=False)
        fold_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
        model_path = os.path.join(model_dir, f'best_model_fold_{fold + 1}.pth')
        if os.path.exists(model_path):
            fold_model.load_state_dict(torch.load(model_path, map_location=device))
            fold_model.to(device)
            fold_model.eval()
            fold_models.append(fold_model)
        else:
            logger.warning(f"Model for fold {fold + 1} not found at {model_path}")
    
    if not fold_models:
        raise ValueError("No trained models found")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'models': fold_models,
        'species_to_label': species_to_label,
        'transform': transform,
        'device': device
    }

def input_fn(request_body, request_content_type):
    """Parse input data for inference"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        if 'image' in input_data:
            # Base64 encoded image
            image_data = base64.b64decode(input_data['image'])
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return image
        else:
            raise ValueError("No 'image' key found in JSON input")
    
    elif request_content_type in ['image/jpeg', 'image/png', 'image/jpg']:
        # Direct image bytes
        image = Image.open(io.BytesIO(request_body)).convert('RGB')
        return image
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction using ensemble of fold models"""
    models = model_dict['models']
    species_to_label = model_dict['species_to_label']
    transform = model_dict['transform']
    device = model_dict['device']
    
    # Preprocess image
    image_tensor = transform(input_data).unsqueeze(0).to(device)
    
    # Get predictions from all fold models
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for model in models:
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.append(predicted.cpu().item())
            all_probabilities.append(probabilities.cpu().numpy()[0])
    
    # Ensemble prediction (majority voting)
    from collections import Counter
    vote_counts = Counter(all_predictions)
    ensemble_prediction = vote_counts.most_common(1)[0][0]
    
    # Average probabilities across models
    import numpy as np
    avg_probabilities = np.mean(all_probabilities, axis=0)
    
    # Convert label to species name
    label_to_species = {v: k for k, v in species_to_label.items()}
    predicted_species = label_to_species[ensemble_prediction]
    
    # Create confidence scores for all classes
    class_probabilities = {}
    for species, label in species_to_label.items():
        class_probabilities[species] = float(avg_probabilities[label])
    
    return {
        'predicted_species': predicted_species,
        'confidence': float(avg_probabilities[ensemble_prediction]),
        'class_probabilities': class_probabilities,
        'individual_fold_predictions': [label_to_species[pred] for pred in all_predictions]
    }

def output_fn(prediction, response_content_type):
    """Format the prediction output"""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")