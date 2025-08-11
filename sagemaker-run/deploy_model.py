#!/usr/bin/env python3

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from datetime import datetime
import json
import base64
from PIL import Image
import io

class SpeciesClassifierPredictor(Predictor):
    """Custom predictor for species classification"""
    
    def __init__(self, endpoint_name, sagemaker_session=None):
        super().__init__(endpoint_name, sagemaker_session)
    
    def predict_image(self, image_path):
        """Predict species from image file using SageMaker runtime directly"""
        import json
        import boto3
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Encode image as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = json.dumps({'image': image_b64})
        
        # Use SageMaker runtime client directly
        runtime = boto3.client('sagemaker-runtime')
        response = runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        return result

def deploy_model(model_s3_path, role, instance_type='ml.m5.large'):
    """Deploy trained model to SageMaker endpoint"""
    
    # Create timestamp for unique endpoint naming
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    endpoint_name = f'species-classifier-{timestamp}'
    
    print(f"Deploying model to endpoint: {endpoint_name}")
    print(f"Model artifacts: {model_s3_path}")
    
    # Create PyTorch model
    model = PyTorchModel(
        model_data=model_s3_path,
        role=role,
        entry_point='inference.py',
        source_dir='.',  # Directory containing inference.py
        framework_version='1.12.0',
        py_version='py38',
        predictor_cls=SpeciesClassifierPredictor
    )
    
    # Deploy model
    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name
        )
        
        print(f"Model deployed successfully to endpoint: {endpoint_name}")
        return predictor, endpoint_name
        
    except Exception as e:
        print(f"Error deploying model: {e}")
        return None, None

def test_endpoint(predictor, test_image_path=None):
    """Test the deployed endpoint"""
    if test_image_path and os.path.exists(test_image_path):
        print(f"Testing endpoint with image: {test_image_path}")
        
        try:
            result = predictor.predict_image(test_image_path)
            print("Prediction result:")
            print(json.dumps(result, indent=2))
            return result
            
        except Exception as e:
            print(f"Error testing endpoint: {e}")
            return None
    else:
        print("No test image provided or file doesn't exist")
        return None

def create_test_client():
    """Create a simple test client for the endpoint"""
    return """
import boto3
import json
import base64
from PIL import Image
import io

class SpeciesClassifierClient:
    def __init__(self, endpoint_name, region='us-east-1'):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    def predict_image(self, image_path):
        \"\"\"Predict species from image file\"\"\"
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Encode image as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = json.dumps({'image': image_b64})
        
        # Make prediction
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        return result
    
    def predict_pil_image(self, pil_image):
        \"\"\"Predict species from PIL Image\"\"\"
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Encode as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        payload = json.dumps({'image': image_b64})
        
        # Make prediction
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        
        result = json.loads(response['Body'].read().decode())
        return result

# Example usage:
# client = SpeciesClassifierClient('your-endpoint-name')
# result = client.predict_image('path/to/your/image.jpg')
# print(result)
"""

def cleanup_endpoint(endpoint_name):
    """Delete the endpoint to avoid charges"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        # Delete endpoint
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} deletion initiated")
        
        # Delete endpoint configuration
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Endpoint config {endpoint_name} deletion initiated")
        
    except Exception as e:
        print(f"Error cleaning up endpoint: {e}")

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Deploy species classifier model')
    parser.add_argument('--action', choices=['deploy', 'test', 'cleanup', 'client'], 
                       default='deploy', help='Action to perform')
    parser.add_argument('--model-s3-path', type=str, required=False,
                       help='S3 path to model artifacts (required for deploy)')
    parser.add_argument('--role', type=str, required=False,
                       help='SageMaker execution role ARN (required for deploy)')
    parser.add_argument('--endpoint-name', type=str, help='Endpoint name for test/cleanup actions')
    parser.add_argument('--test-image', type=str, help='Test image path')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                       help='Instance type for endpoint')
    
    args = parser.parse_args()
    
    if args.action == 'deploy':
        if not args.model_s3_path or not args.role:
            print("--model-s3-path and --role are required for deployment")
            exit(1)
        
        predictor, endpoint_name = deploy_model(
            args.model_s3_path, 
            args.role, 
            args.instance_type
        )
        
        if predictor:
            print(f"\nEndpoint deployed successfully!")
            print(f"Endpoint name: {endpoint_name}")
            print(f"\nTo test the endpoint:")
            print(f"python {__file__} --action test --endpoint-name {endpoint_name} --test-image path/to/image.jpg")
            print(f"\nTo cleanup the endpoint (avoid charges):")
            print(f"python {__file__} --action cleanup --endpoint-name {endpoint_name}")
    
    elif args.action == 'test':
        if not args.endpoint_name:
            print("--endpoint-name is required for testing")
            exit(1)
        
        # Create predictor for existing endpoint
        predictor = SpeciesClassifierPredictor(args.endpoint_name)
        test_endpoint(predictor, args.test_image)
    
    elif args.action == 'cleanup':
        if not args.endpoint_name:
            print("--endpoint-name is required for cleanup")
            exit(1)
        
        cleanup_endpoint(args.endpoint_name)
    
    elif args.action == 'client':
        print("Test client code:")
        print(create_test_client())
        
        # Save to file
        with open('species_classifier_client.py', 'w') as f:
            f.write(create_test_client())
        print("\nClient code saved to 'species_classifier_client.py'")