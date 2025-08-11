#!/usr/bin/env python3

import os
import sys
import json
import argparse
import subprocess
import boto3
from pathlib import Path
import pandas as pd
import torch
import tempfile
import shutil

def test_data_accessibility():
    """Test if training data is accessible from S3"""
    print("🔍 Testing data accessibility...")
    
    # Test S3 access
    s3_client = boto3.client('s3')
    
    # Replace with your actual bucket and paths
    bucket = 'erem-sagemaker-training-data'  # Update this
    csv_key = 'inaturalist-images/new_balto_species_photos_augmented_subsample_other.csv'  # Update this
    
    try:
        # Test CSV file access
        response = s3_client.head_object(Bucket=bucket, Key=csv_key)
        print(f"✅ CSV file accessible: {response['ContentLength']} bytes")
        
        # Download and validate CSV structure
        csv_obj = s3_client.get_object(Bucket=bucket, Key=csv_key)
        df = pd.read_csv(csv_obj['Body'])
        
        print(f"✅ CSV loaded successfully: {len(df)} rows")
        print(f"   Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['species_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Test a few image files
        print("🔍 Testing sample image accessibility...")
        sample_images = df.head(3)  # Test first 3 images
        
        # Check what columns we have for image names
        possible_image_cols = ['image_name', 'filename', 'image_path', 'file_path']
        image_col = None
        
        for col in possible_image_cols:
            if col in df.columns:
                image_col = col
                break
        
        if not image_col:
            print(f"❌ No image column found. Expected one of: {possible_image_cols}")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        print(f"   Using '{image_col}' column for image names")
        
        for idx, row in sample_images.iterrows():
            # Get the image name from the appropriate column
            image_name = row[image_col]
            
            # Since your CSV has bare filenames, construct the full S3 key
            img_key = f"inaturalist-images/{image_name}"
            
            try:
                s3_client.head_object(Bucket=bucket, Key=img_key)
                print(f"✅ Image {idx+1} accessible: {img_key}")
            except Exception as e:
                print(f"❌ Image {idx+1} not accessible: {img_key} - {e}")
                # Try without the training-data prefix in case images are in root
                try:
                    img_key_alt = image_name
                    s3_client.head_object(Bucket=bucket, Key=img_key_alt)
                    print(f"✅ Image {idx+1} found in bucket root: {img_key_alt}")
                except Exception as e2:
                    print(f"   Also tried bucket root, still not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Data accessibility test failed: {e}")
        return False

def test_permissions():
    """Test AWS permissions"""
    print("🔍 Testing AWS permissions...")
    
    try:
        # Test SageMaker permissions
        sagemaker_client = boto3.client('sagemaker')
        sagemaker_client.list_training_jobs(MaxResults=1)
        print("✅ SageMaker permissions OK")
        
        # Test S3 permissions
        s3_client = boto3.client('s3')
        buckets = s3_client.list_buckets()
        print("✅ S3 permissions OK")
        
        # Test specific bucket access
        bucket = 'erem-sagemaker-run'  # Update this
        try:
            s3_client.head_bucket(Bucket=bucket)
            print(f"✅ Bucket '{bucket}' accessible")
        except Exception as e:
            print(f"❌ Bucket '{bucket}' not accessible: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Permission test failed: {e}")
        return False

def test_local_training_script():
    """Test training script locally with minimal data"""
    print("🔍 Testing training script locally...")
    
    try:
        # Create minimal test dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy CSV
            test_csv = os.path.join(temp_dir, 'test_data.csv')
            test_data = pd.DataFrame({
                'species_name': ['Lycorma delicatula'] * 10 + ['other'] * 10,
                'filename': [f'test_image_{i}.jpg' for i in range(20)]
            })
            test_data.to_csv(test_csv, index=False)
            
            # Create dummy images directory and files
            img_dir = os.path.join(temp_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            
            # Create small dummy images
            from PIL import Image
            import numpy as np
            
            for i in range(20):
                # Create 64x64 RGB image
                dummy_img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                dummy_img.save(os.path.join(img_dir, f'test_image_{i}.jpg'))
            
            # Test import and basic functionality
            sys.path.insert(0, '.')
            from train import prepare_data, get_transforms, create_model_efficientnet
            
            # Test data preparation
            df, species_to_label = prepare_data(
                test_csv, 
                ['Lycorma delicatula'], 
                downsample_other=True, 
                downsample_frac=1.0
            )
            print(f"✅ Data preparation successful: {len(df)} samples, {len(species_to_label)} classes")
            
            # Test transforms
            train_transform, val_transform = get_transforms(64)
            print("✅ Transforms created successfully")
            
            # Test model creation
            model = create_model_efficientnet(len(species_to_label))
            print("✅ Model created successfully")
            
            # Test forward pass
            dummy_input = torch.randn(2, 3, 64, 64)
            with torch.no_grad():
                output = model(dummy_input)
            print(f"✅ Forward pass successful: output shape {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Local training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_minimal_sagemaker_job():
    """Create minimal SageMaker job for testing"""
    print("🔍 Creating minimal SageMaker test job...")
    
    import sagemaker
    from sagemaker.pytorch import PyTorch
    from sagemaker.inputs import TrainingInput
    from datetime import datetime
    
    # Minimal configuration
    config = {
        'role': 'arn:aws:iam::211052399214:role/service-role/AmazonSageMaker-ExecutionRole-20250801T124299',  # Update this
        'bucket': 'erem-sagemaker-run',  # Update this
        's3_training_data': 's3://erem-sagemaker-training-data/inaturalist-images/',  # Update this
        's3_output_path': 's3://erem-sagemaker-run/sagemaker-outputs/',
        'instance_type': 'ml.m5.large',  # CPU instance for testing
        'max_run': 1800,  # 30 minutes max
        'use_spot_instances': False,  # No spot for quick test
    }
    
    # Minimal hyperparameters for fast testing
    hyperparameters = {
        'target-species': '"Lycorma delicatula"',
        'csv-file': 'new_balto_species_photos_augmented_subsample_other.csv',
        'img-size': 64,  # Small image size
        'batch-size': 4,  # Small batch size
        'learning-rate': 0.001,
        'num-epochs': 2,  # Very few epochs
        'k-folds': 2,  # Fewer folds
        'epoch-patience': 1,
        'downsample-other': True,
        'downsample-frac': 0.1,  # Use only 10% of data
    }
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        job_name = f'species-classifier-dryrun-{timestamp}'
        
        estimator = PyTorch(
            entry_point='train.py',
            source_dir='.',
            role=config['role'],
            instance_type=config['instance_type'],
            instance_count=1,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters=hyperparameters,
            output_path=config['s3_output_path'],
            use_spot_instances=config['use_spot_instances'],
            max_run=config['max_run'],
            job_name=job_name,
            volume_size=10,  # Small volume
        )
        
        training_input = TrainingInput(
            s3_data=config['s3_training_data'],
            content_type='text/csv'
        )
        
        print(f"✅ Test job '{job_name}' created successfully")
        print("⚠️  Job not started - use --execute flag to actually run it")
        
        return estimator, job_name, training_input
        
    except Exception as e:
        print(f"❌ Failed to create test job: {e}")
        return None, None, None

def syntax_check():
    """Check Python syntax of all scripts"""
    print("🔍 Checking Python syntax...")
    
    scripts = ['train.py', 'inference.py', 'launch_training.py', 'deploy_model.py']
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"❌ Script not found: {script}")
            continue
            
        try:
            with open(script, 'r') as f:
                code = f.read()
            compile(code, script, 'exec')
            print(f"✅ {script} syntax OK")
        except SyntaxError as e:
            print(f"❌ {script} syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ {script} error: {e}")
            return False
    
    return True

def dependency_check():
    """Check if all required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'PIL', 'pandas', 'numpy', 
        'sklearn', 'matplotlib', 'seaborn', 'boto3', 'sagemaker'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def configuration_check():
    """Check configuration values"""
    print("🔍 Checking configuration...")
    
    # Check launch_training.py configuration
    try:
        sys.path.insert(0, '.')
        from launch_training import CONFIG
        
        required_configs = ['role', 'bucket', 's3_training_data']
        issues = []
        
        for key in required_configs:
            if key not in CONFIG:
                issues.append(f"Missing {key}")
            elif 'YOUR_ACCOUNT_ID' in str(CONFIG[key]) or 'your-s3-bucket' in str(CONFIG[key]):
                issues.append(f"{key} contains placeholder values")
        
        if issues:
            print("❌ Configuration issues:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ Configuration looks good")
            return True
            
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Dry run testing for SageMaker species classifier')
    parser.add_argument('--test', choices=[
        'all', 'syntax', 'deps', 'config', 'permissions', 'data', 'local', 'sagemaker'
    ], default='all', help='Which tests to run')
    parser.add_argument('--execute-sagemaker', action='store_true', 
                       help='Actually execute the minimal SageMaker job (costs money!)')
    
    args = parser.parse_args()
    
    print("🚀 Starting dry run tests...\n")
    
    tests_passed = 0
    total_tests = 0
    
    if args.test in ['all', 'syntax']:
        total_tests += 1
        if syntax_check():
            tests_passed += 1
        print()
    
    if args.test in ['all', 'deps']:
        total_tests += 1
        if dependency_check():
            tests_passed += 1
        print()
    
    if args.test in ['all', 'config']:
        total_tests += 1
        if configuration_check():
            tests_passed += 1
        print()
    
    if args.test in ['all', 'permissions']:
        total_tests += 1
        if test_permissions():
            tests_passed += 1
        print()
    
    if args.test in ['all', 'data']:
        total_tests += 1
        if test_data_accessibility():
            tests_passed += 1
        print()
    
    if args.test in ['all', 'local']:
        total_tests += 1
        if test_local_training_script():
            tests_passed += 1
        print()
    
    if args.test in ['all', 'sagemaker']:
        total_tests += 1
        estimator, job_name, training_input = create_minimal_sagemaker_job()
        if estimator is not None:
            tests_passed += 1
            
            if args.execute_sagemaker:
                print("🚀 Executing minimal SageMaker job...")
                try:
                    estimator.fit({'training': training_input}, wait=False)
                    print(f"✅ Job '{job_name}' started successfully!")
                    print("💰 Remember: This will incur AWS charges!")
                except Exception as e:
                    print(f"❌ Failed to start job: {e}")
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready for full training.")
    else:
        print("⚠️  Some tests failed. Fix issues before running full training.")
        sys.exit(1)

if __name__ == "__main__":
    main()