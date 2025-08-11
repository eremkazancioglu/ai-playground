#!/usr/bin/env python3

import os
import tarfile
import boto3
import shutil
from pathlib import Path
import tempfile

def create_deployment_package(job_name, s3_bucket):
    """Create deployment package with all fold models"""
    
    # Download all necessary files
    s3_client = boto3.client('s3')
    
    # Create temp directory
    package_dir = Path('deployment_package')
    package_dir.mkdir(exist_ok=True)
    
    # Download all fold models from checkpoints
    for fold in range(1, 4):  # Assuming 3 folds
        fold_model = f"best_model_fold_{fold}.pth"
        s3_key = f"checkpoints/{fold_model}"
        local_path = package_dir / fold_model
        
        try:
            s3_client.download_file(s3_bucket, s3_key, str(local_path))
            print(f"✅ Downloaded {fold_model}")
        except Exception as e:
            print(f"❌ Failed to download {fold_model}: {e}")
    
    # Download and extract species mapping from output.tar.gz
    try:
        # Download output.tar.gz to temp location
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as temp_file:
            s3_client.download_file(
                s3_bucket, 
                f"sagemaker-outputs/{job_name}/output/model.tar.gz",
                temp_file.name
            )
            
            # Extract species_to_label.json from the archive
            with tarfile.open(temp_file.name, 'r:gz') as tar:
                try:
                    # Extract just the species_to_label.json file
                    species_member = tar.getmember('species_to_label.json')
                    species_file = tar.extractfile(species_member)
                    
                    # Write to our package directory
                    with open(package_dir / 'species_to_label.json', 'wb') as f:
                        f.write(species_file.read())
                    
                    print("✅ Extracted species_to_label.json from output.tar.gz")
                except KeyError:
                    print("❌ species_to_label.json not found in output.tar.gz")
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
    except Exception as e:
        print(f"❌ Failed to download/extract from output.tar.gz: {e}")
    
    # Create model.tar.gz
    with tarfile.open('model.tar.gz', 'w:gz') as tar:
        tar.add(package_dir, arcname='.')
    
    # Upload to S3
    s3_key = f"deployment-models/{job_name}/model.tar.gz"
    s3_client.upload_file('model.tar.gz', s3_bucket, s3_key)
    
    s3_path = f"s3://{s3_bucket}/{s3_key}"
    print(f"✅ Uploaded deployment package to: {s3_path}")
    
    # Cleanup
    shutil.rmtree(package_dir)
    os.remove('model.tar.gz')
    
    return s3_path

if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser(description='Create deployment package')
    parser.add_argument('--job-name', type=str, help='Job name')
    args = parser.parse_args()

    # Usage - replace with your actual values
    # job_name = 'species-classifier-2025-08-05-20-47-57'  # e.g., 'species-classifier-2025-08-05-20-15-30'
    bucket = 'erem-sagemaker-run'
    
    deployment_s3_path = create_deployment_package(args.job_name, bucket)
    print(f"\n🎉 Ready to deploy with:")
    print(f"python deploy_model.py --action deploy --model-s3-path {deployment_s3_path} --role your-role-arn")