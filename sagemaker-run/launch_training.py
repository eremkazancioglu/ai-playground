#!/usr/bin/env python3

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from datetime import datetime
import os

# Configuration
CONFIG = {
    # AWS Configuration
    'role': 'arn:aws:iam::211052399214:role/service-role/AmazonSageMaker-ExecutionRole-20250801T124299',  # Replace with your role
    'region': 'us-east-1',  # Replace with your preferred region
    'bucket': 'erem-sagemaker-run',  # Replace with your S3 bucket
    
    # Data paths in S3
    's3_training_data': 's3://erem-sagemaker-training-data/inaturalist-images/',  # Replace with your S3 data path
    's3_output_path': 's3://erem-sagemaker-run/sagemaker-outputs/',
    's3_checkpoint_path': 's3://erem-sagemaker-run/checkpoints/',
    
    # Training configuration
    'instance_type': 'ml.g4dn.xlarge',  # GPU instance for training
    'instance_count': 1,
    'max_run': 86400,  # 24 hours in seconds
    'use_spot_instances': True,
    'max_wait': 129600,  # 36 hours in seconds (includes spot waiting time)
    
    # Model hyperparameters
    'hyperparameters': {
        'target-species': '"Lycorma delicatula,Zenaida macroura,Acer rubrum,Erigeron canadensis"',
        'csv-file': 'new_balto_species_photos_augmented_subsample_other_with_acer_like.csv',
        'img-size': 240,
        'batch-size': 32,  # Reduced for g4dn.xlarge
        'learning-rate': 0.0001,
        'num-epochs': 20,
        'k-folds': 3,
        'epoch-patience': 5,
        'downsample-other': False,
        'downsample-frac': 0.5,
        's3-bucket': 'erem-sagemaker-run',  # Replace with your bucket
        's3-checkpoint-prefix': 'checkpoints'
    },
    
    # Dry run configuration (smaller, faster, cheaper)
    'dry_run_hyperparameters': {
        'target-species': '"Lycorma delicatula"',  # Test with one species
        'csv-file': 'new_balto_species_photos_augmented_subsample_other_with_acer_like.csv',
        'img-size': 32,  # Much smaller images
        'batch-size': 16,  # Small batch
        'learning-rate': 0.001,
        'num-epochs': 1,  # Just 1 epoch
        'k-folds': 1,  # Simple train test split
        'epoch-patience': 1,
        'downsample-other': True,
        'downsample-frac': 0.02,  # Use only 2% of data
        's3-bucket': 'erem-sagemaker-run',
        's3-checkpoint-prefix': 'checkpoints'
    }
}

def launch_training_job(dry_run=False):
    """Launch SageMaker training job with spot instances and checkpointing"""
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    
    # Create timestamp for unique job naming
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_prefix = 'species-classifier-dryrun' if dry_run else 'species-classifier'
    job_name = f'{job_prefix}-{timestamp}'
    
    # Choose configuration based on dry run
    hyperparameters = CONFIG['dry_run_hyperparameters'] if dry_run else CONFIG['hyperparameters']
    instance_type = 'ml.m5.large' if dry_run else CONFIG['instance_type']  # CPU for dry run
    max_run = 1800 if dry_run else CONFIG['max_run']  # 60 min for dry run
    use_spot = False if dry_run else CONFIG['use_spot_instances']  # No spot for dry run
    volume_size = 10 if dry_run else 30  # Smaller volume for dry run
    
    print(f"Launching {'DRY RUN' if dry_run else 'FULL'} training job: {job_name}")
    print(f"Using role: {CONFIG['role']}")
    print(f"Training data: {CONFIG['s3_training_data']}")
    print(f"Output path: {CONFIG['s3_output_path']}")
    print(f"Instance type: {instance_type}")
    print(f"Max runtime: {max_run//60} minutes")
    
    if dry_run:
        print("🧪 DRY RUN MODE:")
        print(f"   - Using {hyperparameters['downsample-frac']*100}% of data")
        print(f"   - Only {hyperparameters['num-epochs']} epochs")
        print(f"   - {hyperparameters['k-folds']} folds")
        print(f"   - Small images ({hyperparameters['img-size']}x{hyperparameters['img-size']})")
        print(f"   - CPU instance (cheaper)")
    
    # Create PyTorch estimator - different params for spot vs on-demand
    estimator_params = {
        'entry_point': 'train.py',
        'source_dir': '.',
        'role': CONFIG['role'],
        'instance_type': instance_type,
        'instance_count': CONFIG['instance_count'],
        'framework_version': '1.12.0',
        'py_version': 'py38',
        'hyperparameters': hyperparameters,
        'output_path': CONFIG['s3_output_path'],
        'max_run': max_run,
        'enable_sagemaker_metrics': True,
        'metric_definitions': [
            {'Name': 'train_loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
            {'Name': 'train_f1', 'Regex': 'Train F1: ([0-9\\.]+)'},
            {'Name': 'val_loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
            {'Name': 'val_f1', 'Regex': 'Val F1: ([0-9\\.]+)'},
            {'Name': 'fold_f1', 'Regex': 'Fold [0-9]+ F1 Score: ([0-9\\.]+)'},
            {'Name': 'mean_f1', 'Regex': 'Mean F1 Score: ([0-9\\.]+)'}
        ],
        'volume_size': volume_size,
    }

    # Add spot-specific parameters only if using spot instances
    if use_spot:
        estimator_params.update({
            'use_spot_instances': True,
            'max_wait': CONFIG['max_wait'],
            'checkpoint_s3_uri': CONFIG['s3_checkpoint_path']
        })
    else:
        # For on-demand instances (dry run)
        estimator_params.update({
            'use_spot_instances': False,
            'keep_alive_period_in_seconds': 0
        })
        # Only add checkpoint URI for full training, not dry run
        if not dry_run:
            estimator_params['checkpoint_s3_uri'] = CONFIG['s3_checkpoint_path']

    estimator = PyTorch(**estimator_params)
    
    # Define input data
    training_input = TrainingInput(
        s3_data=CONFIG['s3_training_data'],
        content_type='text/csv'
    )
    
    # Start training
    try:
        print("Starting training job...")
        estimator.fit({'training': training_input}, job_name=job_name, wait=False)
        
        print(f"\nTraining job '{job_name}' started successfully!")
        print(f"You can monitor the job in the AWS Console or use:")
        print(f"aws sagemaker describe-training-job --training-job-name {job_name}")
        print(f"\nTo view logs in real-time:")
        print(f"aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name-prefix {job_name}")
        
        return estimator, job_name
        
    except Exception as e:
        print(f"Error starting training job: {e}")
        return None, None

def monitor_training_job(job_name):
    """Monitor training job progress"""
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        while True:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            print(f"Job Status: {status}")
            
            if status in ['Completed', 'Failed', 'Stopped']:
                break
                
            if 'SecondaryStatus' in response:
                print(f"Secondary Status: {response['SecondaryStatus']}")
            
            # Wait before next check
            import time
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error monitoring job: {e}")

def download_results(job_name, local_dir='results'):
    """Download training results and plots"""
    import os
    
    s3_client = boto3.client('s3')
    bucket = CONFIG['bucket']
    
    # Create local directory
    local_dir = os.path.join(local_dir, job_name)
    os.makedirs(local_dir, exist_ok=True)
    
    # Download files from S3 output
    output_prefix = f"sagemaker-outputs/{job_name}/output/"
    
    try:
        # List objects in output directory
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=output_prefix
        )
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                local_path = os.path.join(local_dir, filename)
                
                print(f"Downloading {key} to {local_path}")
                s3_client.download_file(bucket, key, local_path)
        
        print(f"Results downloaded to {local_dir}")
        
    except Exception as e:
        print(f"Error downloading results: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch SageMaker training job')
    parser.add_argument('--action', choices=['launch', 'dry-run', 'monitor', 'download'], 
                       default='launch', help='Action to perform')
    parser.add_argument('--job-name', type=str, help='Job name for monitor/download actions')
    parser.add_argument('--download-dir', type=str, default='results', 
                       help='Local directory for downloading results')
    
    args = parser.parse_args()
    
    if args.action == 'launch':
        estimator, job_name = launch_training_job(dry_run=False)
        if job_name:
            print(f"\n📊 To monitor this job later, run:")
            print(f"python {__file__} --action monitor --job-name {job_name}")
            print(f"\n📥 To download results later, run:")
            print(f"python {__file__} --action download --job-name {job_name}")
    
    elif args.action == 'dry-run':
        print("🧪 Starting DRY RUN (small dataset, 2 epochs, CPU instance)")
        print("💰 Cost: ~$0.15 for 30 minutes on ml.m5.large")
        
        confirm = input("Continue with dry run? (y/N): ")
        if confirm.lower() == 'y':
            estimator, job_name = launch_training_job(dry_run=True)
            if job_name:
                print(f"\n🧪 DRY RUN job '{job_name}' started!")
                print("⏱️  Should complete in ~15-30 minutes")
        else:
            print("Dry run cancelled")
    
    elif args.action == 'monitor':
        if not args.job_name:
            print("Please provide --job-name for monitoring")
        else:
            monitor_training_job(args.job_name)
    
    elif args.action == 'download':
        if not args.job_name:
            print("Please provide --job-name for downloading results")
        else:
            download_results(args.job_name, args.download_dir)