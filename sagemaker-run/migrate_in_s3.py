import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class S3ImageTransfer:
    def __init__(self, destination_bucket: str, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None, region_name: str = 'us-east-1'):
        """
        Initialize S3 client for transferring images
        
        Args:
            destination_bucket: Your personal S3 bucket name
            aws_access_key_id: AWS access key (optional if using IAM roles)
            aws_secret_access_key: AWS secret key (optional if using IAM roles)
            region_name: AWS region
        """
        self.destination_bucket = destination_bucket
        self.source_bucket = 'inaturalist-open-data'
        
        # Initialize S3 client
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # Use default credentials (IAM roles, ~/.aws/credentials, environment variables)
            self.s3_client = boto3.client('s3', region_name=region_name)
    
    def transfer_single_image(self, photo_id: str, extension: str, 
                            destination_key: Optional[str] = None) -> bool:
        """
        Transfer a single image from iNaturalist to your bucket
        
        Args:
            photo_id: Photo ID from dataframe
            extension: File extension from dataframe
            destination_key: Optional custom destination path (defaults to same structure)
            
        Returns:
            bool: True if successful, False otherwise
        """
        source_key = f"photos/{photo_id}/medium.{extension}"
        
        if destination_key is None:
            destination_key = f"inaturalist-images-acer-like/{photo_id}_medium.{extension}"
        
        try:
            # Copy the object from source to destination
            copy_source = {'Bucket': self.source_bucket, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.destination_bucket,
                Key=destination_key
            )
            
            logger.info(f"Successfully transferred {source_key} to {destination_key}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                logger.warning(f"Source file not found: {source_key}")
            else:
                logger.error(f"Error transferring {source_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error transferring {source_key}: {e}")
            return False
    
    def transfer_images_sequential(self, df: pd.DataFrame, 
                                 destination_prefix: str = "inaturalist-images-acer-like") -> dict:
        """
        Transfer images sequentially (slower but more reliable)
        
        Args:
            df: DataFrame with 'photo_id' and 'extension' columns
            destination_prefix: Prefix for destination keys
            
        Returns:
            dict: Summary of transfer results
        """
        results = {'successful': 0, 'failed': 0, 'failed_ids': []}
        
        for idx, row in df.iterrows():
            photo_id = str(row['photo_id'])
            extension = row['extension']
            
            destination_key = f"{destination_prefix}/{photo_id}_medium.{extension}"
            success = self.transfer_single_image(photo_id, extension, destination_key)
            
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
                results['failed_ids'].append(photo_id)
            
            # Optional: Add a small delay to be respectful to the source
            time.sleep(0.1)
        
        return results

def main():

    # Load your dataframe
    df = pd.read_csv("../acer_like_maryland_photo_ids.csv")
    
    # Configure your S3 transfer
    destination_bucket = 'erem-sagemaker-training-data'  # Replace with your bucket name
    
    # Initialize the transfer class
    # Option 1: Using default AWS credentials
    transfer = S3ImageTransfer(destination_bucket)
    
    # Transfer images
    logger.info(f"Starting transfer of {len(df)} images...")
    
    # Choose your transfer method:
    
    # Sequential transfer (recommended for large datasets to avoid rate limiting)
    results = transfer.transfer_images_sequential(df)
    
    # Print results
    logger.info(f"Transfer complete!")
    logger.info(f"Successful transfers: {results['successful']}")
    logger.info(f"Failed transfers: {results['failed']}")
    
    if results['failed_ids']:
        logger.info(f"Failed photo IDs: {results['failed_ids']}")
        
        # Optionally save failed IDs for retry
        failed_df = df[df['photo_id'].isin(results['failed_ids'])]
        failed_df.to_csv('failed_transfers.csv', index=False)
        logger.info("Failed transfers saved to 'failed_transfers.csv'")

if __name__ == "__main__":
    main()