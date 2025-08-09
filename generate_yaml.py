import os
import logging
from typing import List

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yaml_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def generate_yaml(data_dir: str, output_path: str, classes: List[str]) -> bool:
    """
    Generate YAML file for Food-101 dataset.

    Args:
        data_dir (str): Path to dataset directory (e.g., './food-101')
        output_path (str): Path to save YAML file (e.g., './food-101/food101.yaml')
        classes (List[str]): List of class names (e.g., ['pizza', 'grilled_chicken', ...])

    Returns:
        bool: True if YAML created successfully, False otherwise
    """
    try:
        if not os.path.exists(data_dir):
            raise ValueError(f"Dataset directory {data_dir} does not exist")
        
        # Adjusted for Food-101 structure with class folders under images/
        yaml_content = f"""
train: {os.path.join(data_dir, 'images')}
val: {os.path.join(data_dir, 'images')}
nc: {len(classes)}
names: {classes}
"""
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        logger.info(f"YAML file created at {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating YAML file: {str(e)}")
        return False

if __name__ == "__main__":
    data_dir = "./foof/food-101"  # Corrected to match your project structure
    classes = ['pizza', 'grilled_chicken', 'sushi', 'ice_cream', 'hamburger']
    yaml_path = os.path.join(data_dir, 'food101.yaml')
    
    success = generate_yaml(data_dir, yaml_path, classes)
    if success:
        logger.info("YAML generation completed successfully")
    else:
        logger.error("YAML generation failed")