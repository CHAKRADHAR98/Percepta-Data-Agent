"""
Configuration for Bimanual Robot Learning Data Agent
"""
import os
from pathlib import Path


class Config:
    """System-wide configuration"""
    
    # ============================================================================
    # AGENT CONFIGURATION
    # ============================================================================
    AGENT_NAME = "Bimanual Data Agent"
    AGENT_SEED = "bimanual_robot_learning_agent_seed_phrase_2025"
    AGENT_PORT = 8005
    
    # ============================================================================
    # DATASET STRUCTURE
    # ============================================================================
    BASE_DIR = Path(__file__).parent
    DATASET_DIR = BASE_DIR / "dataset"
    OUTPUT_DIR = BASE_DIR / "output"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Create directories
    VIDEOS_RAW_DIR = DATASET_DIR / "videos" / "raw"
    VIDEOS_ANNOTATED_DIR = DATASET_DIR / "videos" / "annotated"
    VIDEOS_FROM_IPFS_DIR = DATASET_DIR / "videos" / "from_ipfs"
    ANNOTATIONS_DIR = DATASET_DIR / "annotations" / "hand_trajectories"
    QUALITY_DIR = DATASET_DIR / "annotations" / "quality_reports"
    METADATA_DIR = DATASET_DIR / "metadata"
    
    @classmethod
    def create_directories(cls):
        """Create all required directories"""
        dirs = [
            cls.DATASET_DIR,
            cls.OUTPUT_DIR,
            cls.TEMP_DIR,
            cls.VIDEOS_RAW_DIR,
            cls.VIDEOS_ANNOTATED_DIR,
            cls.VIDEOS_FROM_IPFS_DIR,
            cls.ANNOTATIONS_DIR,
            cls.QUALITY_DIR,
            cls.METADATA_DIR,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # HAND TRACKING PARAMETERS
    # ============================================================================
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MIN_HAND_CONFIDENCE = 0.4
    HAND_IDENTITY_DISTANCE_THRESHOLD = 0.3
    
    # ============================================================================
    # GRIPPER THRESHOLDS
    # ============================================================================
    GRIPPER_OPEN_THRESHOLD = 0.15
    GRIPPER_CLOSED_THRESHOLD = 0.05
    
    # ============================================================================
    # ROBOT WORKSPACE (meters)
    # ============================================================================
    WORKSPACE_WIDTH = 1.2
    WORKSPACE_HEIGHT = 0.8
    WORKSPACE_DEPTH = 0.6
    WORKSPACE_OFFSET_X = 0.0
    WORKSPACE_OFFSET_Y = 0.4
    WORKSPACE_OFFSET_Z = 0.0
    
    # ============================================================================
    # CAMERA PARAMETERS
    # ============================================================================
    CAMERA_HEIGHT_FROM_GROUND = 1.5  # meters
    CAMERA_TILT_ANGLE = 30  # degrees downward
    
    # ============================================================================
    # PROCESSING PARAMETERS
    # ============================================================================
    SMOOTHING_WINDOW = 5
    INTERPOLATION_MAX_GAP = 5
    
    # ============================================================================
    # QUALITY THRESHOLDS
    # ============================================================================
    MAX_VELOCITY_THRESHOLD = 5.0  # m/s
    MAX_ACCELERATION_THRESHOLD = 50.0  # m/s²
    TELEPORTATION_THRESHOLD = 0.5  # meters
    MIN_QUALITY_SCORE = 40.0
    
    # ============================================================================
    # TASK TAXONOMY
    # ============================================================================
    TASK_TAXONOMY = {
        'Household Chores': [
            'Opening a bottle', 'Closing a bottle', 'Pouring liquid',
            'Stirring', 'Cutting vegetables', 'Opening a jar',
            'Using microwave', 'Loading dishwasher', 'Wiping countertop',
            'Wiping surface', 'Folding clothes', 'Hanging clothes',
            'Making bed', 'Picking up object', 'Placing object',
            'Stacking objects', 'Opening drawer', 'Closing drawer',
            'Opening door', 'Closing door'
        ],
        'Warehouse Operations': [
            'Picking item from shelf', 'Placing item in bin',
            'Scanning barcode', 'Sorting packages', 'Stacking boxes',
            'Opening box', 'Placing item in box', 'Sealing box',
            'Applying tape', 'Labeling package'
        ],
        'Assembly Operations': [
            'Screwing bolt', 'Unscrewing bolt', 'Hammering nail',
            'Using screwdriver', 'Tightening clamp', 'Inserting part',
            'Removing part', 'Aligning components', 'Using wrench',
            'Using pliers', 'Using drill'
        ],
        'Laboratory Tasks': [
            'Pipetting liquid', 'Opening vial', 'Closing vial',
            'Mixing solution', 'Transferring sample', 'Loading centrifuge'
        ],
        'Other': [
            'Custom task'
        ]
    }
    
    # ============================================================================
    # METTA KNOWLEDGE GRAPH INITIALIZATION
    # ============================================================================
    INITIAL_KNOWLEDGE = {
        # Task relationships
        'similar_tasks': [
            ('opening_bottle', 'opening_jar'),
            ('opening_bottle', 'closing_bottle'),
            ('pouring_liquid', 'transferring_sample'),
            ('cutting_vegetables', 'using_knife'),
            ('screwing_bolt', 'unscrewing_bolt'),
        ],
        # Task requirements
        'task_requirements': [
            ('opening_bottle', 'bimanual', True),
            ('opening_bottle', 'gripper_action', True),
            ('pouring_liquid', 'bimanual', True),
            ('cutting_vegetables', 'bimanual', True),
            ('picking_item_from_shelf', 'bimanual', False),
        ]
    }


# Initialize directories on import
Config.create_directories()