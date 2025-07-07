"""
Configuration utilities for loading and validating configuration files.
"""

import json
from typing import Dict, Any
from pathlib import Path
from jsonschema import validate, ValidationError


class ConfigManager:
    """Handles configuration loading and validation."""
    
    def __init__(self, config_path: str = "config.json", schema_path: str = "config_schema.json"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the configuration file
            schema_path: Path to the JSON schema file
        """
        self.config_path = Path(config_path)
        self.schema_path = Path(schema_path)
        self.config = None
        self.schema = None
    
    def load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema for validation."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
            return self.schema
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file: {e}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            return self.config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def validate_config(self) -> None:
        """Validate the configuration against the schema."""
        if self.config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        if self.schema is None:
            raise ValueError("Schema not loaded. Call load_schema() first.")
        
        try:
            validate(instance=self.config, schema=self.schema)
            print("✓ Configuration validation successful")
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
    
    def load_and_validate(self) -> Dict[str, Any]:
        """Load configuration and validate it against schema."""
        self.load_schema()
        self.load_config()
        self.validate_config()
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save_config(self, path: str = None) -> None:
        """Save the current configuration to file."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Configuration saved to {save_path}")
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.get('checkpointing.checkpoint_dir', 'checkpoints'),
            self.get('logging.log_dir', 'logs'),
            'results/figures',
            'results/models',
            'results/predictions'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created directories: {', '.join(directories)}")
    
    def validate_paths(self) -> None:
        """Validate that required paths exist."""
        data_file = self.get('data.processed_file')
        
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"✓ Data file exists: {data_file}")


def load_config(config_path: str = "config/config.json", schema_path: str = "config/config_schema.json") -> Dict[str, Any]:
    """
    Convenient function to load and validate configuration.
    
    Args:
        config_path: Path to configuration file
        schema_path: Path to schema file
    
    Returns:
        Validated configuration dictionary
    """
    config_manager = ConfigManager(config_path, schema_path)
    config = config_manager.load_and_validate()
    config_manager.create_directories()
    config_manager.validate_paths()
    return config


if __name__ == "__main__":
    # Test the configuration manager
    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Project name: {config['project_name']}")
        print(f"Model architecture: {config['model']['architecture']}")
        print(f"Batch size: {config['training']['batch_size']}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")