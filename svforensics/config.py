"""
Configuration management for SVforensics.

This module provides functions to load, save, and manage configuration settings.
The settings include default paths for all modules and other configurable parameters.
Configuration is stored in human-readable JSON files that can be edited directly.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths for configuration files
CONFIG_DIR = os.path.join("config")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "svforensics.json")
DEFAULT_PLOT_CONFIG_PATH = os.path.join(CONFIG_DIR, "plot_config.json")

# Global variable to cache configuration
_config_cache = None
_plot_config_cache = None

def get_config_path() -> str:
    """
    Get the path to the configuration file, respecting environment variable if set.
    
    Returns:
        str: Path to the configuration file
    """
    env_path = os.environ.get("SVFORENSICS_CONFIG_PATH")
    if env_path:
        return env_path
    return DEFAULT_CONFIG_PATH

def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    Uses a cache to avoid reading the file multiple times.
    
    Args:
        config_path: Path to the configuration file (optional)
        force_reload: Force reloading from disk even if cached
        
    Returns:
        Dict[str, Any]: The configuration dictionary
    """
    global _config_cache
    
    # Use cached config if available and not forcing reload
    if _config_cache is not None and not force_reload and config_path is None:
        return _config_cache
    
    if config_path is None:
        config_path = get_config_path()
    
    logger.info(f"Loading configuration from: {config_path}")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Successfully loaded configuration from {config_path}")
                # Update cache only if using default path
                if config_path == get_config_path():
                    _config_cache = config
                return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise
    else:
        logger.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

def load_plot_config(config_path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load plot configuration from JSON file.
    Uses a cache to avoid reading the file multiple times.
    
    Args:
        config_path: Path to the plot configuration file (optional)
        force_reload: Force reloading from disk even if cached
        
    Returns:
        Dict[str, Any]: The plot configuration dictionary
    """
    global _plot_config_cache
    
    # Use default path if not specified
    if config_path is None:
        config_path = DEFAULT_PLOT_CONFIG_PATH
    
    # Use cached config if available and not forcing reload
    if _plot_config_cache is not None and not force_reload and config_path == DEFAULT_PLOT_CONFIG_PATH:
        return _plot_config_cache
    
    logger.info(f"Loading plot configuration from: {config_path}")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Successfully loaded plot configuration from {config_path}")
                # Update cache only if using default path
                if config_path == DEFAULT_PLOT_CONFIG_PATH:
                    _plot_config_cache = config
                return config
        except Exception as e:
            logger.error(f"Error loading plot configuration from {config_path}: {str(e)}")
            # Return default configuration on error
            return get_default_plot_config()
    else:
        logger.error(f"Plot configuration file {config_path} not found. Using default configuration.")
        return get_default_plot_config()

def get_default_plot_config() -> Dict[str, Any]:
    """
    Get default plot configuration.
    
    Returns:
        Dict[str, Any]: Default plot configuration dictionary
    """
    return {
        "figure_size": [14, 8],
        "bins": 60,
        "colors": {
            "same_speaker": "skyblue",
            "different_speaker": "coral",
            "case_score": "red"
        },
        "use_kde": False,
        "show_mean_lines": False,
        "show_std_lines": False,
        "alpha": 0.7,
        "linewidth": 2,
        "font_sizes": {
            "title": 20,
            "labels": 16,
            "legend": 12
        },
        "dpi": 300,
        "localization": {
            "en": {
                "title": "Speaker Comparison Analysis",
                "x_axis": "Score",
                "y_axis": "Frequency",
                "legend": {
                    "same_speaker": "Same Speaker",
                    "different_speaker": "Different Speaker",
                    "case_score": "Case Score"
                },
                "mean_line": "Mean",
                "std_line": "Standard Deviation"
            },
            "pt_BR": {
                "title": "Análise de Comparação de Locutor",
                "x_axis": "Pontuação",
                "y_axis": "Frequência",
                "legend": {
                    "same_speaker": "Mesmo Locutor",
                    "different_speaker": "Locutor Diferente",
                    "case_score": "Pontuação do Caso"
                },
                "mean_line": "Média",
                "std_line": "Desvio Padrão"
            }
        }
    }

def get_localized_text(key: str, language: str = "en", config_path: Optional[str] = None) -> str:
    """
    Get localized text from plot configuration.
    
    Args:
        key: Dot-separated path to the text in the localization section
        language: Language code (default: "en")
        config_path: Path to the plot configuration file (optional)
        
    Returns:
        str: Localized text or the key itself if not found
    """
    # Load plot configuration
    plot_config = load_plot_config(config_path)
    
    # Check if localization section exists
    if "localization" not in plot_config:
        logger.warning("No localization section in plot configuration")
        return key
    
    # Check if requested language exists, fall back to English if not
    if language not in plot_config["localization"]:
        logger.warning(f"Language '{language}' not found in localization. Falling back to English.")
        language = "en"
        
        # If English is also not available, return the key
        if language not in plot_config["localization"]:
            return key
    
    # Get the requested text using dot notation
    text = plot_config["localization"][language]
    for part in key.split("."):
        if part in text:
            text = text[part]
        else:
            # Key not found in the localization
            logger.warning(f"Localization key '{key}' not found for language '{language}'")
            return key
    
    # Return the text if it's a string, otherwise return the key
    if isinstance(text, str):
        return text
    return key

def save_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: The configuration dictionary
        config_path: Path to save the configuration file (optional)
    """
    global _config_cache
    
    if config_path is None:
        config_path = get_config_path()
    
    logger.info(f"Saving configuration to: {config_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4, sort_keys=True)
        logger.info(f"Successfully saved configuration to {config_path}")
        
        # Update cache if saving to default path
        if config_path == get_config_path():
            _config_cache = config
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        raise

def save_plot_config(config: Dict[str, Any], config_path: Optional[str] = None) -> None:
    """
    Save plot configuration to JSON file.
    
    Args:
        config: The plot configuration dictionary
        config_path: Path to save the plot configuration file (optional)
    """
    global _plot_config_cache
    
    if config_path is None:
        config_path = DEFAULT_PLOT_CONFIG_PATH
    
    logger.info(f"Saving plot configuration to: {config_path}")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4, sort_keys=True)
        logger.info(f"Successfully saved plot configuration to {config_path}")
        
        # Update cache if saving to default path
        if config_path == DEFAULT_PLOT_CONFIG_PATH:
            _plot_config_cache = config
    except Exception as e:
        logger.error(f"Error saving plot configuration to {config_path}: {str(e)}")
        raise

def get_path(key: str, override_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get a path from the configuration or use an override if provided.
    
    Args:
        key: The key for the path in the paths section
        override_path: An override path that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        str: The path value
    """
    # If an override is provided, use it
    if override_path is not None:
        return override_path
    
    # Otherwise, load from config
    if config is None:
        config = load_config()
    
    if "paths" not in config or key not in config["paths"]:
        logger.error(f"Path key '{key}' not found in configuration.")
        raise KeyError(f"Path key '{key}' not found in configuration.")
    
    return config["paths"][key]

def get_audio_config(key: str, override_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get an audio configuration value or use an override if provided.
    
    Args:
        key: The key for the value in the audio section
        override_value: An override value that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        Any: The configuration value
    """
    # If an override is provided, use it
    if override_value is not None:
        return override_value
    
    # Otherwise, load from config
    if config is None:
        config = load_config()
    
    if "audio" not in config or key not in config["audio"]:
        logger.error(f"Audio config key '{key}' not found in configuration.")
        raise KeyError(f"Audio config key '{key}' not found in configuration.")
    
    return config["audio"][key]

def get_processing_config(key: str, override_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a processing configuration value or use an override if provided.
    
    Args:
        key: The key for the value in the processing section
        override_value: An override value that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        Any: The configuration value
    """
    # If an override is provided, use it
    if override_value is not None:
        return override_value
    
    # Otherwise, load from config
    if config is None:
        config = load_config()
    
    if "processing" not in config or key not in config["processing"]:
        logger.error(f"Processing config key '{key}' not found in configuration.")
        raise KeyError(f"Processing config key '{key}' not found in configuration.")
    
    return config["processing"][key]

def get_testlists_config(key: str, override_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a test lists configuration value or use an override if provided.
    
    Args:
        key: The key for the value in the testlists section
        override_value: An override value that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        Any: The configuration value
    """
    # If an override is provided, use it
    if override_value is not None:
        return override_value
    
    # Otherwise, load from config
    if config is None:
        config = load_config()
    
    if "testlists" not in config or key not in config["testlists"]:
        logger.error(f"Test lists config key '{key}' not found in configuration.")
        raise KeyError(f"Test lists config key '{key}' not found in configuration.")
    
    return config["testlists"][key]

def get_model_config(key: str, override_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a model configuration value or use an override if provided.
    
    Args:
        key: The key for the value in the model section
        override_value: An override value that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        Any: The configuration value
    """
    # If an override is provided, use it
    if override_value is not None:
        return override_value
    
    # Otherwise, load from config
    if config is None:
        config = load_config()
    
    if "model" not in config or key not in config["model"]:
        logger.error(f"Model config key '{key}' not found in configuration.")
        raise KeyError(f"Model config key '{key}' not found in configuration.")
    
    return config["model"][key]

def get_ui_config(key: str, override_value: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a UI configuration value or use an override if provided.
    
    Args:
        key: The key for the value in the ui section
        override_value: An override value that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        Any: The configuration value
    """
    # If an override is provided, use it
    if override_value is not None:
        return override_value
    
    # Otherwise, load from config
    if config is None:
        config = load_config()
    
    if "ui" not in config or key not in config["ui"]:
        logger.error(f"UI config key '{key}' not found in configuration.")
        raise KeyError(f"UI config key '{key}' not found in configuration.")
    
    return config["ui"][key]

def get_default_language(override_value: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the default language for the UI from the configuration.
    
    Args:
        override_value: An override language that takes precedence if provided
        config: The configuration dictionary (optional, will be loaded if not provided)
        
    Returns:
        str: The default language code (e.g., "en", "pt_BR")
    """
    try:
        return get_ui_config("default_language", override_value, config)
    except KeyError:
        logger.warning("Default language not found in configuration. Using 'en' as fallback.")
        return "en"

def update_config(new_values: Dict[str, Any], config: Optional[Dict[str, Any]] = None, save: bool = True) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        new_values: Dictionary with new values to update (can be nested)
        config: The configuration dictionary (optional, will be loaded if not provided)
        save: Whether to save the updated configuration
        
    Returns:
        Dict[str, Any]: The updated configuration dictionary
    """
    if config is None:
        config = load_config()
    
    # Helper function to recursively update nested dictionaries
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    # Update configuration
    updated_config = update_dict(config.copy(), new_values)
    
    # Save updated configuration if requested
    if save:
        save_config(updated_config)
    
    return updated_config

def update_plot_config(new_values: Dict[str, Any], config: Optional[Dict[str, Any]] = None, save: bool = True) -> Dict[str, Any]:
    """
    Update plot configuration with new values.
    
    Args:
        new_values: Dictionary with new values to update (can be nested)
        config: The plot configuration dictionary (optional, will be loaded if not provided)
        save: Whether to save the updated configuration
        
    Returns:
        Dict[str, Any]: The updated plot configuration dictionary
    """
    if config is None:
        config = load_plot_config()
    
    # Helper function to recursively update nested dictionaries
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    # Update configuration
    updated_config = update_dict(config.copy(), new_values)
    
    # Save updated configuration if requested
    if save:
        save_plot_config(updated_config)
    
    return updated_config

def setup_directories() -> None:
    """
    Create all directories defined in the configuration paths section.
    This ensures that the required directory structure exists.
    """
    config = load_config()
    
    for key, path in config["paths"].items():
        if key.endswith("_dir") or key.endswith("_prefix"):
            dir_path = path
            if key.endswith("_prefix"):
                dir_path = os.path.dirname(path)
            
            if not os.path.exists(dir_path):
                logger.info(f"Creating directory: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)

def get_config_editor_info() -> str:
    """
    Return information about how to edit the configuration file.
    
    Returns:
        str: Information string
    """
    config_path = get_config_path()
    return (
        f"Configuration file location: {os.path.abspath(config_path)}\n"
        f"You can edit this file directly with any text editor to change settings.\n"
        f"All settings can also be overridden via command-line parameters."
    )

def reload_config() -> Dict[str, Any]:
    """
    Force reload configuration from disk.
    
    Returns:
        Dict[str, Any]: The reloaded configuration dictionary
    """
    return load_config(force_reload=True)

def reload_plot_config() -> Dict[str, Any]:
    """
    Force reload plot configuration from disk.
    
    Returns:
        Dict[str, Any]: The reloaded plot configuration dictionary
    """
    return load_plot_config(force_reload=True)

if __name__ == "__main__":
    # When run directly, setup directories
    config = load_config()
    setup_directories()
    print("Configuration loaded and directories created.")
    print("\n" + get_config_editor_info()) 