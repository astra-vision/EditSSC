"""
Configuration Loader for Diffusion Experiments

This module provides utilities to:
1. Load base configuration from YAML files
2. Override specific config values from command-line arguments
3. Save the final configuration for reproducibility
"""

import yaml
import argparse
import os
import re
from typing import Any, Dict
from pathlib import Path


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.

    Automatically converts string scientific notation (e.g., '1e-6') to floats
    to ensure proper type handling.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert string scientific notation to floats (YAML sometimes loads 1e-6 as string)
    config = _convert_scientific_notation_strings(config)

    print(f"✓ Loaded base config from: {yaml_path}")
    return config


def _convert_scientific_notation_strings(obj: Any) -> Any:
    """
    Recursively convert string scientific notation (e.g., '1e-6', '1e-08') to floats.

    This fixes an issue where YAML loads values like '1e-6' as strings instead of floats.
    """
    if isinstance(obj, dict):
        return {key: _convert_scientific_notation_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_scientific_notation_strings(item) for item in obj]
    elif isinstance(obj, str):
        # Check if string looks like scientific notation (e.g., '1e-6', '1.0e-08', '1e+5')
        scientific_pattern = r'^[+-]?(\d+\.?\d*|\d*\.\d+)[eE][+-]?\d+$'
        if re.match(scientific_pattern, obj.strip()):
            try:
                return float(obj)
            except ValueError:
                return obj
        return obj
    else:
        return obj


def override_config(config: Dict[str, Any], overrides: str) -> Dict[str, Any]:
    """
    Override configuration values using dot notation.

    This function allows you to modify nested config values using a simple
    dot notation syntax from the command line.

    Args:
        config: The base configuration dictionary
        overrides: String with format "key1=value1 key2.nested=value2"
                  Example: "batch_size=10 train.lr=0.001"

    Returns:
        Updated configuration dictionary

    Example:
        >>> config = {'batch_size': 1, 'train': {'lr': 0.0005}}
        >>> override_config(config, 'batch_size=10 train.lr=0.001')
        {'batch_size': 10, 'train': {'lr': 0.001}}
    """
    if not overrides or overrides.strip() == "":
        return config

    # Split the override string into individual key=value pairs
    override_pairs = overrides.split()

    for pair in override_pairs:
        if '=' not in pair:
            print(f"⚠ Warning: Skipping malformed override '{pair}' (missing '=')")
            continue

        key_path, value_str = pair.split('=', 1)

        # Parse the value (try to convert to appropriate type)
        value = parse_value(value_str)

        # Set the value in the config dictionary
        set_nested_value(config, key_path, value)
        print(f"  → Overriding: {key_path} = {value}")

    return config


def parse_value(value_str: str) -> Any:
    """
    Parse a string value and convert it to the appropriate Python type.

    This function tries to intelligently convert string values to:
    - bool (for 'true'/'false')
    - list (for YAML list syntax like '[1,2,4]' or '[1, 2, 4]')
    - int (for integer strings)
    - float (for decimal strings)
    - None (for 'null'/'none')
    - str (fallback)

    Args:
        value_str: String representation of the value

    Returns:
        Parsed value with appropriate type

    Examples:
        >>> parse_value("true")
        True
        >>> parse_value("[1,2,4]")
        [1, 2, 4]
        >>> parse_value("42")
        42
        >>> parse_value("3.14")
        3.14
        >>> parse_value("hello")
        'hello'
    """
    # Strip surrounding quotes (single or double) if present
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        value_str = value_str[1:-1]

    # Handle boolean values
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # Handle null/none
    if value_str.lower() in ['null', 'none']:
        return None

    # Try to parse as YAML list (e.g., [1,2,4] or [1, 2, 4])
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            # Use yaml.safe_load to parse the list
            parsed = yaml.safe_load(value_str)
            if isinstance(parsed, list):
                return parsed
        except (yaml.YAMLError, ValueError):
            pass  # If parsing fails, continue to other parsers

    # Try to parse as integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string if nothing else works
    return value_str


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using dot notation and array indexing.

    This function navigates through nested dictionaries and sets the value
    at the specified path. It creates intermediate dictionaries if needed.
    It also supports array/list indexing with bracket notation.

    Args:
        config: The configuration dictionary to modify
        key_path: Dot-separated path to the value (e.g., "train.optimizer.lr")
                 Can also include array indices (e.g., "tri_size[2]")
        value: The value to set

    Examples:
        >>> config = {'train': {'lr': 0.001}}
        >>> set_nested_value(config, 'train.lr', 0.01)
        >>> config
        {'train': {'lr': 0.01}}

        >>> config = {'tri_size': [128, 128, 32]}
        >>> set_nested_value(config, 'tri_size[2]', 64)
        >>> config
        {'tri_size': [128, 128, 64]}
    """
    keys = key_path.split('.')

    # Navigate to the parent of the final key
    current = config
    for key in keys[:-1]:
        # Check if this key has array indexing (e.g., "list[0]")
        array_match = re.match(r'(.+)\[(\d+)\]$', key)
        if array_match:
            base_key = array_match.group(1)
            index = int(array_match.group(2))

            if base_key not in current:
                current[base_key] = []
            current = current[base_key][index]
        else:
            # Regular dictionary key
            if key not in current:
                current[key] = {}
            current = current[key]

    # Set the final value (handle array indexing for the last key too)
    final_key = keys[-1]
    array_match = re.match(r'(.+)\[(\d+)\]$', final_key)

    if array_match:
        # It's an array access like "tri_size[2]"
        base_key = array_match.group(1)
        index = int(array_match.group(2))

        if base_key not in current:
            raise KeyError(f"Cannot set array element: '{base_key}' does not exist in config")

        if not isinstance(current[base_key], list):
            raise TypeError(f"Cannot use array indexing: '{base_key}' is not a list")

        if index >= len(current[base_key]):
            raise IndexError(f"Index {index} out of range for '{base_key}' (length {len(current[base_key])})")

        # Set the array element
        current[base_key][index] = value
    else:
        # Regular dictionary key
        current[final_key] = value


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save the final configuration to a YAML file for reproducibility.

    This ensures you can always see exactly what configuration was used
    for each experiment.

    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Saved final config to: {save_path}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for experiment configuration.

    This sets up the argument parser to handle:
    - Base config file path
    - Working directory for outputs
    - Configuration overrides

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Run diffusion experiment with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with base config only
  python script.py --config common_diffusion_base.yaml --work-dir exp_default

  # Override batch size and learning rate
  python script.py --config common_diffusion_base.yaml --work-dir exp_bs10 \\
      --cfg-options "batch_size=10 diff_lr=0.001"

  # Override nested values
  python script.py --config common_diffusion_base.yaml --work-dir exp_test \\
      --cfg-options "steps=50 use_ddim=true save_interval=1000"
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to base YAML configuration file'
    )

    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='Directory where experiment outputs will be saved'
    )

    parser.add_argument(
        '--cfg-options',
        type=str,
        default="",
        help='Configuration overrides in format "key1=value1 key2=value2"'
    )

    return parser.parse_args()


def load_experiment_config() -> tuple[Dict[str, Any], str]:
    """
    Main function to load and prepare experiment configuration.

    This function orchestrates the entire config loading process:
    1. Parse command-line arguments
    2. Load base YAML config
    3. Apply overrides
    4. Set up work directory
    5. Save final config

    Returns:
        Tuple of (config_dict, work_directory_path)
    """
    # Parse command-line arguments
    args = parse_arguments()

    print("\n" + "="*60)
    print("🔧 Loading Experiment Configuration")
    print("="*60)

    # Load base configuration from YAML
    config = load_yaml_config(args.config)

    # Apply command-line overrides
    if args.cfg_options:
        print("\n📝 Applying overrides:")
        config = override_config(config, args.cfg_options)

    # Update save_path in config to match work-dir
    config['save_path'] = args.work_dir

    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Work directory: {work_dir}")

    # Save the final configuration for reproducibility
    config_save_path = work_dir / "config_used.yaml"
    save_config(config, config_save_path)

    print("="*60)
    print("✅ Configuration ready!\n")

    return config, str(work_dir)


# Example usage and testing
if __name__ == "__main__":
    """
    Test the config loader with the common diffusion base config.

    Usage:
        python config_loader.py --config common_diffusion_base.yaml \\
            --work-dir test_exp --cfg-options "batch_size=10"
    """
    config, work_dir = load_experiment_config()

    # Print some key config values for verification
    print("\n📊 Key Configuration Values:")
    print("-" * 40)
    print(f"Dataset: {config.get('dataset', 'N/A')}")
    print(f"Batch size: {config.get('batch_size', 'N/A')}")
    print(f"Diffusion steps: {config.get('steps', 'N/A')}")
    print(f"Learning rate: {config.get('diff_lr', 'N/A')}")
    print(f"Save path: {config.get('save_path', 'N/A')}")
    print("-" * 40)

