#!/usr/bin/env python3
"""
Save YAML configuration script for BoltzGen skill.

This script saves YAML configuration text generated from user conversation
to a file, supporting the conversational configuration workflow.

Usage:
    python save_yaml_config.py --yaml-text "<yaml_content>" --output-dir <dir> [--filename <name>]
    python save_yaml_config.py --input-file <path> --output-dir <dir> [--filename <name>]
"""

import argparse
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def validate_yaml_content(yaml_text: str) -> bool:
    """
    Validate YAML content format.

    Args:
        yaml_text: YAML configuration text to validate

    Returns:
        True if valid, False otherwise

    Raises:
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        yaml.safe_load(yaml_text)
        return True
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format: {e}")


def generate_filename(protocol: Optional[str] = None, prefix: str = "boltzgen_config") -> str:
    """
    Generate a filename for the YAML configuration.

    Args:
        protocol: BoltzGen protocol name (e.g., 'protein-anything', 'peptide-anything')
        prefix: Filename prefix

    Returns:
        Generated filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if protocol:
        # Sanitize protocol name for filename
        safe_protocol = protocol.replace("-", "_").replace(" ", "_")
        return f"{prefix}_{safe_protocol}_{timestamp}.yaml"
    else:
        return f"{prefix}_{timestamp}.yaml"


def save_yaml_config(
    yaml_text: str,
    output_dir: Union[str, Path],
    filename: Optional[str] = None,
    protocol: Optional[str] = None,
    validate: bool = True,
    create_dir: bool = True
) -> Path:
    """
    Save YAML configuration text to a file.

    Args:
        yaml_text: YAML configuration text content
        output_dir: Directory to save the file
        filename: Optional custom filename (auto-generated if not provided)
        protocol: BoltzGen protocol for filename generation
        validate: Whether to validate YAML format before saving
        create_dir: Whether to create output directory if it doesn't exist

    Returns:
        Path to the saved YAML file

    Raises:
        ValueError: If validation fails or path is invalid
        OSError: If file writing fails
    """
    # Convert to Path object
    output_path = Path(output_dir)

    # Create directory if needed
    if create_dir and not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Validate YAML content
    if validate:
        validate_yaml_content(yaml_text)

    # Generate filename if not provided
    if not filename:
        filename = generate_filename(protocol=protocol)

    # Full file path
    file_path = output_path / filename

    # Write YAML content
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(yaml_text)
        print(f"✓ YAML configuration saved to: {file_path}")
        return file_path
    except OSError as e:
        raise OSError(f"Failed to write file: {e}")


def load_yaml_from_file(input_file: Union[str, Path]) -> str:
    """
    Load YAML content from an existing file.

    Args:
        input_file: Path to existing YAML file

    Returns:
        YAML content as string

    Raises:
        FileNotFoundError: If input file doesn't exist
        OSError: If file reading fails
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()
    except OSError as e:
        raise OSError(f"Failed to read file: {e}")


def extract_protocol_from_yaml(yaml_text: str) -> Optional[str]:
    """
    Extract protocol name from YAML content if present.

    Args:
        yaml_text: YAML configuration text

    Returns:
        Protocol name if found, None otherwise
    """
    try:
        config = yaml.safe_load(yaml_text)
        if isinstance(config, dict) and 'protocol' in config:
            return config['protocol']
    except yaml.YAMLError:
        pass
    return None


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Save YAML configuration for BoltzGen skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--yaml-text',
        type=str,
        help='YAML configuration text content (inline or from conversation)'
    )
    input_group.add_argument(
        '--input-file',
        type=str,
        help='Path to existing YAML file to copy/move'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./configs',
        help='Directory to save the YAML file (default: ./configs)'
    )
    parser.add_argument(
        '--filename',
        type=str,
        help='Custom filename for the saved YAML (auto-generated if not provided)'
    )
    parser.add_argument(
        '--protocol',
        type=str,
        help='BoltzGen protocol name for filename generation (e.g., protein-anything)'
    )

    # Behavior options
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Validate YAML format before saving (default: True)'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip YAML validation'
    )
    parser.add_argument(
        '--create-dir',
        action='store_true',
        default=True,
        help='Create output directory if it doesn\'t exist (default: True)'
    )

    args = parser.parse_args()

    # Determine validation setting
    validate = args.validate and not args.no_validate

    # Get YAML content
    if args.yaml_text:
        yaml_text = args.yaml_text
        # Try to extract protocol from YAML if not provided
        if not args.protocol:
            args.protocol = extract_protocol_from_yaml(yaml_text)
    else:
        yaml_text = load_yaml_from_file(args.input_file)

    # Save the configuration
    try:
        saved_path = save_yaml_config(
            yaml_text=yaml_text,
            output_dir=args.output_dir,
            filename=args.filename,
            protocol=args.protocol,
            validate=validate,
            create_dir=args.create_dir
        )

        # Print summary
        print(f"\nConfiguration Details:")
        print(f"  - Output path: {saved_path}")
        print(f"  - Protocol: {args.protocol or 'Not specified'}")
        print(f"  - Validated: {validate}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())