# Parent Commit: ff32a4019b8eb452687b8a8b9a01edacee434140
# SHA1 Hash: 44974b8f8441556e26bd1e7d1b6e492fb9cc3d44

# Originally generated with ChatGPT-4o on 2024-12-13, modified
import os
import sys
import subprocess
import re

def sanitize_filename(filename):
    """
    Sanitize filename using the specified algorithm:
    - Convert to lowercase
    - Replace spaces with underscores
    - Remove special characters
    """
    # Convert to lowercase
    sanitized = filename.lower()
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Remove specified special characters
    sanitized = re.sub(r'[()[\]/,:."\']', '', sanitized)
    
    return sanitized

def convert_drawio_file(script_dir):
    """
    Find and convert a single DrawIO file in the script directory
    
    Args:
        script_dir (str): Directory to search for .drawio file
    """
    # Find the single .drawio file
    drawio_files = [
        f for f in os.listdir(script_dir) 
        if f.lower().endswith('.drawio')
    ]
    
    if len(drawio_files) == 0:
        print("No .drawio file found in the directory.")
        sys.exit(1)
    
    if len(drawio_files) > 1:
        print("Multiple .drawio files found. Only one file is expected.")
        sys.exit(1)
    
    input_file = os.path.join(script_dir, drawio_files[0])
    
    # Create output directory name based on DrawIO filename
    base_filename = os.path.splitext(drawio_files[0])[0]
    output_dir_name = sanitize_filename(base_filename)
    output_dir = os.path.join(script_dir, output_dir_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output file paths
    output_file = os.path.join(output_dir, 'drawio.owl')
    ttl_file = os.path.join(output_dir, 'drawio.ttl')
    
    try:
        # Prepare default parser commands
        parser_commands = [
            '-m', 'url', 
            '-o', 'http://gbad.archives.gov.on.ca', 
            '-p', 'http://gbad.archives.gov.on.ca/'
        ]
        
        # Prepare full argument list for _run
        full_args = [input_file] + parser_commands
        
        # Call _run function (assuming it's defined in the parent context)
        sys.argv = full_args
        import draw_io_parser  # Assuming this is imported in parent context
        
        # Capture stdout to write OWL file
        import io
        import contextlib
        
        with open(output_file, 'w') as owl_out:
            with contextlib.redirect_stdout(owl_out):
                draw_io_parser._run()
        
        print(f"OWL Output saved to: {output_file}")
        
        # Convert OWL to TTL using robot
        robot_cmd = [
            'java', 
            '-jar', 
            os.path.join(script_dir, 'robot.jar'), 
            'convert', 
            '-i', output_file, 
            '-o', ttl_file
        ]
        
        subprocess.run(robot_cmd, check=True)
        print(f"TTL Output saved to: {ttl_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    """
    Main function to convert a single DrawIO file
    """
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert the DrawIO file
    convert_drawio_file(script_dir)

if __name__ == '__main__':
    main()
