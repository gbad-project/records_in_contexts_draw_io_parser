# Originally generated with ChatGPT-4o on 2024-08-23,
# with subsequent modifications

import os
import sys
import platform
import glob
import subprocess
import hashlib

def parse():
    def quote_argument(arg):
        """Properly escape and quote arguments for subprocess."""
        return subprocess.list2cmdline([arg])

    def usage():
        print("Usage: python script.py <input_drawio_file> [optional commands for the parser]")
        sys.exit(1)

    if len(sys.argv) < 2:
        usage()

    def calculate_md5(file_path):
        """Calculate the MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()

    # Get the input file path
    input_file = sys.argv[1]
    optional_commands = sys.argv[2:]

    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}")
        sys.exit(1)

    # Construct the output file path
    output_file_dir = os.path.dirname(input_file)
    output_file_name = os.path.basename(input_file).replace('.drawio', '').lower().replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('/', '').replace(',', '').replace(':', '').replace('.', '').replace('"', '').replace("'", '') + '.owl'
    output_file = os.path.join(output_file_dir, output_file_name)

    # Construct the TTL file path
    ttl_file = os.path.join(output_file_dir, os.path.basename(output_file).replace('.owl', '.ttl'))

    ttl_file_hash = ''
    # Check if the TTL file exists and get its hash
    if os.path.isfile(ttl_file):
        ttl_file_hash = calculate_md5(ttl_file)

    # Default optional commands if none are provided
    if not optional_commands:
        optional_commands = ["-m", "url",
                             "-o", "http://gbad.archives.gov.on.ca",
                             "-p", "http://gbad.archives.gov.on.ca/"]

    # Properly quote the optional commands
    quoted_commands = ' '.join([quote_argument(arg) for arg in optional_commands])

    # Construct the python command
    python_command = f'cat "{input_file}" | python draw_io_parser.py {quoted_commands} > "{output_file}"'

    # Print and execute the python command
    print(f"Executing command: {python_command}")
    subprocess.run(python_command, shell=True, check=True)

    # Print the output message
    print(f"Manchester OWL Output saved to: {output_file}")

    # Convert the OWL file to TTL
    robot_sh_path = "./robot.sh"
    conversion_command = f'{robot_sh_path} convert -i "{output_file}" -o "{ttl_file}"'

    # Print and execute the conversion command
    print(f"Executing command: {conversion_command}")
    subprocess.run(conversion_command, shell=True, check=True)

    # Check if the TTL file exists and if it has changed
    if not os.path.isfile(ttl_file):
        print(f"Error: This TTL does not exist: {ttl_file}")
    elif ttl_file_hash == calculate_md5(ttl_file):
        print(f"Warning: This TTL exists but is unchanged: {ttl_file}")
    else:
        print(f"TTL Output saved to: {ttl_file}")

def parse_auth():
    def find_first_drawio_file(directory):
        """
        Finds the first .drawio file in the specified directory.
        """
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".drawio"):
                    return os.path.join(root, file)
        return None

    # Path to your original Python script
    original_script_path = "./process_drawio.py"  # Ensure this path is correct

    # Set the directory containing the graph
    graph_dir = "gbad/schema/authority"

    # Find the first *.drawio file in the graph directory
    drawio_file = find_first_drawio_file(graph_dir)
    if not drawio_file:
        print(f"No .drawio file found in the directory: {graph_dir}")
        sys.exit(1)

    # Set desired args
    args = [
        "-m", "url",
        "-c", "none",
        "-o", "https://data.archives.gov.on.ca",
        "-p", "https://data.archives.gov.on.ca/"
    ]

    # Construct the python command
    python_command = ['python', original_script_path, drawio_file] + args

    # Print and execute the python command
    print(f"Executing command: {' '.join(python_command)}")
    subprocess.run(python_command, check=True)
    
def is_cygwin():
    """Check if the system is Cygwin."""
    return platform.system() == 'CYGWIN'

def resolve_symlink(script_path):
    """Resolve symbolic links to get the actual path of the script."""
    while os.path.islink(script_path):
        script_path = os.readlink(script_path)
        if not os.path.isabs(script_path):
            script_path = os.path.join(os.path.dirname(sys.argv[0]), script_path)
    return os.path.abspath(script_path)

def robot():
    # Determine the path to this script
    robot_script = os.path.abspath(sys.argv[0])

    # Resolve symbolic links to get the actual script path
    robot_script = resolve_symlink(robot_script)

    # Directory that contains the script
    script_dir = os.path.dirname(robot_script)

    # Check if the system is Cygwin
    cygwin = is_cygwin()

    # Construct the Java command
    jar_path = os.path.join(script_dir, 'robot.jar')
    if cygwin:
        # Use cygpath to convert paths for Cygwin
        jar_path = subprocess.check_output(['cygpath', '-w', jar_path]).decode().strip()

    # Prepare the command
    java_command = ['java', '-jar', jar_path] + sys.argv[1:]

    # Execute the Java command
    print(f"Executing command: {' '.join(java_command)}")
    subprocess.run(java_command, check=True)

if __name__ == '__main__':
    parse_auth()

