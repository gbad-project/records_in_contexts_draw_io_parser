# Originally generated with ChatGPT-4o on 2024-08-23,
# with subsequent modifications

import os
import sys
import glob
import subprocess
import hashlib

def map_rml():
    # Define directories
    rml_dir = "gbad/schema/authority"
    ttl_root = "gbad/mapping/target"

    # Find the .rml file
    rml_files = glob.glob(os.path.join(rml_dir, "*.rml"))

    if rml_files:
        rml = rml_files[0]  # Assuming you want the first .rml file found
        rml_filename = os.path.splitext(os.path.basename(rml))[0]

        # Create target directory if it does not exist
        ttl_dir = os.path.join(ttl_root, rml_filename)
        os.makedirs(ttl_dir, exist_ok=True)
        
        # Define the output file
        ttl = os.path.join(ttl_dir, "mapped.ttl")

        # Run the Java command
        java_command = ["java", "-jar", "rmlmapper*", "-s", "turtle", "-m", rml, "-o", ttl]
        subprocess.run(java_command, check=True)
    else:
        print("No .rml files found in the directory.")

if __name__ == '__main__':
    map_rml()
