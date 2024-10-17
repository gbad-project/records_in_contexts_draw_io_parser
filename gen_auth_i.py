import os
import re
import glob

# Originally generated with ChatGPT-4o on 2024-08-29,
# with modifications; ported to Python on 2024-10-17

# Set the directory containing the graph
graph_dir = "gbad/schema/authority_i"

# Find the first .drawio file in the graph directory
drawio_files = glob.glob(os.path.join(graph_dir, "*.drawio"))
drawio_file = drawio_files[0] if drawio_files else None

if not drawio_file:
    print("No .drawio file found in the specified directory.")
    exit(1)

# Define a function to get the custom end range for different patterns
ranges = {
    "VAR": 11,       # Custom range for VAR
    "SUC": 4,       # Custom range for SUC
    "AUTHTP": 2,   # Custom range for AUTHTP
    "PRED": 5,      # Custom range for PRED
    "CONTAG": 14,    # Custom range for CONTAG
    "DATECONT": 14  # Custom range for DATECONT
}

def get_end_range(pattern): return ranges.get(pattern, 20)  # Default end range is 20

# Read the content of the original .drawio file
with open(drawio_file, "r") as file:
    file_content = file.read()

# Define the replacement patterns
replacement_patterns = [
    (r"PAR_{start}", "PAR_{i}"),
    (r"VAR_{start}", "VAR_{i}"),
    (r"Schema/Authority/CorporateBodyType/{AUTHTP_{start}}", "Schema/Authority/CorporateBodyType/{AUTHTP_{i}}"),
    (r"{RICO_VERSION}/{SUC_{start}}/CorporateBody/1", "{RICO_VERSION}/{SUC_{i}}/CorporateBody/1"),
    (r"{RICO_VERSION}/{PRED_{start}}/CorporateBody/1", "{RICO_VERSION}/{PRED_{i}}/CorporateBody/1"),
    (r"{RICO_VERSION}/{HEADING}/AgentControlRelation/{start}", "RiC-O_1-0-1/{HEADING}/AgentControlRelation/{i}"),
    (r"{RICO_VERSION}/{CONTAG_{start}}/CorporateBody/1", "RiC-O_1-0-1/{CONTAG_{i}}/CorporateBody/1"),
    (r"KB/Date/{DATECONT_{start}_END}", "KB/Date/{DATECONT_{i}_END}"),
    (r"KB/Date/{DATECONT_{start}_BEGINNING}", "KB/Date/{DATECONT_{i}_BEGINNING}")
]

# Loop through the desired patterns and perform replacements
for pattern in list(ranges.keys()):
    # Get the custom end range for the current pattern
    custom_end = get_end_range(pattern)

    # Loop through the range for the current pattern
    for i in range(2, custom_end + 1):  # Start from 2 (start + 1)
        # Perform the replacements
        updated_content = file_content
        for old, new in replacement_patterns:
            # Replace {start} and {i} placeholders in the new pattern
            old_pattern = old.format(start=1)
            new_pattern = new.format(i=i)
            # Perform the replacement
            updated_content = re.sub(old_pattern, new_pattern, updated_content)

        # Define the output file name with the new suffix
        output_file = f"{os.path.splitext(drawio_file)[0]}_{pattern}_{i}.drawio"

        # Save the modified content to the new file
        with open(output_file, "w") as output:
            output.write(updated_content)

        print(f"Generated {output_file} with replacements for {pattern}.")
