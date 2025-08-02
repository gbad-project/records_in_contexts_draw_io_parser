#!/bin/bash

# originally generated with Gemini Free Version on 2024-06-05,
# revised on 2024-07-10, with modifications

# Path to your original script (replace with the actual path)
ORIGINAL_SCRIPT_PATH="./parse.sh"

# Set the directory containing the graph
# script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
graph_dir="gbad/schema/description-listings"

# Find the first *.drawio file in the graph directory
drawio_file=$(find "$graph_dir" -type f -name "*.drawio" | head -n 1)

export BASE_URI='https://data.archives.gov.on.test.gbad.ca'

# Set desired args
args="-m url \
      -c none \
      --label-disable \
      -o $BASE_URI/Schema/Mapping \
      -x map  -p $BASE_URI/Schema/Mapping# \
      -x rico -p https://www.ica.org/standards/RiC/ontology# \
      -x rdfs -p http://www.w3.org/2000/01/rdf-schema# \
      -x owl  -p http://www.w3.org/2002/07/owl# \
      -x add  -p file:///$PWD/gbad/schema/description-listings.ttl
      "
      #-x add  -p https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/
      #-x dcat -p http://www.w3.org/ns/dcat# \
      #-x skos -p http://www.w3.org/2004/02/skos/core#
      #"

# Construct the python command
verbose=""  # -vvv or empty string - verbosity for ROBOT
python_command="\"$ORIGINAL_SCRIPT_PATH\" \"$drawio_file\" $verbose $args"

# Run the parser script
eval "$python_command"
