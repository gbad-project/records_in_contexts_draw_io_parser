# Originally generated with ChatGPT-4o on 2024-08-23,
# with subsequent modifications

import os
#import sys
import glob
import subprocess
#import hashlib
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, DCTERMS, NamespaceManager
from pprint import pprint
import argparse
import shutil

def map_rml(schema_code):
    """
    Returns a tuple of (rml, rmlmapper, ttl) paths.
    """

    # Define directories
    if schema_code == 'add':
        rml_dir = "gbad/schema/description-listings"
    elif schema_code == 'auth':
        rml_dir = "gbad/schema/authority"
    else:
        raise Exception(f"Fatal error: Schema code not supplied.")
    
    ttl_root = "gbad/mapping/target"
    rmlmapper_dir = "."

    # Find the .rml file
    rml_files = glob.glob(os.path.join(os.path.normpath(rml_dir), "*.rml"))
    rmlmapper_files = glob.glob(os.path.join(os.path.normpath(rmlmapper_dir), "rmlmapper*"))

    return_tuple = (None, None, None)
    if (rml_files and rmlmapper_files):
        rml = rml_files[0]  # Assuming you want the first .rml file found
        rmlmapper = rmlmapper_files[0] # Same assumption for mapper jar
        rml_filename = os.path.splitext(os.path.basename(rml))[0]

        # Create target directory if it does not exist
        ttl_dir = os.path.join(os.path.normpath(ttl_root), rml_filename)
        os.makedirs(ttl_dir, exist_ok=True)
        
        # Define the output file
        mapped_filename = "mapped.ttl"
        ttl = os.path.join(ttl_dir, mapped_filename)

        if os.path.exists(ttl):
            mapped_backup_filename = "mapped.ttl.backup"
            ttl_backup = os.path.join(ttl_dir, mapped_backup_filename)
            try:
                os.rename(ttl, ttl_backup)
                print(f"File '{mapped_filename}' already exists - renamed to '{mapped_backup_filename}'")
            except PermissionError:
                print(f"Aborted: File '{mapped_filename}' already exists and cannot be renamed for backup due to a permission error.")

        return_tuple = (rml, rmlmapper, ttl)
        print("Initiated mapping params:")
        pprint(return_tuple)

        # Run the Java command
        java_command = ["java", "-jar", rmlmapper, "-s", "turtle", "-m", rml, "-o", ttl]
        try:
            subprocess.run(java_command, check=True)

            if os.path.exists(ttl):
                file_size_bytes = os.path.getsize(ttl)
                file_size_mb = file_size_bytes / (1024 * 1024)

                if file_size_mb > 10:
                    print(f"Converted file is larger than 10 MB ({file_size_mb:.2f} MB) - trying to rename to LARGE...")
                    try:
                        mapped_large_filename = "mapped_LARGE.ttl"
                        os.rename(ttl, mapped_large_filename)
                        print(f"Successfully renamed to '{mapped_large_filename}'")
                        # Update returned params
                        ttl = os.path.join(ttl_dir, mapped_large_filename)
                        return_tuple = (rml, rmlmapper, ttl)
                    except PermissionError:
                        print(f"Aborted: Could not rename due to a permission error.")
                else:
                    pass
            
            print(f"Successfully mapped '{rml}' to '{ttl}'\n")
        
        except Exception as e:
            print(f"Failed to run mapper jar: '{e}'")
    else:
        print("No .rml and/or mapper files found in specified paths.")
    
    return return_tuple
    
def postprocess(graph_path):
    # Create the input RDF graph
    base_uri = 'https://data.archives.gov.on.ca'
    format = 'turtle'  # Adjust the format as needed
    g = Graph()

    # Define custom prefixes
    rico_uri = 'https://www.ica.org/standards/RiC/ontology#'
    rico = ('rico', Namespace(rico_uri))
    ns = ('', Namespace(URIRef(f"{base_uri}/")))

    # Define common prefixes
    rdf = ('rdf', RDF)
    rdfs = ('rdfs', RDFS)
    owl = ('owl', OWL)

    try:    
        g.parse(graph_path,
                format=format)
        print(f"Successfully read a graph from '{graph_path}'",
              f"\nNumber of triples in the graph: {len(g)}")
        #print(g.serialize(format='turtle')) # debug
    except Exception as e:
        print(f"Failed to read graph from '{graph_path}'",
              f"\nError: '{e}'")
        
    # Bind prefixes to namespaces
    g.namespace_manager.bind(*rico)
    g.namespace_manager.bind(*ns, replace=True) # otherwise defaults to mapping
    g.namespace_manager.bind(*rdf)
    g.namespace_manager.bind(*rdfs)
    g.namespace_manager.bind(*owl)

    # Iterate over namespaces
    #for prefix, uri in g.namespace_manager.namespaces():
    #    print(f"Prefix: {prefix}, URI: {uri}")

    return g

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map schema of choice")
    parser.add_argument("schema", help="Choose one: add or auth.")

    args = parser.parse_args()

    rml, rmlmapper, ttl = map_rml(str(args.schema).lower())
    graph = postprocess(ttl)
