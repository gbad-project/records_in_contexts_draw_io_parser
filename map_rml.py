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
from io import BytesIO
import re
import json
import importlib.util

BASE_URI = 'https://data.archives.gov.on.test.gbad.ca/'

def map_rml(config_path, dataset_name, rml_path=None):
    """
    Returns a tuple of (rml, rmlmapper, ttl) paths.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    dataset_config = config[dataset_name]

    # Define directories
    rml_dir = os.path.dirname(dataset_config['drawio_file'])
    
    rml_files = []
    if rml_path:  # Assume it is relative to rml_dir
        try_rml_subdir = os.path.join(os.path.normpath(rml_dir),
                                      os.path.normpath(rml_path))
        if os.path.isdir(try_rml_subdir):
            rml_dir = try_rml_subdir
        else:  # this is actually a path to RML file
            rml_files.append(try_rml_subdir)
    
    ttl_root = "gbad/mapping/target"
    #rmlmapper_dir = "riconverted_general_authority_to_ric-o_model_2024-11-25_pz"
    rmlmapper_dir = "."

    # Find the .rml file
    if len(rml_files) == 0:
        rml_files = glob.glob(os.path.join(os.path.normpath(rml_dir), "**/*.rml"), recursive=True)

    return_tuple = (None, None, None)
    if (rml_files):
        rml = rml_files[0]  # Assuming you want the first .rml file found
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
                #os.rename(ttl, ttl_backup)
                shutil.copy2(ttl, ttl_backup)
                print(f"File '{mapped_filename}' already exists - copied to '{mapped_backup_filename}'")
            except PermissionError:
                print(f"Aborted: File '{mapped_filename}' already exists and cannot be renamed for backup due to a permission error.")

        def extract_source_csv_path(rml_path):
            try:
                print("Trying to extract source CSV path from RML...")
                with open(rml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                pattern = r'rml:source "(.+)"'
                match = re.search(pattern, content)

                if match:
                    source_csv_path = match.group(1)
                    source_csv_filename = os.path.splitext(os.path.basename(source_csv_path))[0]
                    print(f"Extracted rml:source path: '{source_csv_path}'")
                    return (source_csv_path, source_csv_filename)
                else:
                    raise Exception(f"Error: rml:source not found in the RML file.")
            except FileNotFoundError:
                print(f"Error: File '{rml_path}' not found.")
                return (None, None)
            except Exception as e:
                print(f"Error: {e}")
                return (None, None)

        source_csv_path, source_csv_filename = extract_source_csv_path(rml)
        if source_csv_filename:
            subdir = os.path.join(ttl_dir, source_csv_filename)
            os.makedirs(subdir, exist_ok=True)
            ttl = os.path.join(subdir, mapped_filename)

        def map_using_rmlmapper(rml, ttl):
            """Run the RMLMapper Java command."""
            nonlocal rmlmapper_dir
            rmlmapper_files = glob.glob(os.path.join(os.path.normpath(rmlmapper_dir), "rmlmapper*"))
            
            if (rml_files and rmlmapper_files):
                rmlmapper = rmlmapper_files[0] # Same assumption for mapper jar
            else:
                raise Exception("No mapper file found in specified path.")

            return_tuple = (rml, rmlmapper, ttl)
            print("Initiated mapping params:")
            pprint(return_tuple)

            java_command = [
                "java", "-jar", rmlmapper,
                "-s", "turtle",
                "-m", rml,
                "-o", ttl,
                "-b", BASE_URI[:-1]
            ]
            try:
                print(f"\n\nRunning Java command: '{" ".join(java_command)}'\n\n")
                subprocess.run(java_command, check=True, capture_output=True, text=True)
            # Catch CalledProcessError specifically for command failures
            except subprocess.CalledProcessError as e:
                print(f"Failed to run mapper jar (exit code: {e.returncode}):")
                print(f"Command: {e.cmd}")
                if e.stdout:
                    print(f"STDOUT:\n{e.stdout}")
                if e.stderr:
                    print(f"STDERR:\n{e.stderr}")
                # You can re-raise the exception if you want to propagate it after logging
                raise
            except Exception as e:
                print(f"Failed to run mapper jar: '{e}'")
                raise
            return return_tuple

        def map_using_pyrml(rml, ttl):
            """Create an instance of RML Mapper with PyRML."""
            raise NotImplementedError()
            return_tuple = (rml, 'pyrml', ttl)
            print("Initiated mapping params:")
            pprint(return_tuple)

            try:
                from pyrml import PyRML
                mapper = PyRML.get_mapper()
                mapped_graph = mapper.convert(rml)
                mapped = mapped_graph.serialize(format='turtle')
                os.makedirs(os.path.dirname(ttl), exist_ok=True)
                with open(ttl, 'w') as f:
                    f.write(mapped)
            except Exception as e:
                print(f"Failed to map using pyrml: '{e}'")
                raise
            return return_tuple
        
        try:
            return_tuple = map_using_rmlmapper(rml, ttl)
            #return_tuple = map_using_pyrml(rml, ttl)
            rml, rmlmapper, ttl = return_tuple
            if os.path.exists(ttl):
                file_size_bytes = os.path.getsize(ttl)
                file_size_mb = file_size_bytes / (1024 * 1024)

                if file_size_mb > 10:
                    print(f"Converted file is larger than 10 MB ({file_size_mb:.2f} MB) - trying to rename to LARGE...")
                    try:
                        mapped_large_filename = "mapped_LARGE.ttl"
                        large_ttl = os.path.join(os.path.dirname(ttl), mapped_large_filename)
                        os.rename(ttl, large_ttl)
                        print(f"Successfully renamed to '{mapped_large_filename}'")
                        # Update returned params
                        ttl = large_ttl
                        return_tuple = (rml, rmlmapper, ttl)
                    except PermissionError:
                        print(f"Aborted: Could not rename due to a permission error.")
                else:
                    pass
            print(f"Successfully mapped '{rml}' to '{ttl}'\n")
        except Exception as e:
            print(f"\n\nException occurred: {e}\n\n")
    else:
        print("No .rml file found in specified path.")
    
    return return_tuple
    
def postprocess(graph_path, postprocessor_path=None):
    # Create the input RDF graph
    base_kb_uri = URIRef(os.path.join(BASE_URI, "KB"))
    base_schema_uri = URIRef(os.path.join(BASE_URI, "Schema"))
    base_auth_uri = URIRef(os.path.join(base_schema_uri, "Authority"))
    base_add_uri = URIRef(os.path.join(base_schema_uri, "Description-Listings"))
    base_mapping_uri = URIRef(os.path.join(base_schema_uri, "Mapping"))
    format = 'turtle'  # Adjust the format as needed
    g = Graph()

    # Define custom prefixes
    rico_uri = 'https://www.ica.org/standards/RiC/ontology#'
    rico = ('rico', Namespace(rico_uri))
    ns = ('', Namespace(BASE_URI))
    auth = ('auth', Namespace(f"{base_auth_uri}/"))
    add = ('add', Namespace(f"{base_add_uri}/"))

    # Define common prefixes
    rdf = ('rdf', RDF)
    rdfs = ('rdfs', RDFS)
    owl = ('owl', OWL)

    total_count = 0
    def print_total_count(): print(f"\nNumber of triples in the graph: {total_count}")

    print("Initiating postprocessing...")
    try:    
        g.parse(graph_path,
                format=format,
                publicID=BASE_URI[:-1])
        total_count = len(g)
        print(f"Successfully read a graph from '{graph_path}'")
        print_total_count()
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

    if postprocessor_path:
        spec = importlib.util.spec_from_file_location("postprocessor", postprocessor_path)
        postprocessor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(postprocessor_module)
        g, has_changed = postprocessor_module.postprocess(g)
    else:
        has_changed = False

    return g, has_changed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map schema of choice")
    parser.add_argument("--config", help="Path to the configuration file.")
    parser.add_argument("--dataset", help="The name of the dataset to process.")
    parser.add_argument("rml_path", nargs='?', help="Optional path to an RML file")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset_config = config[args.dataset]
    postprocessor_path = dataset_config.get('postprocessor')

    rml_path, rmlmapper_path, ttl_path = map_rml(args.config, args.dataset, str(args.rml_path).lower())
    graph, has_changed = postprocess(ttl_path, postprocessor_path)

    def save_postprocessed_graph(
            output_format = 'nt', # assumed to be quickest
            ttl_path = ttl_path):
        # Serialize and print the RDF graph
        #output_format = 'ttl' # more lightweight and readable
        output_encoding = 'utf-8' # just to be sure
        ttl_filename = os.path.basename(ttl_path)
        postprocessed_filename = f'{ttl_filename[:-4]}_postprocessed.{output_format}'
        postprocessed_path = os.path.join(os.path.dirname(ttl_path), postprocessed_filename)
        #postprocessed_serialized = graph.serialize(format=output_format)
        # FYI, serialize returns:
        # bytes if destination is None and encoding is not None.
        # str if destination is None and encoding is None.
        #with open(postprocessed_path, 'w') as f:
        #    f.write(postprocessed_serialized)
        # Output to memory for speed
        postprocessed_serialized = BytesIO()
        graph.serialize(destination=postprocessed_serialized,
                        format=output_format,
                        encoding=output_encoding)
        # Save to a file from BytesIO
        with open(postprocessed_path, 'wb') as f: # Use 'wb' for binary write mode
            f.write(postprocessed_serialized.getvalue())
        print(f"\n\nSuccessfully saved postprocessed graph at: '{postprocessed_path}'")
        return postprocessed_serialized
    
    if has_changed:
        postprocessed_ttl_content = save_postprocessed_graph()
    #print(postprocessed_ttl_content)
