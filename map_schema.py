from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, DCTERMS
import pandas as pd
import re
import urllib.parse
from pprint import pprint
import os
import argparse
import glob
import requests

# Set labels for reference fields
auth_heading_label = 'HEADING'
add_refd_label = 'REFD'
add_ref_add_label = 'REF_ADD'
add_ref_file_label = 'REF_FILE'
add_title_label = 'TITLE'
private_mnemonics = ['ARCHAU', 'CMTAU']
rico_version_mask = r'{RICO_VERSION}'

triplesmap_label = 'TriplesMap'
uriref_str_label = 'uriref_str'
map_predicate_label = 'map_predicate'
map_object_label = 'map_object'

# combine_turtle_files generated with Claude 3.5 Sonnet
# on 2024-08-29, with modifications
def add_suppl_triples(source_graph: Graph, root_folder, format="turtle"):
    formats = {
        'turtle': ['ttl', 'turtle'],
        'nt': ['nt'],
        'n3': ['n3'],
        'xml': ['rdf', 'owl', 'xml'],
        'json-ld': ['jsonld', 'json-ld'],
        'nquads': ['nq'],
        'trig': ['trig']
    }

    # Walk through the directory tree
    for folder_path, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Get the file extension
            file_ext = filename.split('.')[-1]

            # Iterate over formats and check if the extension matches
            for format_name, extensions in formats.items():
                if ((file_ext in extensions) & (format_name == format)):
                    file_path = os.path.join(folder_path, filename)
                    print(f"Adding a supplemental '{format_name}' file: '{file_path}'")
                    
                    # Parse the Turtle file and add its contents to the combined graph
                    source_graph.parse(file_path, format=format)

    return source_graph

def __init__(schema_code, source_filename=None):
    # Define GBAD schema ontology
    base_data_uri = 'https://data.archives.gov.on.ca'
    #base_gbad_uri = URIRef(f"{base_data_uri}/RiC-O_1-0-1")
    base_gbad_uri = base_data_uri
    base_schema_uri = URIRef(f"{base_data_uri}/Schema")
    #base_kb_uri = URIRef(f"{base_data_uri}/KB")
    base_auth_uri = URIRef(f"{base_schema_uri}/Authority")
    base_add_uri = URIRef(f"{base_schema_uri}/Description-Listings")
    base_mapping_uri = URIRef(f"{base_schema_uri}/Mapping")

    base_uri_prefix = f"{base_data_uri}/"
    schema_term = 'Schema'
    auth_term = 'Authority'
    add_term = 'Description-Listings'
    maps_term = 'Mapping'
    kb_term = 'KB'

    # Any supported schema namespaces
    schema_regex_str = rf'^({auth_term}|{add_term}|{maps_term})/.*'
    schema_regex = re.compile(schema_regex_str, flags=re.IGNORECASE)

    # Any mnemonic-based URIs in GBAD URI syntax
    mnemonic_pattern = r"\{([A-Z:_\d\.]+)\}"
    mnemonic_regex = re.compile(rf"({mnemonic_pattern})/([a-zA-Z]+)(/\d+)?")
    # Pattern to capture within-mnemonic iterators
    mnemonic_i_pattern = r"(\d+)\.\.(\d+)"
    mnemonic_i_regex = re.compile(mnemonic_i_pattern)

    # This intends to support any RiC-O versions, past and future
    semver_pattern = r'(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?'
    def gbadify_rico_version(semver_str): return 'RiC-O_' + semver_str.replace('.', '-')
    # The commented below are useful to recognize any RiC-O version mask in GBAD URIs
    #def gbadify_rico_pattern(semver_pattern): return 'RiC-O_' + semver_pattern.replace(r'\.', '-')
    #gbad_term_pattern = gbadify_rico_pattern(semver_pattern)

    # Logic for getting the current version
    rico_uri = 'https://www.ica.org/standards/RiC/ontology#'
    def get_rico_version():
        # Try to request the OWL file using content negotiation
        headers = {'Accept': 'application/xml'}
        response = requests.get(rico_uri, headers=headers)
        # Check if we received RDF/XML content
        try:
            rico_graph = Graph()
            rico_graph.parse(data=response.text, format="xml")
            query = f"""
            SELECT ?versionIRI WHERE {{
                ?s <{OWL.versionIRI}> ?versionIRI .
            }}
            """
            for row in rico_graph.query(query):
                pattern = re.compile(rf'\/({semver_pattern})$')
                match = pattern.search(row.versionIRI)
                return match.group(1)
        except:
            pass

        return None
    try:
        gbad_term = gbadify_rico_version(get_rico_version())
    except:
        exit(f"Exiting. Fatal error: Could not resolve RiC-O version from '{rico_uri}'")
    def substitute_rico_version_mask(s): return str(s).replace(rico_version_mask, gbad_term) if str(s).startswith(rico_version_mask) else str(s)

    def prettify_rdfs_label(literal_str):
        # Remove base data prefix
        if literal_str.startswith(base_uri_prefix):
            literal_str = str(literal_str[len(base_uri_prefix):])

        # Schema entities
        if literal_str.lower().startswith(schema_term.lower() + '/'):
            literal_str = str(literal_str[len(schema_term)+1:])
            match = schema_regex.search(literal_str)
            if match:
                literal_str = match.group(0)
                #schema_group = match.group(1)
                #literal_str = str(literal_str[len(schema_group)+1:])
                #literal_str = literal_str + f' ({schema_group} Schema Entity)'

        # KB entities
        if literal_str.lower().startswith(kb_term.lower() + '/'):
            literal_str = str(literal_str[len(kb_term)+1:])
            literal_str = literal_str + ' (Knowledge Base Entity'
            match = re.search(mnemonic_pattern, literal_str)
            if match:
                mnemonic = match.group(1)
                literal_str = f'{mnemonic}'
                #literal_str = literal_str + f' from "{mnemonic}"'
            #literal_str = literal_str + ')'

        # GBAD entities
        if literal_str.startswith(rico_version_mask):
            literal_str = str(literal_str[len(rico_version_mask)+1:])
            match = mnemonic_regex.match(literal_str)
            if match:
                mnemonic_group = match.group(1) # in curly brackets
                mnemonic  = match.group(2)
                rico_class = match.group(3)
                instance_number = match.group(4)
                #literal_str = f'{mnemonic_group} ({rico_class} Entity'
                #if instance_number:
                #    instance_number = instance_number[1:] # leading slash removed
                #    literal_str = literal_str + f' #{instance_number}'
                #literal_str = literal_str + f' from "{mnemonic}")'
                literal_str = f'{mnemonic_group}'
        
        return literal_str

    # Choose ontology to map
    base_uri = base_data_uri
    suppl_graph_dir = None

    # Set schema-specific params
    if schema_code == 'add':
        # Assume the first file found
        graph_dir = 'gbad/schema/description-listings'
        graph_path = glob.glob(os.path.join(graph_dir, "*.ttl"))[0]
        
        # ADD: Choose source CSV for mapping
        if source_filename is None:
            source_path = 'gbad/mapping/source/description_head_6.csv'
    
    elif schema_code == 'auth':
        #suppl_graph_dir = 'gbad/schema/authority_AgentControlRelation'
        # Assume the first file found
        graph_dir = 'gbad/schema/authority/'
        graph_path = glob.glob(os.path.join(graph_dir, "*.ttl"))[0]

        # Authority: Choose source CSV for mapping
        if source_filename is None:
            source_path = 'gbad/mapping/source/authority_head_6.csv'

    else:
        raise Exception(f"Fatal error: Schema code not supplied.")

    if source_filename:
        source_path = f'gbad/mapping/source/{source_filename}.csv'

    rml_path = graph_path[:-3]+ "rml"

    # Create the input RDF graph
    g = Graph(base = base_uri)
    g.parse(graph_path,
            format="turtle")  # Adjust the format as needed

    # Add additional triples
    if suppl_graph_dir:
        g = add_suppl_triples(g, suppl_graph_dir, format="turtle")

    # Define custom prefixes
    rico = ('rico', Namespace(rico_uri))
    ns = ('data', Namespace(URIRef(f"{base_uri}/")))

    # Define common prefixes
    rdf = ('rdf', RDF)
    rdfs = ('rdfs', RDFS)
    owl = ('owl', OWL)

    # Define RML-specific prefixes
    rml = ('rml', Namespace('http://semweb.mmlab.be/ns/rml#'))
    rr = ('rr', Namespace('http://www.w3.org/ns/r2rml#'))
    ql = ('ql', Namespace('http://semweb.mmlab.be/ns/ql#'))
    csvw = ('csvw', Namespace('http://www.w3.org/ns/csvw#'))

    # Bind prefixes to namespaces
    g.namespace_manager.bind(*rico)
    g.namespace_manager.bind(*rdf)
    g.namespace_manager.bind(*rdfs)
    g.namespace_manager.bind(*owl)
    g.namespace_manager.bind(*ns)
    g.namespace_manager.bind(*rml)
    g.namespace_manager.bind(*rr)
    g.namespace_manager.bind(*ql)
    g.namespace_manager.bind(*csvw)
    g.namespace_manager.bind(*csvw)

    #print(g.serialize(format='turtle'))

    # Query to get all subjects, predicates, and objects
    query = f"""
    SELECT ?subject ?predicate ?object
    WHERE {{
    ?subject ?predicate ?object.
    }}
    """
    # Execute the query
    result = g.query(query)

    # List to hold the parsed results
    parsed_results = []

    # Process the results and create new triples
    for row in result:
        subject = row.subject
        predicate = row.predicate
        object = row.object

        parsed_results.append({
            'subject': subject,
            'predicate': predicate,
            'object': object
        })
        
    #print(parsed_results[:5]) # debug

    # Convert the parsed results to a dataframe
    parsed_df = pd.DataFrame(parsed_results)

    def normalize_uri(uri, ns_manager):
        if isinstance(uri, URIRef):
            return ns_manager.normalizeUri(uri)
        return None

    # SELECT ?s a ?o
    subjects_df = parsed_df[
        (parsed_df['predicate'].apply(lambda x: str(normalize_uri(x, g.namespace_manager))) == 'rdf:type') &
        (parsed_df['object'].apply(lambda x: str(normalize_uri(x, g.namespace_manager)).startswith(f"{rico[0]}:")))
    ].loc[:,['subject','object']]

    def extract_uriref_str(uriref):
        norm_uri = normalize_uri(uriref, g.namespace_manager)
        if not norm_uri:
            #map_series = uriref_str_to_map(uriref)
            #if map_series[map_predicate_label]:
                # This is a tricky part but really important because
                # otherwise nodes that are drawn as non-class nodes
                # are simply dropped. So this part tries to process
                # the "uriref" (which is really a literal in this case)
                # to produce a map, and if successful, that means that
                # input uriref is already uriref_str, so we are returning it.
            # Sorry, this is even simpler! Any input uriref which is not norm_uri
            # actually has to be returned as uriref because it means that it is
            # has to be passed on as a literal. By contrast, if only the if block
            # above is implemented, non-map series structures literals are dropped.
            # Thus, we are simply returning any literal as uriref.
            return uriref
            #return None
        # Replace namespace URIs with prefix codes
        uriref_str = str(norm_uri)
        # Remove base URI prefix
        uriref_str = uriref_str.replace(f"{ns[0]}:", '')
        # Decode special URI entities
        uriref_str = urllib.parse.unquote(uriref_str)
        return uriref_str
    
    def triplesmap_clean(str):
        # Replace with underscores anything but Latin letters, numbers, hyphens, and underscores
        triplesmap_name = re.sub(r'[^0-9a-z_-]', '_', str, flags=re.IGNORECASE)
        return triplesmap_name

    def generate_triplesmap_name(row):
        # This implementation assumes that subject URIs are unique
        subject_str = row[uriref_str_label]
        cleaned_subject = triplesmap_clean(subject_str)
        return cleaned_subject
    
    # Necessary to init namespace manager for uriref_str_to_map
    # Initialize an RDF graph
    mapping = Graph(base = URIRef(f"{base_gbad_uri}/"))
    
    def uriref_str_to_map(uriref_str):
        map_predicate = None
        map_object = None

        def series(map_predicate, map_object):
            map_series = pd.Series({
                map_predicate_label: map_predicate,
                map_object_label: map_object
            })
            return map_series

        if not uriref_str:
            return series(map_predicate, map_object)
        
        uriref_str = re.sub('\s+', ' ', uriref_str)

        def remove(predicate: URIRef, uriref_str):
            sin_predicate = re.sub(f"^{str(predicate)}\s+", "", uriref_str)
            sin_predicate = sin_predicate.strip('"')
            return sin_predicate
        
        def norm(uriref):
            return str(normalize_uri(uriref, g.namespace_manager))
        
        # Literal mapped from source
        if uriref_str.startswith(norm(rml[1].reference)):
            map_predicate = rml[1].reference
            map_object = Literal(remove(norm(map_predicate), uriref_str))
        # URI mapped from source
        elif uriref_str.startswith(norm(rr[1].template)):
            map_predicate = rr[1].template
            cleaned_uri = remove(norm(map_predicate), uriref_str)
            encoded_uri = URIRef(urllib.parse.quote(cleaned_uri, safe=''))
            if isinstance(cleaned_uri, URIRef): # check if true URI or rr:template
                map_object = URIRef(encoded_uri)
            else:
                cleaned_uri = substitute_rico_version_mask(cleaned_uri)
                map_object = Literal(cleaned_uri)
        # Constant URI
        elif uriref_str.startswith(norm(rr[1].constant)):
            map_predicate = rr[1].constant
            cleaned_uri = remove(norm(map_predicate), uriref_str)
            encoded_uri = URIRef(urllib.parse.quote(cleaned_uri, safe=''))
            if isinstance(cleaned_uri, URIRef): # check if true URI or constant literal
                map_object = URIRef(encoded_uri)
            else:
                map_object = Literal(cleaned_uri)
        # Treat anything else as a literal
        else:
            map_object = Literal(uriref_str)

        return series(map_predicate, map_object)

    def generate_rico_name(row):
        object_uri = row['object']
        object_str = str(normalize_uri(object_uri, g.namespace_manager))
        cleaned_object = object_str
        return cleaned_object
    
    def extract_mnemonic(row):
        map_predicate = row[map_predicate_label]
        map_object = row[map_object_label]
        if map_object:
            if map_predicate == rml[1].reference:
                return map_object
            elif map_predicate != rr[1].template:
                return None
            # Consider replacing this with more robust, findall logic
            # later on to allow for true multiple masks
            map_object = substitute_rico_version_mask(map_object)
            matches = re.findall(mnemonic_pattern, map_object)
            if matches:
                if len(matches) > 1:
                    print("At most one rr:template is allowed per subject map ",
                          f"whereas multiple are given in: '{map_object}'")
                    return None
                return matches[0]
        return None
    
    rico_name_label = 'RiC-O Name'.replace(' ','_')
    mnemonic_label = 'Authority Mnemonic'.replace(' ','_')

    # Note for next line that it is the only one that applies to series, all other to df
    subjects_df[uriref_str_label] = subjects_df['subject'].apply(extract_uriref_str)
    subjects_df[triplesmap_label] = subjects_df.apply(generate_triplesmap_name, axis=1)
    subjects_df[rico_name_label] = subjects_df.apply(generate_rico_name, axis=1)
    # Well, and the next one is also series only because uriref_str_to_map can then be reused outside of apply context
    subjects_df[[map_predicate_label, map_object_label]] = subjects_df[uriref_str_label].apply(uriref_str_to_map)
    subjects_df[mnemonic_label] = subjects_df.apply(extract_mnemonic, axis=1)
    subjects_df.drop(['object', uriref_str_label], axis=1, inplace=True)

    # Convert preprocessed DataFrame to HTML
    #from IPython.display import display, HTML, Markdown
    #sorted_columns = [triplesmap_label, rico_name_label, map_predicate_label, map_object_label, mnemonic_label, 'subject']
    #display_table = subjects_df[subjects_df[map_predicate_label].notnull()][sorted_columns].head(10).sort_values(by=triplesmap_label, ascending=True)
    #html_table = display_table.to_html(index=False) # for Jupyter Notebook
    #display(HTML(html_table)) # for Jupyter Notebook
    #print("\n\nSubjects Dataframe Preview:")
    #subjects_df.info()
    #print("\n", "\n\n".join([str(display_table.iloc[i]) for i in range(len(display_table))])) # debug

    # Add useful columns from subjects dataset for matching within loop later
    # The column name stays unique so we should just remember that RiC-O name refers to subject
    parsed_df = pd.merge(parsed_df, subjects_df[['subject', rico_name_label, triplesmap_label]], on='subject', how='left')

    # Also extract map predicates and objects for each object
    # Note that the below are for object, not subject, even though columns are called the same
    # Also note for next line that it is the only one that applies to series, all other to df
    parsed_df[uriref_str_label] = parsed_df['object'].apply(extract_uriref_str)
    # Well, and the next one is also series only because uriref_str_to_map can then be reused outside of apply context
    parsed_df[[map_predicate_label, map_object_label]] = parsed_df[uriref_str_label].apply(uriref_str_to_map)
    parsed_df[mnemonic_label] = parsed_df.apply(extract_mnemonic, axis=1)
    parsed_df.drop(uriref_str_label, axis=1, inplace=True)

    # Sort and only show those that have a predicate
    #display_table = parsed_df[parsed_df[uriref_str_label].notnull()].head(10).sort_values(by=triplesmap_label, ascending=True)
    #print("\n\nAll Triples Dataframe Preview:")
    #parsed_df.info()
    #print("\n", "\n\n".join([str(display_table.iloc[i]) for i in range(len(display_table))])) # debug

    # Define blank nodes and triples
    #agent_name_map = BNode()
    #agent_map = BNode()

    # Triples for :AgentNameAUTH13
    #mapping.add((maps[1].AgentNameAUTH13, RDF.type, rr[1].TriplesMap))

    # Additional RML masks
    iterator_mask = r'1'

    # This function is unused as long as URIs are hardcoded in draw.io graphs
    def construct_uri_mask(subjects_df, i):
        global mnemonic_label
        mnemonics = subjects_df.loc[:, mnemonic_label].tolist()
        subject_row = subjects_df.loc[i, :]
        rico_class = subject_row[rico_name_label][5:]
        #uri_mask = subject_row[uri_mask_label]
        try:
            if base_uri == base_auth_uri: # An Authority source
                if auth_heading_label in mnemonics:
                    uri_mask = f'{{{auth_heading_label}}}/{rico_class}/{iterator_mask}'
            elif base_uri == base_add_uri:
                if add_ref_add_label in mnemonics: # We have a LISTINGS source
                    if add_ref_file_label in mnemonics:
                        uri_mask = f'{{{add_ref_add_label}}}/{{{add_ref_file_label}}}/{rico_class}/{iterator_mask}'
                    else: # No Listings-level reference code
                        uri_mask = f'{{{add_ref_add_label}}}/{{{add_title_label}}}/{rico_class}/{iterator_mask}'
                elif add_refd_label in mnemonics: # We have a DESCRIPTION source
                    uri_mask = f'{{{add_refd_label}}}/{rico_class}/{iterator_mask}'
            
            return uri_mask
        except UnboundLocalError:
            print(f'No valid identifiers found for an ADD source:\n{subject_row}')
            return None
    
    # Initialize a mapping RDF graph
    mapping = Graph(base = URIRef(f"{base_gbad_uri}/"))
    
    # Define custom prefix
    maps = ('', Namespace(URIRef(f"{base_mapping_uri}#")))

    # Bind prefixes to namespaces
    mapping.namespace_manager.bind(*rico)
    mapping.namespace_manager.bind(*rdf)
    mapping.namespace_manager.bind(*rdfs)
    mapping.namespace_manager.bind(*owl)
    mapping.namespace_manager.bind(*rml)
    mapping.namespace_manager.bind(*rr)
    mapping.namespace_manager.bind(*ql)
    mapping.namespace_manager.bind(*csvw)
    mapping.namespace_manager.bind(*maps)

    def get_mnemonic_i_from_to(mnemonic):
        mnemonic_i_from, mnemonic_i_to = 1, 1
        if mnemonic:
            mnemonic_i_matches = mnemonic_i_regex.findall(mnemonic)
            mnemonic_i_match_count = len(mnemonic_i_matches)
            if mnemonic_i_match_count == 0:
                pass
            elif mnemonic_i_match_count > 1:
                raise Exception(f"Error while handling '{mnemonic}' mnemonic: ",
                                f"{mnemonic_i_match_count} increment requests detected while max one allowed.")
            else:
                mnemonic_i_from, mnemonic_i_to = int(mnemonic_i_matches[0][0]), int(mnemonic_i_matches[0][1])
        return mnemonic_i_from, mnemonic_i_to
    
    # Construct RML graph
    for i, subject_row in subjects_df.iterrows():
        # This refers to the original subject URI from drawio graph
        # which is being used to uniquely identify subject
        subject_uri = subject_row['subject']
        subject_mnemonic = subject_row[mnemonic_label]

        # Skip private fields removed from input data
        if subject_mnemonic in private_mnemonics:
            continue

        # Define TriplesMap
        triples_map = maps[1][subject_row[triplesmap_label]]
        mapping.add((triples_map, RDF.type, rr[1].TriplesMap))

        # Collect subjectmap predicate and object from subject df
        # These will be added to the graph and then used later on
        subject_map_predicate = subject_row[map_predicate_label]
        uri_mask = subject_row[map_object_label]
        #URIRef(urllib.parse.unquote(str(subject)))
        #uri_mask = construct_uri_mask(subjects_df, i)
        
        # Define an empty Subject Map
        subject_map = BNode()
        mapping.add((triples_map, rr[1].subjectMap, subject_map))

        # Remove prefix from RiC-O name from subject df and add to graph
        rico_name = subject_row[rico_name_label]
        rico_class = rico_name[5:]
        # So this adds the rdf:type definition
        mapping.add((subject_map, rr[1]['class'], rico[1][rico_class]))

        # If no valid RML definitions in the graph
        if not subject_map_predicate:
            #if isinstance(triples_map, BNode):
            #    # Means that 
            #    continue
            # Replace the blank node with subject as literal
            # Well, this is not really a subject "uri" in this case
            # or shouldn't be because URIs have to be set up via rr:constant
            if subject_uri: # not sure if it is at all possible for this to be null
                mapping.add((subject_map, rr[1].constant, Literal(subject_uri)))
            continue # because cannot move forward with map predicate undefined
            # Also note that rr:subject is incompatible with logical source

        # Define Logical Source
        logical_source = BNode()
        mapping.add((triples_map, rml[1].logicalSource, logical_source))
        mapping.add((logical_source, rml[1].source, Literal(source_path)))
        mapping.add((logical_source, rml[1].referenceFormulation, ql[1].CSV))
        #mapping.add((logical_source, rml[1].iterator, Literal(iterator_mask)))

        # Add map predicate and object from df to subject map
        mapping.add((subject_map, subject_map_predicate, uri_mask))

        # Deal with predicates and objects in full triples df
        # Subset triples with the subject and RiC-O class from i-loop
        objectmap_df = parsed_df[(
            (parsed_df['subject']==subject_uri) &
            (parsed_df[rico_name_label] == rico_name)
        )]
        for k, parsed_result in objectmap_df.iterrows():
            # Only focus on RiC-O or RDFS predicates
            predicate = parsed_result['predicate']
            norm_predicate = normalize_uri(predicate, mapping.namespace_manager)
            is_rico = (norm_predicate.startswith(f"{rico[0]}:"))
            is_rdfs = (norm_predicate.startswith(f"{rdfs[0]}:"))
            if (is_rico | is_rdfs):
                # Now we can actually iterate over objects
                object = parsed_result['object']
                object_map_predicate = parsed_result[map_predicate_label]
                object_map_object = parsed_result[map_object_label]
                object_mnemonic = parsed_result[mnemonic_label]

                # Handle possible increment requests in object mnemonic
                object_mnemonic_i_from, object_mnemonic_i_to = get_mnemonic_i_from_to(object_mnemonic)
                for object_mnemonic_i in range(object_mnemonic_i_from, object_mnemonic_i_to + 1):
                    # Define a predicate-object map
                    predicate_object_map = BNode()
                    mapping.add((triples_map, rr[1].predicateObjectMap, predicate_object_map))

                    # Add predicate to predicate-object map
                    mapping.add((predicate_object_map, rr[1].predicate, URIRef(predicate)))

                    # Define an empty object map within the predicate-object map
                    object_map = BNode()
                    mapping.add((predicate_object_map, rr[1].objectMap, object_map))

                    # If not RiC-O, then nothing applies and just attach as literal
                    # In the current version of drawio parser only rdfs:label is supported
                    # and such, so this is essential to bypass these. However, I am not
                    # sure at this point how well this would work if other namespaces
                    # were fully supported by drawio parser.
                    if not is_rico: # any other namespace
                        if norm_predicate == 'rdfs:label': # handle labels from drawio parser
                            mapping.add((object_map, rr[1].termType, rr[1].Literal)) # print as literal
                            # Only using object_map_object here because object_map_predicate is irrelevant
                            pretty_omo = prettify_rdfs_label(object_map_object)
                            mapping.add((object_map, rr[1].template, Literal(pretty_omo)))
                            continue
                        
                        mapping.add((object_map, rr[1].constant, Literal(object))) # point to constant URI
                        continue

                    # This concerns only constant literals, meaning nodes
                    # in drawio graph for which no mapping logic is defined
                    if not object_map_predicate:
                        # So these are simply added as predicate and object, no predicate-object map
                        if object_map_object: # sometimes it may be empty
                            mapping.add((object_map, rr[1].constant, object_map_object)) 
                        continue
                    
                    # Now let's finally attach the object to the object map
                    # Case when the object is supposed to reference another Subject map
                    if object in set(subjects_df['subject']):
                        triplesmap = maps[1][subjects_df[subjects_df['subject']==object][triplesmap_label].iloc[0]]
                        mapping.add((object_map, rr[1].parentTriplesMap, triplesmap))
                        #join_condition = BNode()
                        #mapping.add((object_map, rr[1].joinCondition, join_condition))
                        #mnemonic = parsed_result[mnemonic_label]
                        #mapping.add((join_condition, rr[1].child, Literal(mnemonic)))
                        #mapping.add((join_condition, rr[1].parent, Literal(mnemonic)))
                    else:
                        object_mnemonic_ith = mnemonic_i_regex.sub(str(object_mnemonic_i), object_mnemonic) if object_mnemonic_i_to > 1 else object_mnemonic
                        if object_mnemonic_ith in private_mnemonics:
                            mapping.add((object_map, rr[1].constant, URIRef(f"censored#{object_mnemonic_ith}")))
                            continue
                        object_map_predicate = parsed_result[map_predicate_label]
                        object_map_object = parsed_result[map_object_label]
                        if object_map_object: # just in case user forgot to set it in drawio
                            # Logic to substitute increment request with an actual number for object
                            object_map_object_ith = mnemonic_i_regex.sub(str(object_mnemonic_i), str(object_map_object)) if object_mnemonic_i_to > 1 else str(object_map_object)
                            object_map_object_ith = URIRef(object_map_object_ith) if isinstance(object_map_object, URIRef) else Literal(object_map_object_ith)
                            mapping.add((object_map, object_map_predicate, object_map_object_ith))

    # Serialize and print the RDF graph
    ttl = mapping.serialize(format='turtle')
    with open(rml_path, 'w') as f:
        f.write(mapping.serialize(format='turtle'))
    print(f"\n\nSuccessfully saved RML map to: '{rml_path}'")
    #print(ttl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map schema of choice")
    parser.add_argument("schema", help="Choose one: add or auth.")
    parser.add_argument("source", nargs='?', help="Filename of source CSV without extension. Defaults to the head=6 version for chosen schema.")

    args = parser.parse_args()

    __init__(str(args.schema).lower(), args.source)
