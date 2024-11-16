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
#import uuid

# Prohibit trimming pd prints in shell
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)

# Set labels for reference fields
auth_heading_label = 'HEADING'
add_refd_label = 'REFD'
add_ref_add_label = 'REF_ADD'
add_ref_file_label = 'REF_FILE'
add_title_label = 'TITLE'
private_mnemonics = ['ARCHAU', 'CMTAU']
auth_authtp_label = 'AUTHTP'
rico_version_mask = r'{RICO_VERSION}'
rico_authtp_mask = r'{RICO_AUTHTP}'
rico_authtp_dict = {
    'CorporateBody': r'/(Corporate Name|[ABC] Ontario Government Name)/',
    'Family': r'/Family Name/',
    'Place': r'/Geographic Name/',
    'Person': r'/Personal Name/'
}
#uuid_label = 'UUID'

triplesmap_label = 'TriplesMap'
uriref_str_label = 'uriref_str'
map_predicate_label = 'map_predicate'
map_object_label = 'map_object'
has_increment_label = 'has_increment_request'

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
    #NAMESPACE_UUID = uuid.uuid5(uuid.NAMESPACE_URL, f"{base_gbad_uri}/")
    #print(f"Namespace UUID v5 for <{base_gbad_uri}/>: {NAMESPACE_UUID}")
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

    def get_second_term():
        if schema_code == 'auth':
            return auth_term
        elif schema_code == 'add':
            return add_term
        else:
            raise Exception(f"Fatal error: Schema code not supplied or supported.")

    # Any supported schema namespaces
    schema_regex_str = rf'^({auth_term}|{add_term}|{maps_term})/.*?/([^/]+)/?$'
    schema_regex = re.compile(schema_regex_str, flags=re.IGNORECASE)

    # Any mnemonic-based URIs in GBAD URI syntax
    mnemonic_pattern = r"\{([A-Z:_\d\.]+)\}"
    mnemonic_regex = re.compile(rf"([a-zA-Z]+)/({mnemonic_pattern})/?(.*)")
    # Pattern to capture within-mnemonic iterators
    mnemonic_i_pattern = r"(\d+)\.\.(\d+)"
    mnemonic_i_regex = re.compile(mnemonic_i_pattern)

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

    # This intends to support any RiC-O versions, past and future
    semver_pattern = r'(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?'
    def gbadify_rico_version(semver_str): return 'RiC-O_' + semver_str.replace('.', '-')
    # The commented below are useful to recognize any RiC-O version mask in GBAD URIs
    def gbadify_rico_pattern(semver_pattern): return 'RiC-O_' + semver_pattern.replace(r'\.', '-')
    gbad_term_pattern = gbadify_rico_pattern(semver_pattern)
    gbad_term_regex = re.compile(rf"^({gbad_term_pattern})/.+", re.IGNORECASE)

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
    
    ### Start block for downloading RiC-O version
    #try:
    #    gbad_term = gbadify_rico_version(get_rico_version())
    #except:
    #    exit(f"Exiting. Fatal error: Could not resolve RiC-O version from '{rico_uri}'")
    #def substitute_rico_version_mask(s): return str(s).replace(rico_version_mask, gbad_term) if str(s).startswith(rico_version_mask) else str(s)
    ### End block for downloading RiC-O version

    # The below works but commented out for now because did not do the trick without accessing input CSV values
    # That is, the UUID is only unique to the TriplesMap, so all entities generated from it have the same UUID
    # Define UUID replacement logic - support any position of number but only allowed chars
    #uuid_pattern = f"%7B({uuid_label}_?(\d*)|(\d*)_?{uuid_label})%7D" # using encoded because rr:constant will be used
    #uuid_regex = re.compile(uuid_pattern) # let's make it case-sensitive to enforce strictness for this special word
    #def substitute_uuid(uriref, entity_name):
    #    str(uriref)
    #    s = uuid_regex.sub(str(uuid.uuid5(NAMESPACE_UUID, entity_name)), str(uriref))
    #    return URIRef(s) if isinstance(uriref, URIRef) else Literal(s)

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
                last_term = match.group(2)
                literal_str = last_term
                #schema_group = match.group(1)
                #literal_str = str(literal_str[len(schema_group)+1:])
                #literal_str = literal_str + f' ({schema_group} Schema Entity)'

        # KB entities
        if literal_str.lower().startswith(kb_term.lower() + '/'):
            literal_str = str(literal_str[len(kb_term)+1:])
            #literal_str = literal_str + ' (Knowledge Base Entity'
            match = re.search(mnemonic_regex, literal_str)
            if match:
                rico_ish_class = match.group(1)
                mnemonic = match.group(3)
                optional_rest = match.group(4)
                literal_str = f'{{{mnemonic}}} ({rico_ish_class})'
                #literal_str = literal_str + f' from "{mnemonic}"'
            #literal_str = literal_str + ')'

        # GBAD entities - deprecated as of 2024-11-14 :(
        #gbad_term_match = gbad_term_regex.match(literal_str)
        #if gbad_term_match:
        #    matched_gbad_term = gbad_term_match.group(1)
        #    literal_str = str(literal_str[len(matched_gbad_term)+1:])
        #    match = mnemonic_regex.match(literal_str)
        #    if match:
        #        mnemonic_group = match.group(1) # in curly brackets
        #        mnemonic  = match.group(2)
        #        rico_class = match.group(3)
        #        instance_number = match.group(4)
                #literal_str = f'{mnemonic_group} ({rico_class} Entity'
                #if instance_number:
                #    instance_number = instance_number[1:] # leading slash removed
                #    literal_str = literal_str + f' #{instance_number}'
                #literal_str = literal_str + f' from "{mnemonic}")'
        #        literal_str = f'{{{mnemonic}}} ({rico_class})'
        
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
        source_path = f'gbad/mapping/source/{source_filename}'

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

    # Namespaces for FnO to work at RML mapping
    fnml = ('fnml', Namespace('http://semweb.mmlab.be/ns/fnml#'))
    fno = ('fno', Namespace('https://w3id.org/function/ontology#'))
    idlab_fn = ('idlab-fn', Namespace('https://w3id.org/imec/idlab/function#'))
    grel = ('grel', Namespace('http://users.ugent.be/~bjdmeest/function/grel.ttl#'))

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
    g.namespace_manager.bind(*fnml)
    g.namespace_manager.bind(*fno)
    g.namespace_manager.bind(*idlab_fn)
    g.namespace_manager.bind(*grel)

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

    rico_authtp_subjects = dict() # keeping these out for future use
    def disaggregate_rico_authtp(spo):
        subject_uri, predicate_uri, object_uri = spo
        rico_disaggregated_subjects = []
        rico_disaggregated_objects = []
        rico_disaggregated_triples = []

        # Necessary to make matches and replacements work
        rico_authtp_mask_encoded = urllib.parse.quote(rico_authtp_mask, safe='')
        subject_mask = rico_authtp_mask_encoded if isinstance(subject_uri, URIRef) else rico_authtp_mask
        object_mask = rico_authtp_mask_encoded if isinstance(object_uri, URIRef) else rico_authtp_mask

        # Replacing subject
        if subject_mask in str(subject_uri):
            for authtp_rico_class in rico_authtp_dict.keys():
                # If contains {RICO_AUTHTP}
                # Add two subjects for easy separate triplesmap creation later on
                for authtp_i in [1, 2]:
                    authtp_column_name = f"{auth_authtp_label}_{authtp_i}"
                    replacement = f"{authtp_rico_class}_{authtp_column_name}"
                    rico_disaggregated_subject_uri = str(subject_uri).replace(subject_mask, replacement)

                    if isinstance(subject_uri, URIRef):
                        rico_disaggregated_subject_uri = URIRef(rico_disaggregated_subject_uri)
                    else:
                        rico_disaggregated_subject_uri = Literal(rico_disaggregated_subject_uri)
                    rico_disaggregated_subjects.append(rico_disaggregated_subject_uri)
                    # Keep an external list of these for future use
                    if not rico_disaggregated_subject_uri in rico_authtp_subjects.keys():
                        true_rico_disaggregated_subject_uri = str(subject_uri).replace(subject_mask, authtp_rico_class)
                        if isinstance(subject_uri, URIRef):
                            true_rico_disaggregated_subject_uri = URIRef(true_rico_disaggregated_subject_uri)
                        else:
                            true_rico_disaggregated_subject_uri = Literal(true_rico_disaggregated_subject_uri)
                        rico_authtp_subjects[rico_disaggregated_subject_uri] = (true_rico_disaggregated_subject_uri, authtp_column_name)

                    # If is a triple like ?s a rico:Thing
                    if ((predicate_uri == rdf[1].type) and
                        (object_uri == rico[1].Thing)):
                        rico_disaggregated_object_uri = URIRef(str(object_uri).replace('Thing',
                                                                                        authtp_rico_class))
                        rico_disaggregated_triples.append((rico_disaggregated_subject_uri,
                                                        predicate_uri,
                                                        rico_disaggregated_object_uri))
        else:
            rico_disaggregated_subjects.append(subject_uri)

        if len(rico_disaggregated_triples) == 0: # if not a triple like ?s a rico:Thing
            # Replacing object
            if object_mask in str(object_uri):
                for authtp_rico_class in rico_authtp_dict.keys():
                    # If contains {RICO_AUTHTP}
                    # Add two subjects for easy separate triplesmap creation later on
                    for authtp_i in [1, 2]:
                        authtp_column_name = f"{auth_authtp_label}_{authtp_i}"
                        replacement = f"{authtp_rico_class}_{authtp_column_name}"
                        rico_disaggregated_object_uri = str(object_uri).replace(object_mask, replacement)

                        if isinstance(object_uri, URIRef):
                            rico_disaggregated_object_uri = URIRef(rico_disaggregated_object_uri)
                        else:
                            rico_disaggregated_object_uri = Literal(rico_disaggregated_object_uri)
                        rico_disaggregated_objects.append(rico_disaggregated_object_uri)
            elif len(rico_disaggregated_objects) == 0: 
                rico_disaggregated_objects.append(object_uri)
            
            # Collect all subjects and objects
            for rico_disaggregated_subject in rico_disaggregated_subjects:
                for rico_disaggregated_object in rico_disaggregated_objects:
                    rico_disaggregated_triples.append((rico_disaggregated_subject,
                                                    predicate_uri,
                                                    rico_disaggregated_object))

        return rico_disaggregated_triples

    # Process the results and create new triples
    for row in result:
        subject = row.subject
        predicate = row.predicate
        object = row.object

        rico_disaggregated_triples = disaggregate_rico_authtp((subject, predicate, object))
        for s, p, o in rico_disaggregated_triples:
            parsed_results.append({
                'subject': s,
                'predicate': p,
                'object': o
            })
        
    #print(parsed_results[:5]) # debug

    # Convert the parsed results to a dataframe
    parsed_df = pd.DataFrame(parsed_results)
    # Dropping duplicates is necessary due to the duplicated classes
    # in rico_authtp_dict as this duplicates #rr_template__KB_CorporateBody__HEADING__
    # thus producing an error at RML mapping
    #parsed_df = parsed_df.drop_duplicates()
    #parsed_df.to_csv('parsed_df.csv') # for debug

    def normalize_uri(uri, ns_manager):
        if isinstance(uri, URIRef):
            return ns_manager.normalizeUri(uri)
        return None

    # SELECT ?s a ?o
    subjects_df = parsed_df[
        (parsed_df['predicate'].apply(lambda x: str(normalize_uri(x, g.namespace_manager))) == 'rdf:type') &
        (parsed_df['object'].apply(lambda x: str(normalize_uri(x, g.namespace_manager)).startswith(f"{rico[0]}:")))
    ].loc[:,['subject','object']] # So to be sure, object is the rdf:type URI here

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
                #cleaned_uri = substitute_rico_version_mask(cleaned_uri)
                map_object = Literal(cleaned_uri)
        # Constant URI
        elif uriref_str.startswith(norm(rr[1].constant)):
            map_predicate = rr[1].constant
            cleaned_uri = remove(norm(map_predicate), uriref_str)
            encoded_uri = URIRef(urllib.parse.quote(cleaned_uri, safe=":/#?&="))
            if isinstance(encoded_uri, URIRef): # check if true URI or constant literal
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
        #triplesmap_name = row[triplesmap_label]
        if map_object:
            if map_predicate == rml[1].reference:
                return map_object
            elif map_predicate != rr[1].template:
                return None
            # Consider replacing this with more robust, findall logic
            # later on to allow for true multiple masks
            #map_object = substitute_rico_version_mask(map_object)
            #map_object = substitute_uuid_mask(map_object, triplesmap_name)
            matches = re.findall(mnemonic_pattern, map_object)
            if matches:
                if len(matches) > 1:
                    other_mnemonics = ", ".join([f"{{{match}}}" for match in matches[1:]])
                    print("At most one rr:template is allowed per subject map ",
                          f"whereas multiple are given in: '{map_object}'. ",
                          f"By default logic, the leftmost mnemonic is deliberately chosen as the main one.",
                          f"Thus, {{{matches[0]}}} will be processed as the main mnemonic, "
                          f"and all the others will be passed to RML as is: {other_mnemonics}", "\n")
                    #return None
                return matches[0]
        return None
    
    rico_name_label = 'RiC-O Name'.replace(' ','_')
    mnemonic_label = 'Authority Mnemonic'.replace(' ','_')

    def collect_incremented_uri(row, column, disaggregated_series_list):
        #row_id = row.name
        mnemonic = row[mnemonic_label]
        column_uri = row[column]
        # Uncomment the below if want to allow increments outside of mnemonics
        #mnemonic_i_from, mnemonic_i_to = get_mnemonic_i_from_to(column_uri)
        mnemonic_i_from, mnemonic_i_to = get_mnemonic_i_from_to(mnemonic)
        row[f'original_{column}'] = row[column]
        for mnemonic_i in range(mnemonic_i_from, mnemonic_i_to + 1):
            new_row = row.copy()
            column_value = mnemonic_i_regex.sub(str(mnemonic_i), str(column_uri))
            if column_value:
                new_row[column] = URIRef(column_value) if isinstance(column_uri, URIRef) else Literal(column_value) 
            disaggregated_series_list.append(new_row)
        return row
    
    disaggregated_subject_rows = []
    def collect_incremented_subject_uri(row): return collect_incremented_uri(row, 'subject', disaggregated_subject_rows)

    # Note for next line that it is the only one that applies to series, all other to df
    subjects_df[uriref_str_label] = subjects_df['subject'].apply(extract_uriref_str)
    # Well, and the next one is also series only because uriref_str_to_map can then be reused outside of apply context
    subjects_df[[map_predicate_label, map_object_label]] = subjects_df[uriref_str_label].apply(uriref_str_to_map)
    subjects_df[mnemonic_label] = subjects_df.apply(extract_mnemonic, axis=1)\
    
    # Now that we have mnemonics generated, let's honor any increment requests
    disaggregated_subject_rows = []
    def collect_incremented_subject_uri(row): return collect_incremented_uri(row, 'subject', disaggregated_subject_rows)
    subjects_df = subjects_df.apply(collect_incremented_subject_uri, axis=1)
    # Creating new frame so that there is no duplication wih previous
    subjects_df = pd.DataFrame(disaggregated_subject_rows)
    # Let's regenerate cols above for simplicity now that rows are disaggregated
    subjects_df[uriref_str_label] = subjects_df['subject'].apply(extract_uriref_str)
    subjects_df[[map_predicate_label, map_object_label]] = subjects_df[uriref_str_label].apply(uriref_str_to_map)
    subjects_df[mnemonic_label] = subjects_df.apply(extract_mnemonic, axis=1)
    # Now that all cols have been disaggregated, let's generate remaining useful cols
    subjects_df[triplesmap_label] = subjects_df.apply(generate_triplesmap_name, axis=1)
    subjects_df[rico_name_label] = subjects_df.apply(generate_rico_name, axis=1)
    # Let's drop the object (i.e., rdf:type) because it's now in rico_name_label
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
    # The line below is really important, or triples will be lost!
    parsed_df = parsed_df.rename(columns={'subject': 'original_subject'}) 
    parsed_df = pd.merge(parsed_df, subjects_df[['original_subject', 'subject', rico_name_label, triplesmap_label]], on='original_subject', how='left')
    # Also extract map predicates and objects for each object
    # Note that the below are for object, not subject, even though columns are called the same
    # Also note for next line that it is the only one that applies to series, all other to df
    parsed_df[uriref_str_label] = parsed_df['object'].apply(extract_uriref_str)
    # Well, and the next one is also series only because uriref_str_to_map can then be reused outside of apply context
    parsed_df[[map_predicate_label, map_object_label]] = parsed_df[uriref_str_label].apply(uriref_str_to_map)
    parsed_df[mnemonic_label] = parsed_df.apply(extract_mnemonic, axis=1)

    # Now that we have mnemonics generated, let's honor any increment requests
    disaggregated_object_rows = []
    def collect_incremented_object_uri(row): return collect_incremented_uri(row, 'object', disaggregated_object_rows)
    parsed_df = parsed_df.apply(collect_incremented_object_uri, axis=1)
    # Creating new frame so that there is no duplication wih previous
    parsed_df = pd.DataFrame(disaggregated_object_rows)
    # Let's regenerate cols above for simplicity now that rows are disaggregated
    parsed_df[uriref_str_label] = parsed_df['object'].apply(extract_uriref_str)
    parsed_df[[map_predicate_label, map_object_label]] = parsed_df[uriref_str_label].apply(uriref_str_to_map)
    parsed_df[mnemonic_label] = parsed_df.apply(extract_mnemonic, axis=1)

    # Now that all cols have been disaggregated, drop the temporary field
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
    mapping.namespace_manager.bind(*fnml)
    mapping.namespace_manager.bind(*fno)
    mapping.namespace_manager.bind(*idlab_fn)
    mapping.namespace_manager.bind(*grel)

    def fno_map_value_unless_isnull(
            rml_g,
            return_tuple, input_tuples):
        return_predicate, return_object = return_tuple

        # Define a wrapper function
        fno_wrapper = BNode()

        # Use the controls_if function to conditionally map based on non-empty value
        controls_if_pomap = BNode()
        rml_g.add((fno_wrapper, rr[1].predicateObjectMap, controls_if_pomap))
        rml_g.add((controls_if_pomap, rr[1].predicate, fno[1].executes))
        controls_if_omap = BNode()
        rml_g.add((controls_if_pomap, rr[1].objectMap, controls_if_omap))
        rml_g.add((controls_if_omap, rr[1].constant, grel[1].controls_if))

        # Define the arguments for the if condition
        # First argument: Check if input value is empty
        mnemonic_isnull_pomap = BNode()
        rml_g.add((fno_wrapper, rr[1].predicateObjectMap, mnemonic_isnull_pomap))
        rml_g.add((mnemonic_isnull_pomap, rr[1].predicate, grel[1].bool_b))
        mnemonic_isnull_omap = BNode()
        rml_g.add((mnemonic_isnull_pomap, rr[1].objectMap, mnemonic_isnull_omap))
        # A nested function
        nested_fno_logic = BNode()
        rml_g.add((mnemonic_isnull_omap, fnml[1].functionValue, nested_fno_logic))
        # Nested function definition
        mnemonic_isnull_nested_def_pomap = BNode()
        rml_g.add((nested_fno_logic, rr[1].predicateObjectMap, mnemonic_isnull_nested_def_pomap))
        rml_g.add((mnemonic_isnull_nested_def_pomap, rr[1].predicate, fno[1].executes))
        mnemonic_isnull_nested_def_omap = BNode()
        rml_g.add((mnemonic_isnull_nested_def_pomap, rr[1].objectMap, mnemonic_isnull_nested_def_omap))
        rml_g.add((mnemonic_isnull_nested_def_omap, rr[1].constant, idlab_fn[1].isNull))
        # Nested function argument
        mnemonic_isnull_nested_arg_pomap = BNode()
        rml_g.add((nested_fno_logic, rr[1].predicateObjectMap, mnemonic_isnull_nested_arg_pomap))
        rml_g.add((mnemonic_isnull_nested_arg_pomap, rr[1].predicate, idlab_fn[1].str))
        mnemonic_isnull_nested_arg_omap = BNode()
        rml_g.add((mnemonic_isnull_nested_arg_pomap, rr[1].objectMap, mnemonic_isnull_nested_arg_omap))
        # Here goes the climax of checking - the input value
        for input_tuple in input_tuples:
            rml_g.add((mnemonic_isnull_nested_arg_omap, input_tuple[0], input_tuple[1]))

        # If the input value is not null, use the value in the object map
        mnemonic_uri_mask_pomap = BNode()
        rml_g.add((fno_wrapper, rr[1].predicateObjectMap, mnemonic_uri_mask_pomap))
        rml_g.add((mnemonic_uri_mask_pomap, rr[1].predicate, grel[1].any_false))
        mnemonic_uri_mask_omap = BNode()
        rml_g.add((mnemonic_uri_mask_pomap, rr[1].objectMap, mnemonic_uri_mask_omap))
        # Here goes the climax of writing - the return value
        rml_g.add((mnemonic_uri_mask_omap, return_predicate, return_object))
        
        return fno_wrapper
    
    def fno_map_this_string_match(
            rml_g,
            return_tuple, input_value_tuples, regex_tuples):
        # Define a wrapper function
        nested_fno_wrapper = BNode()
        # Nested function definition
        nested_def_pomap = BNode()
        rml_g.add((nested_fno_wrapper, rr[1].predicateObjectMap, nested_def_pomap))
        rml_g.add((nested_def_pomap, rr[1].predicate, fno[1].executes))
        nested_def_omap = BNode()
        rml_g.add((nested_def_pomap, rr[1].objectMap, nested_def_omap))
        rml_g.add((nested_def_omap, rr[1].constant, grel[1].string_match))

        # Nested function argument 1
        nested_fun_arg_1_pomap = BNode()
        rml_g.add((nested_fno_wrapper, rr[1].predicateObjectMap, nested_fun_arg_1_pomap))
        rml_g.add((nested_fun_arg_1_pomap, rr[1].predicate, grel[1].valueParameter))
        nested_fun_arg_1_omap = BNode()
        rml_g.add((nested_fun_arg_1_pomap, rr[1].objectMap, nested_fun_arg_1_omap))
        # Here goes the climax of checking - the input_value_1
        for input_value_tuple_1 in input_value_tuples:
            # Define a wrapper function
            nested_tostring_fno_wrapper = BNode()
            # Nested function definition
            nested_tostring_def_pomap = BNode()
            rml_g.add((nested_tostring_fno_wrapper, rr[1].predicateObjectMap, nested_tostring_def_pomap))
            rml_g.add((nested_tostring_def_pomap, rr[1].predicate, fno[1].executes))
            nested_tostring_def_omap = BNode()
            rml_g.add((nested_tostring_def_pomap, rr[1].objectMap, nested_tostring_def_omap))
            rml_g.add((nested_tostring_def_omap, rr[1].constant, grel[1].string_toString))
            # Nested function argument 1
            nested_tostring_fun_arg_1_pomap = BNode()
            rml_g.add((nested_tostring_fno_wrapper, rr[1].predicateObjectMap, nested_tostring_fun_arg_1_pomap))
            rml_g.add((nested_tostring_fun_arg_1_pomap, rr[1].predicate, grel[1].p_any_e))
            nested_tostring_fun_arg_1_omap = BNode()
            rml_g.add((nested_tostring_fun_arg_1_pomap, rr[1].objectMap, nested_tostring_fun_arg_1_omap))
            rml_g.add((nested_tostring_fun_arg_1_omap, input_value_tuple_1[0], input_value_tuple_1[1]))
            rml_g.add((nested_fun_arg_1_omap, fnml[1].functionValue, nested_tostring_fno_wrapper))
        
        # Nested function argument 2
        nested_fun_arg_2_pomap = BNode()
        rml_g.add((nested_fno_wrapper, rr[1].predicateObjectMap, nested_fun_arg_2_pomap))
        rml_g.add((nested_fun_arg_2_pomap, rr[1].predicate, grel[1].p_string_regex))
        nested_fun_arg_2_omap = BNode()
        rml_g.add((nested_fun_arg_2_pomap, rr[1].objectMap, nested_fun_arg_2_omap))
        # Here goes the climax of checking - the input_value_2
        for input_value_tuple_2 in regex_tuples:
            rml_g.add((nested_fun_arg_2_omap, input_value_tuple_2[0], input_value_tuple_2[1]))

        # Convert string match output to boolean
        input_tuples = [(fnml[1].functionValue, nested_fno_wrapper)]
        fno_wrapper = fno_map_value_unless_isnull(
            rml_g = rml_g,
            return_tuple = return_tuple,
            input_tuples = input_tuples)
        
        return fno_wrapper

    def add_custom_triple_to_triplesmap(predicate_uri, object_var, triples_map):
        # Define a predicate-object map
        predicate_object_map = BNode()
        mapping.add((triples_map, rr[1].predicateObjectMap, predicate_object_map))
        # Add predicate to predicate-object map
        mapping.add((predicate_object_map, rr[1].predicate, URIRef(predicate_uri)))
        # Define an empty object map within the predicate-object map
        object_map = BNode()
        mapping.add((predicate_object_map, rr[1].objectMap, object_map))
        # Add object to object map
        object = object_var if isinstance(object_var, URIRef) else Literal(object_var)
        mapping.add((object_map, rr[1].template, object))
        return None
    
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
        triplesmap_name = subject_row[triplesmap_label]
        triples_map = maps[1][triplesmap_name]
        mapping.add((triples_map, RDF.type, rr[1].TriplesMap))

        # Define Logical Source
        logical_source = BNode()
        mapping.add((triples_map, rml[1].logicalSource, logical_source))
        mapping.add((logical_source, rml[1].source, Literal(source_path)))
        mapping.add((logical_source, rml[1].referenceFormulation, ql[1].CSV))
        #mapping.add((logical_source, rml[1].iterator, Literal(iterator_mask)))

        # Collect subjectmap predicate and object from subject df
        # These will be added to the graph and then used later on
        authtp_column_name = None # only set for RICO_AUTHTP replaced subjects
        if subject_uri in rico_authtp_subjects.keys():
            true_subject_uri, authtp_column_name = rico_authtp_subjects[subject_uri]
            true_subject_po = uriref_str_to_map(extract_uriref_str(true_subject_uri))
            subject_map_predicate = true_subject_po[map_predicate_label]
            uri_mask = true_subject_po[map_object_label]
        else:
            subject_map_predicate = subject_row[map_predicate_label]
            uri_mask = subject_row[map_object_label]
        #URIRef(urllib.parse.unquote(str(subject)))
        #uri_mask = construct_uri_mask(subjects_df, i)

        # This is where the actual UUID substitution happens, right before writing to RML
        # Commented out because we are not using UUIDs eventually, as of yet
        #uri_mask = substitute_uuid(uri_mask, triplesmap_name)
        
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

        if not subject_mnemonic:
            # Add map predicate and object from df to subject map
            mapping.add((subject_map, subject_map_predicate, uri_mask))
        #
        # Now that we have handled all the no-mnemonic cases (both when no valid RML
        # syntax in the drawio graph AND when the syntax is there but no mnemonic
        # used), let's handle cases with both valid RML syntax and mnemonic set.
        #
        # We will use an FnO logic to leave subject maps empty at mapping
        # when the value of the source CSV column in empty.
        #
        # Here comes:
        else:
            input_tuples = [(rml[1].reference, Literal(subject_mnemonic))]
            return_tuple = (subject_map_predicate, uri_mask)
            fno_mnemonic_logic = fno_map_value_unless_isnull(
                    rml_g = mapping,
                    return_tuple = return_tuple,
                    input_tuples = input_tuples)
            
            # IMPORTANT! Note that the below only executes if mnemonic is set in drawio,
            # so for entities defined as constants RICO_AUTHTP will be bugged
            if authtp_column_name:
                authtp_value = rico_authtp_dict[rico_class]
                authtp_column_tuples = [(rml[1].reference, Literal(authtp_column_name))]
                authtp_value_tuples = [(rr[1].template, Literal(authtp_value)),
                                        (rr[1].termType, rr[1].Literal)]
                return_tuple = (fnml[1].functionValue, fno_mnemonic_logic)
                rico_authtp_string_match = fno_map_this_string_match(
                    rml_g = mapping,
                    return_tuple = return_tuple,
                    input_value_tuples = authtp_column_tuples,
                    regex_tuples = authtp_value_tuples)
                mapping.add((subject_map, fnml[1].functionValue, rico_authtp_string_match))
            else:
                mapping.add((subject_map, fnml[1].functionValue, fno_mnemonic_logic))

        # Record source mnemonic as a triple
        # Commenting out for now because not sure yet
        # how exactly in RDF we want this implemented
        #if subject_mnemonic:
        #    mnemonic_schema_uri = URIRef(f"{str(base_schema_uri)}/{get_second_term()}/Mnemonic/{subject_mnemonic}")
        #    predicate_for_old_mnemonic = rico[1].hasOrHadIdentifier
        #    add_custom_triple_to_triplesmap(predicate_for_old_mnemonic, mnemonic_schema_uri, triples_map)

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
                original_object = parsed_result['original_object']
                object_map_predicate = parsed_result[map_predicate_label]
                object_map_object = parsed_result[map_object_label]
                object_mnemonic = parsed_result[mnemonic_label]
                rdfs_label_triple = None # to use later

                # Handle possible increment requests in object mnemonic
                object_mnemonic_i_from, object_mnemonic_i_to = get_mnemonic_i_from_to(object_mnemonic)
                for object_mnemonic_i in range(object_mnemonic_i_from, object_mnemonic_i_to + 1):
                    #if (parsed_result['original_subject']==subject_row['original_subject'] and
                    #    object_mnemonic != subject_mnemonic):
                    #    continue
                    # Define a predicate-object map
                    predicate_object_map = BNode()
                    pom_create_triple = (triples_map, rr[1].predicateObjectMap, predicate_object_map)
                    mapping.add(pom_create_triple)

                    # Add predicate to predicate-object map
                    pom_predicate_triple = (predicate_object_map, rr[1].predicate, URIRef(predicate))
                    mapping.add(pom_predicate_triple)

                    # Define an empty object map within the predicate-object map
                    object_map = BNode()
                    om_create_triple = (predicate_object_map, rr[1].objectMap, object_map)
                    mapping.add(om_create_triple)

                    # If not RiC-O, then nothing applies and just attach as literal
                    # In the current version of drawio parser only rdfs:label is supported
                    # and such, so this is essential to bypass these. However, I am not
                    # sure at this point how well this would work if other namespaces
                    # were fully supported by drawio parser.
                    if not is_rico: # any other namespace
                        if norm_predicate == 'rdfs:label': # handle labels from drawio parser
                            if rdfs_label_triple: # already added - remove empty nodes and continue
                                mapping.remove(pom_create_triple)
                                mapping.remove(pom_predicate_triple)
                                mapping.remove(om_create_triple)
                            else:
                                mapping.add((object_map, rr[1].termType, rr[1].Literal)) # print as literal
                                # Only using map object here because object_map_predicate is irrelevant.
                                # But using uri_mask of subject as map object and not object_map_object
                                # because the former has been disaggregated but the latter has not
                                pretty_omo = prettify_rdfs_label(uri_mask) 
                                rdfs_label_triple = (object_map, rr[1].template, Literal(pretty_omo))
                                mapping.add(rdfs_label_triple)
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
                        # Well, the below does seem to work but may be a bad idea because object's mnemonic
                        # does not necessarily have to match subject's mnemonic (what if they are separate increments?)
                        # Thus, I commented out that block
                        #if ((object != original_object) and # means it is one of the disaggregated ones
                        #    (str(object_mnemonic_ith) != str(subject_mnemonic))): # mismatched phantoms - remove empty nodes and continue
                        #    mapping.remove(pom_create_triple)
                        #    mapping.remove(pom_predicate_triple)
                        #    mapping.remove(om_create_triple)
                        #    continue
                        if object_mnemonic_ith in private_mnemonics:
                            mapping.add((object_map, rr[1].constant, URIRef(f"censored#{object_mnemonic_ith}")))
                            continue
                        object_map_predicate = parsed_result[map_predicate_label]
                        object_map_object = parsed_result[map_object_label]
                        if object_map_object: # just in case user forgot to set it in drawio
                            # Logic to substitute increment request with an actual number for object
                            object_map_object_ith = mnemonic_i_regex.sub(str(object_mnemonic_i), str(object_map_object)) if object_mnemonic_i_to > 1 else str(object_map_object)
                            object_map_object_ith = URIRef(object_map_object_ith) if isinstance(object_map_object_ith, URIRef) else Literal(object_map_object_ith)
                            mapping.add((object_map, object_map_predicate, object_map_object_ith))

    # Serialize and print the RDF graph
    ttl = mapping.serialize(format='turtle')
    with open(rml_path, 'w') as f:
        f.write(ttl)
    print(f"\n\nSuccessfully saved RML map to: '{rml_path}'")
    #print(ttl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Map schema of choice")
    parser.add_argument("schema", help="Choose one: add or auth.")
    parser.add_argument("source", nargs='?', help="Filename of source CSV without extension. Defaults to the head=6 version for chosen schema.")

    args = parser.parse_args()

    __init__(str(args.schema).lower(), args.source)
