from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, DCTERMS
import pandas as pd
import re
import urllib.parse
from pprint import pprint

def __init__():
    # Define GBAD schema ontology
    base_data_uri = 'https://data.archives.gov.on.ca'
    base_gbad_uri = URIRef(f"{base_data_uri}/RiC-O_1-0-1")
    base_schema_uri = URIRef(f"{base_data_uri}/schema")
    base_kb_uri = URIRef(f"{base_data_uri}/KB")
    base_auth_uri = URIRef(f"{base_schema_uri}/authority")
    base_add_uri = URIRef(f"{base_schema_uri}/description-listings")
    base_mapping_uri = URIRef(f"{base_schema_uri}/mapping")

    # Choose ontology to map
    base_uri = base_data_uri
    graph_path = 'gbad/schema/authority/general_authority_to_ric-o_model_2024-08-20_pz.ttl'

    # Create the input RDF graph
    g = Graph(base = base_uri)
    g.parse(graph_path,
            format="turtle")  # Adjust the format as needed

    # Define custom prefixes
    rico_uri = 'https://www.ica.org/standards/RiC/ontology#'
    rico = ('rico', Namespace(rico_uri))
    ns = ('data', Namespace(URIRef(f"{base_uri}/")))

    # Define common prefixes
    rdf = ('rdf', RDF)
    rdfs = ('rdfs', RDFS)
    owl = ('owl', OWL)

    # Bind prefixes to namespaces
    g_namespace_manager = g.namespace_manager
    g_namespace_manager.bind(*rico)
    g_namespace_manager.bind(*rdf)
    g_namespace_manager.bind(*rdfs)
    g_namespace_manager.bind(*owl)
    g_namespace_manager.bind(*ns)

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

        object_query = f"""
        SELECT ?object
        WHERE {{
        <{subject}> rdf:type ?object .
        FILTER (STRSTARTS(STR(?object), "{rico[1]}"))
        }}
        """
        # Execute the query
        object_result = g.query(object_query)
        for object_row in object_result:
            parsed_result = {
                'subject': subject,
                'predicate': predicate,
                'object': object
            }
            parsed_results.append(parsed_result)
        
    #print(parsed_results[:5]) # debug

    # Convert the parsed results to a dataframe
    parsed_df = pd.DataFrame(parsed_results)

    def normalize_uri(uri):
        if isinstance(uri, URIRef):
            return g_namespace_manager.normalizeUri(uri)
        return None

    # SELECT ?s a ?o
    subjects_df = parsed_df[
        (parsed_df['predicate'].apply(lambda x: str(normalize_uri(x))) == 'rdf:type') &
        (parsed_df['object'].apply(lambda x: str(normalize_uri(x)).startswith(f"{rico[0]}:")))
    ].loc[:,['subject','object']]

    def extract_uriref_str(uriref):
        # Replace namespace URIs with prefix codes
        uriref_str = str(normalize_uri(uriref))
        # Remove base URI prefix
        uriref_str = uriref_str.replace(f"{ns[0]}:", '')
        # Decode special URI entities
        uriref_str = urllib.parse.unquote(uriref_str)
        return uriref_str

    def generate_triplesmap_name(row):
        # This assumes that subject URIs are unique
        subject_str = extract_uriref_str(row['subject'])
        print(uriref_str_to_map(subject_str))
        # Replace with underscores anything but Latin letters, numbers, hyphens, and underscores
        cleaned_subject = re.sub(r'[^0-9a-z_-]', '_', subject_str, flags=re.IGNORECASE)
        return cleaned_subject
    
    # Define RML-specific prefixes
    rml = ('rml', Namespace('http://semweb.mmlab.be/ns/rml#'))
    rr = ('rr', Namespace('http://www.w3.org/ns/r2rml#'))
    ql = ('ql', Namespace('http://semweb.mmlab.be/ns/ql#'))
    csvw = ('csvw', Namespace('http://www.w3.org/ns/csvw#'))
    
    def uriref_str_to_map(uriref_str: str):
        predicate = None
        object = None
        uriref_str = re.sub('\s+', ' ', uriref_str)
        def remove(predicate: URIRef, uriref_str):
            sin_predicate = re.sub(f"^{str(predicate)}\s+", "", uriref_str)
            return sin_predicate
        # Literal mapped from source
        if uriref_str.startswith(str(normalize_uri(rml[1].reference))):
            predicate = rml[1].reference
            object = Literal(remove(predicate, uriref_str))
        # URI mapped from source
        elif uriref_str.startswith(str(normalize_uri(rr[1].template))):
            predicate = rr[1].template
            object = URIRef(remove(predicate, uriref_str))
        # Constant URI
        elif uriref_str.startswith(str(normalize_uri(rr[1].constant))):
            predicate = rr[1].constant
            object = URIRef(remove(predicate, uriref_str))
        # Treat anything else as a literal
        else:
            object = Literal(uriref_str)
        return (predicate, object)

    def generate_rico_name(row):
        object_uri = row['object']
        object_str = str(normalize_uri(object_uri))
        cleaned_object = object_str
        return cleaned_object
    
    def extract_mnemonic(row):
        subject_str = str(row['object'])

    def is_private_mnemonic(subject_row):
        global private_mnemonics
        subject_uri_str = subject_row['subject']
        # Looks up directly in hardcoded URIs, with %7B and %7D being URI entities for curly brackets
        is_private = any(f"%7B{mnemonic}%7D" in subject_uri_str for mnemonic in private_mnemonics)
        return is_private

    triplesmap_label = 'TriplesMap'
    rico_name_label = 'RiC-O Name'.replace(' ','_')
    mnemonic_label = 'Authority Mnemonic'.replace(' ','_')

    subjects_df[triplesmap_label] = subjects_df.apply(generate_triplesmap_name, axis=1)
    subjects_df[rico_name_label] = subjects_df.apply(generate_rico_name, axis=1)
    subjects_df[rico_name_label] = subjects_df.apply(generate_rico_name, axis=1)
    subjects_df.drop('object', axis=1, inplace=True)

    # Convert preprocessed DataFrame to HTML
    #from IPython.display import display, HTML, Markdown
    sorted_columns = [triplesmap_label, rico_name_label, 'subject']
    display_table = subjects_df[sorted_columns].sort_values(by=triplesmap_label, ascending=True)
    #html_table = display_table.to_html(index=False) # for Jupyter Notebook
    #display(HTML(html_table)) # for Jupyter Notebook
    #print(display_table[[triplesmap_label, rico_name_label]]) # debug
    return None
    # Initialize an RDF graph
    mapping = Graph(base = URIRef(f"{base_gbad_uri}/"))
    source_path = 'gbad/mapping/source/authority_head_6.csv'

    # Define custom prefix
    maps = ('data', Namespace(URIRef(f"{base_mapping_uri}#")))

    # Bind prefixes to namespaces
    map_namespace_manager = mapping.namespace_manager
    map_namespace_manager.bind(*rico)
    map_namespace_manager.bind(*rdf)
    map_namespace_manager.bind(*rdfs)
    map_namespace_manager.bind(*owl)
    map_namespace_manager.bind(*rml)
    map_namespace_manager.bind(*rr)
    map_namespace_manager.bind(*ql)
    map_namespace_manager.bind(*csvw)
    map_namespace_manager.bind(*maps)

    # Define blank nodes and triples
    #agent_name_map = BNode()
    #agent_map = BNode()

    # Triples for :AgentNameAUTH13
    #mapping.add((maps[1].AgentNameAUTH13, RDF.type, rr[1].TriplesMap))

    # Set labels for reference fields
    auth_heading_label = 'HEADING'
    add_refd_label = 'REFD'
    add_ref_add_label = 'REF_ADD'
    add_ref_file_label = 'REF_FILE'
    add_title_label = 'TITLE'
    private_mnemonics = ['ARCHAU', 'CMTAU']

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
        
    def is_private_mnemonic(subject_row):
        global private_mnemonics
        subject_uri_str = subject_row['subject']
        # Looks up directly in hardcoded URIs, with %7B and %7D being URI entities for curly brackets
        is_private = any(f"%7B{mnemonic}%7D" in subject_uri_str for mnemonic in private_mnemonics)
        return is_private

    # Construct RML graph
    for i, subject_row in subjects_df.iterrows():
        # Skip private fields removed from input data
        if is_private_mnemonic(subject_row):
            print(f"Aha! Here we go: '{subject_row['subject']}'")
            continue

        # Define TriplesMap
        subject = maps[1][subject_row[triplesmap_label]]
        mapping.add((subject, RDF.type, rr[1].TriplesMap))

        # Define Logical Source
        logical_source = BNode()
        mapping.add((subject, rml[1].logicalSource, logical_source))
        mapping.add((logical_source, rml[1].source, Literal(source_path)))
        mapping.add((logical_source, rml[1].referenceFormulation, ql[1].CSV))
        #mapping.add((logical_source, rml[1].iterator, Literal(iterator_mask)))

        # Define Subject Map
        subject_map = BNode()
        mapping.add((subject, rr[1].subjectMap, subject_map))
        #URIRef(urllib.parse.unquote(str(subject)))
        #uri_mask = construct_uri_mask(subjects_df, i)
        uri_mask = urllib.parse.unquote(str(subject_row['subject']))
        mapping.add((subject_map, rr[1].template, Literal(uri_mask)))
        rico_class = subject_row[rico_name_label][5:]
        mapping.add((subject_map, rr[1]['class'], rico[1][rico_class]))
        
        # Define Predicate-Object Map
        for parsed_result in parsed_results:
            #break
            if ((parsed_result['subject'] == subject_row['subject']) &
                (parsed_result[rico_name_label] == subject_row[rico_name_label])
            ):
                predicate = parsed_result['predicate']
                if ((map_namespace_manager.normalizeUri(predicate).startswith(f"{rico[0]}:")) |
                    (map_namespace_manager.normalizeUri(predicate).startswith(f"{rdfs[0]}:"))
                ):
                    predicate_object_map = BNode()
                    mapping.add((subject, rr[1].predicateObjectMap, predicate_object_map))
                    mapping.add((predicate_object_map, rr[1].predicate, URIRef(predicate)))
                    object = parsed_result['object']
                    object_map = BNode()
                    mapping.add((predicate_object_map, rr[1].objectMap, object_map))
                    # Case when the object is supposed to reference another Subject map
                    if object in set(subjects_df['subject']):
                        triplesmap = maps[1][subjects_df[subjects_df['subject']==parsed_result['object']][triplesmap_label].iloc[0]]
                        mapping.add((object_map, rr[1].parentTriplesMap, triplesmap))
                        #join_condition = BNode()
                        #mapping.add((object_map, rr[1].joinCondition, join_condition))
                        #mnemonic = parsed_result[mnemonic_label]
                        #mapping.add((join_condition, rr[1].child, Literal(mnemonic)))
                        #mapping.add((join_condition, rr[1].parent, Literal(mnemonic)))
                    elif isinstance(object, URIRef):
                        mapping.add((object_map, rr[1].constant, object))
                        #rico_class = subjects_df[subjects_df['subject']==object][rico_name_label].iloc[0]
                        #rico_class_uri = URIRef(f"{rico[1]}{rico_class[5:]}")
                        #parsed_object = parse_node(object, rico_class_uri)
                        #object_mnemonic = parsed_object[mnemonic_label]
                        # Case when the object is supposed to reference a table row
                        #if mnemonic:
                        #    mapping.add((object_map, rml[1].reference , Literal(object_mnemonic)))
                        # Case when the object is supposed to be an external URI, or
                        # is actually supposed to reference a table row but is unmatched
                        #else:
                        #    mapping.add((object_map, rr[1].constant , object))
                    # Case when the object is a supposed to be a literal
                    elif isinstance(object, Literal):
                        #node_label_dict = break_node_label(object)
                        #mnemonic = node_label_dict[mnemonic_label]
                        if is_private_mnemonic(subject_row):
                            mapping.add((object_map, rr[1].constant, URIRef(f"censored#{mnemonic}")))
                            continue
                        mapping.add((object_map, rml[1].reference, Literal(mnemonic)))

    # Serialize and print the RDF graph
    print(mapping.serialize(format='turtle'))

if __name__ == '__main__':
    __init__()
