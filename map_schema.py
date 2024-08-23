from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, DCTERMS
import pandas as pd
import re
import urllib.parse
from pprint import pprint

# Set labels for reference fields
auth_heading_label = 'HEADING'
add_refd_label = 'REFD'
add_ref_add_label = 'REF_ADD'
add_ref_file_label = 'REF_FILE'
add_title_label = 'TITLE'
private_mnemonics = ['ARCHAU', 'CMTAU']

triplesmap_label = 'TriplesMap'
uriref_str_label = 'uriref_str'
map_predicate_label = 'map_predicate'
map_object_label = 'map_object'

def __init__():
    # Define GBAD schema ontology
    base_data_uri = 'https://data.archives.gov.on.ca'
    base_gbad_uri = URIRef(f"{base_data_uri}/RiC-O_1-0-1")
    base_schema_uri = URIRef(f"{base_data_uri}/schema")
    #base_kb_uri = URIRef(f"{base_data_uri}/KB")
    base_auth_uri = URIRef(f"{base_schema_uri}/authority")
    base_add_uri = URIRef(f"{base_schema_uri}/description-listings")
    base_mapping_uri = URIRef(f"{base_schema_uri}/mapping")

    # Choose ontology to map
    base_uri = base_data_uri
    graph_path = 'gbad/schema/authority/general_authority_to_ric-o_model_2024-08-20_pz.ttl'
    rml_path = graph_path[:-3]+ "rml"

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

    def generate_triplesmap_name(row):
        # This implementation assumes that subject URIs are unique
        subject_str = row[uriref_str_label]
        # Replace with underscores anything but Latin letters, numbers, hyphens, and underscores
        cleaned_subject = re.sub(r'[^0-9a-z_-]', '_', subject_str, flags=re.IGNORECASE)
        return cleaned_subject
    
    # Necessary to init namespace manager for uriref_str_to_map
    # Initialize an RDF graph
    mapping = Graph(base = URIRef(f"{base_gbad_uri}/"))
    source_path = 'gbad/mapping/source/authority_head_6.csv'
    
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
            map_object = Literal(remove(norm(map_predicate), uriref_str))
        # Constant URI
        elif uriref_str.startswith(norm(rr[1].constant)):
            map_predicate = rr[1].constant
            map_object = URIRef(remove(norm(map_predicate), uriref_str))
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
            if map_predicate != rr[1].template:
                return None
            pattern = r"\{([A-Z]+)\}"
            matches = re.findall(pattern, map_object)
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
    sorted_columns = [triplesmap_label, rico_name_label, map_predicate_label, map_object_label, mnemonic_label, 'subject']
    display_table = subjects_df[subjects_df[map_predicate_label].notnull()][sorted_columns].head(10).sort_values(by=triplesmap_label, ascending=True)
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
    #parsed_df.drop(uriref_str_label, axis=1, inplace=True)

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
    source_path = 'gbad/mapping/source/authority_head_6.csv'

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

        # Remove prefix from RiC-O name from subject df and add to graph
        rico_name = subject_row[rico_name_label]
        rico_class = rico_name[5:]
        # So this adds the rdf:type definition
        mapping.add((subject_map, rr[1]['class'], rico[1][rico_class]))

        # Deal with predicates and objects in full triples df
        # Subset triples with the subject and RiC-O class from i-loop
        objectmap_df = parsed_df[(
            (parsed_df['subject']==subject_uri) &
            (parsed_df[rico_name_label] == rico_name)
        )]
        for k, parsed_result in objectmap_df.iterrows():
            # Only focus on RiC-O or RDFS predicates
            predicate = parsed_result['predicate']
            if ((normalize_uri(predicate, mapping.namespace_manager).startswith(f"{rico[0]}:")) |
                (normalize_uri(predicate, mapping.namespace_manager).startswith(f"{rdfs[0]}:"))
            ):
                # Now we can actually iterate over objects
                object = parsed_result['object']
                object_map_predicate = parsed_result[map_predicate_label]
                object_map_object = parsed_result[map_object_label]
                object_mnemonic = parsed_result[mnemonic_label]

                # Define a predicate-object map
                predicate_object_map = BNode()
                mapping.add((triples_map, rr[1].predicateObjectMap, predicate_object_map))

                # Add predicate to predicate-object map
                mapping.add((predicate_object_map, rr[1].predicate, URIRef(predicate)))

                # Define an empty object map within the predicate-object map
                object_map = BNode()
                mapping.add((predicate_object_map, rr[1].objectMap, object_map))

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
                    if object_mnemonic in private_mnemonics:
                        mapping.add((object_map, rr[1].constant, URIRef(f"censored#{object_mnemonic}")))
                        continue
                    object_map_predicate = parsed_result[map_predicate_label]
                    object_map_object = parsed_result[map_object_label]
                    if object_map_object: # just in case user forgot to set it in drawio
                        mapping.add((object_map, object_map_predicate, object_map_object))

    # Serialize and print the RDF graph
    ttl = mapping.serialize(format='turtle')
    with open(rml_path, 'w') as f:
        f.write(mapping.serialize(format='turtle'))
    print(f"\n\nSuccessfully saved RML map to: '{rml_path}'")
    #print(ttl)

if __name__ == '__main__':
    __init__()
