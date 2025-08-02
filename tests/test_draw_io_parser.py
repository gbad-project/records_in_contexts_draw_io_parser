"""
Tests the DrawIOXMLTree class in the way it would be used when running
draw_io_parser.py
"""

from pathlib import Path
from unittest import TestCase

from draw_io_parser import (
    DEFAULT_CAPITALISATION_SCHEME, DEFAULT_INDENTATION, DEFAULT_MAX_GAP,
    DrawIOXMLTree, SerialisationConfig, individual_blocks, serialise)

import io
import sys
from unittest.mock import patch
import rdflib
from rdflib import Graph, URIRef, Namespace
from rdflib.namespace import RDF, OWL, RDFS

_examples_directory = Path.cwd() / "examples"

_serialisation_config = SerialisationConfig(
    infer_type_of_literals=True,
    include_preamble=False,
    ontology_iri=None,
    prefix=None,
    prefix_iri=None,
    indentation=DEFAULT_INDENTATION,
    include_label=True)

_metacharacters = [(",", "-"), ("[", "{"), ("]", "}")]


# class TestDrawIOParser(TestCase):
#     """
#     Tests that the parsing of the .drawio files in the examples/ directory
#     gives the expected results
#     """
#
#     def test_examples_without_preamble(self) -> None:
#         """
#         Tests, for each .drawio file in the examples/ directory, that the OWL
#         generated (without preamble) is equal to that contained in the
#         corresponding _without_preamble.owl file
#         """
#         self.maxDiff = None  # pylint: disable=invalid-name
#         for path in _examples_directory.iterdir():
#             if path.suffix != ".drawio":
#                 continue
#             with open(path, "r", encoding="utf-8") as draw_io_file:
#                 draw_io_xml_tree = DrawIOXMLTree(draw_io_file.read())
#             blocks = individual_blocks(
#                 draw_io_xml_tree.individuals_and_arrows(
#                     False, DEFAULT_MAX_GAP),
#                 _metacharacters,
#                 "",
#                 DEFAULT_CAPITALISATION_SCHEME)
#             owl = serialise(blocks, _serialisation_config)
#             with open(
#                     _examples_directory / f"{path.stem}_without_preamble.owl",
#                     "r",
#                     encoding="utf-8") as owl_file:
#                 self.assertEqual(owl.strip(), owl_file.read().strip())
#
#     def test_example_with_preamble(self) -> None:
#         """
#         Tests, for one of the .drawio files in the examples/ directory, that the
#         OWL generated is equal to that contained in the corresponding
#         .owl file with the correct preamble (with the ontology IRI specified
#         here to be the same as that in the .owl file)
#         """
#         self.maxDiff = None  # pylint: disable=invalid-name
#         path = _examples_directory / "koronakommisjonen.drawio"
#         with open(path, "r", encoding="utf-8") as draw_io_file:
#             draw_io_xml_tree = DrawIOXMLTree(draw_io_file.read())
#         blocks = individual_blocks(
#             draw_io_xml_tree.individuals_and_arrows(False, DEFAULT_MAX_GAP),
#             [],
#             "",
#             DEFAULT_CAPITALISATION_SCHEME)
#         serialisation_config = SerialisationConfig(
#             infer_type_of_literals=True,
#             include_preamble=True,
#             ontology_iri="ontology://generated-from-draw-io/2024-04-26T01-31-21",
#             prefix=None,
#             prefix_iri=None,
#             indentation=DEFAULT_INDENTATION,
#             include_label=True)
#         owl = serialise(blocks, serialisation_config)
#         with open(
#                 _examples_directory / f"{path.stem}.owl",
#                 "r",
#                 encoding="utf-8") as owl_file:
#             self.assertEqual(owl.strip(), owl_file.read().strip())


class TestDynamicOntologyLoading(TestCase):
    """
    Tests the dynamic ontology loading functionality.
    """

    def _create_mock_rico_ontology_from_hardcoded_data(self):
        """
        Generates a mock rico.rdf file from the original hardcoded lists.
        """
        g = Graph()

        prefixes = {
            'rico': 'https://www.ica.org/standards/RiC/ontology#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }

        namespaces = {prefix: Namespace(uri) for prefix, uri in prefixes.items()}

        for prefix, namespace in namespaces.items():
            g.bind(prefix, namespace)

        _classes = ["owl:DatatypeProperty", "rico:AccumulationRelation", "rico:Activity", "rico:ActivityDocumentationRelation", "rico:ActivityType", "rico:Agent", "rico:AgentControlRelation", "rico:AgentHierarchicalRelation", "rico:AgentName", "rico:AgentTemporalRelation", "rico:AgentToAgentRelation", "rico:Appellation", "rico:AppellationRelation", "rico:AuthorityRelation", "rico:AuthorshipRelation", "rico:CarrierExtent", "rico:CarrierType", "rico:ChildRelation", "rico:Concept", "rico:ContentType", "rico:Coordinates", "rico:CorporateBody", "rico:CorporateBodyType", "rico:CorrespondenceRelation", "rico:CreationRelation", "rico:Date", "rico:DateType", "rico:DemographicGroup", "rico:DerivationRelation", "rico:DescendanceRelation", "rico:DocumentaryFormType", "rico:Event", "rico:EventRelation", "rico:EventType", "rico:Extent", "rico:ExtentType", "rico:Family", "rico:FamilyRelation", "rico:FamilyType", "rico:FunctionalEquivalenceRelation", "rico:Group", "rico:GroupSubdivisionRelation", "rico:Identifier", "rico:IdentifierType", "rico:Instantiation", "rico:InstantiationExtent", "rico:InstantiationToInstantiationRelation", "rico:IntellectualPropertyRightsRelation", "rico:KnowingOfRelation", "rico:KnowingRelation", "rico:Language", "rico:LeadershipRelation", "rico:LegalStatus", "rico:ManagementRelation", "rico:Mandate", "rico:MandateRelation", "rico:MandateType", "rico:Mechanism", "rico:MembershipRelation", "rico:MigrationRelation", "rico:Name", "rico:OccupationType", "rico:OrganicOrFunctionalProvenanceRelation", "rico:OrganicProvenanceRelation", "rico:OwnershipRelation", "rico:PerformanceRelation", "rico:Person", "rico:PhysicalLocation", "rico:Place", "rico:PlaceName", "rico:PlaceRelation", "rico:PlaceType", "rico:Position", "rico:PositionHoldingRelation", "rico:PositionToGroupRelation", "rico:ProductionTechniqueType", "rico:Proxy", "rico:Record", "rico:RecordPart", "rico:RecordResource", "rico:RecordResourceExtent", "rico:RecordResourceGeneticRelation", "rico:RecordResourceHoldingRelation", "rico:RecordResourceToInstantiationRelation", "rico:RecordResourceToRecordResourceRelation", "rico:RecordSet", "rico:RecordSetType", "rico:RecordState", "rico:Relation", "rico:RepresentationType", "rico:RoleType", "rico:Rule", "rico:RuleRelation", "rico:RuleType", "rico:SequentialRelation", "rico:SiblingRelation", "rico:SpouseRelation", "rico:TeachingRelation", "rico:TemporalRelation", "rico:Thing", "rico:Title", "rico:Type", "rico:TypeRelation", "rico:UnitOfMeasurement", "rico:WholePartRelation", "rico:WorkRelation"]
        _object_properties = ["rdfs:subPropertyOf", "rico:affectsOrAffected", "rico:agentHasOrHadLocation", "rico:authorizedBy", "rico:authorizes", "rico:contained", "rico:containsOrContained", "rico:containsTransitive", "rico:describesOrDescribed", "rico:directlyContains", "rico:directlyFollowsInSequence", "rico:directlyIncludes", "rico:directlyPrecedesInSequence", "rico:documentedBy", "rico:documents", "rico:existsOrExistedIn", "rico:expressesOrExpressed", "rico:followedInSequence", "rico:followsInSequenceTransitive", "rico:followsInTime", "rico:followsOrFollowed", "rico:hadComponent", "rico:hadConstituent", "rico:hadPart", "rico:hadSubdivision", "rico:hadSubevent", "rico:hadSubordinate", "rico:hasAccumulator", "rico:hasActivityType", "rico:hasAddressee", "rico:hasAncestor", "rico:hasAuthor", "rico:hasBeginningDate", "rico:hasBirthDate", "rico:hasBirthPlace", "rico:hasCarrierType", "rico:hasChild", "rico:hasCollector", "rico:hasComponentTransitive", "rico:hasConstituentTransitive", "rico:hasContentOfType", "rico:hasCopy", "rico:hasCreationDate", "rico:hasCreator", "rico:hasDateType", "rico:hasDeathDate", "rico:hasDeathPlace", "rico:hasDescendant", "rico:hasDestructionDate", "rico:hasDirectComponent", "rico:hasDirectConstituent", "rico:hasDirectPart", "rico:hasDirectSubdivision", "rico:hasDirectSubevent", "rico:hasDirectSubordinate", "rico:hasDocumentaryFormType", "rico:hasDraft", "rico:hasEndDate", "rico:hasEventType", "rico:hasExtent", "rico:hasExtentType", "rico:hasFamilyAssociationWith", "rico:hasFamilyType", "rico:hasGeneticLinkToRecordResource", "rico:hasIdentifierType", "rico:hasModificationDate", "rico:hasOrHadAgentName", "rico:hasOrHadAllMembersWithCategory", "rico:hasOrHadAllMembersWithContentType", "rico:hasOrHadAllMembersWithCreationDate", "rico:hasOrHadAllMembersWithDocumentaryFormType", "rico:hasOrHadAllMembersWithLanguage", "rico:hasOrHadAllMembersWithLegalStatus", "rico:hasOrHadAllMembersWithRecordState", "rico:hasOrHadAnalogueInstantiation", "rico:hasOrHadAppellation", "rico:hasOrHadAuthorityOver", "rico:hasOrHadCategory", "rico:hasOrHadType", "rico:hasOrHadComponent", "rico:hasOrHadConstituent", "rico:hasOrHadController", "rico:hasOrHadCoordinates", "rico:hasOrHadCorporateBodyType", "rico:hasOrHadCorrespondent", "rico:hasOrHadDemographicGroup", "rico:hasOrHadDerivedInstantiation", "rico:hasOrHadDigitalInstantiation", "rico:hasOrHadEmployer", "rico:hasOrHadHolder", "rico:hasOrHadIdentifier", "rico:hasOrHadInstantiation", "rico:hasOrHadIntellectualPropertyRightsHolder", "rico:hasOrHadJurisdiction", "rico:hasOrHadLanguage", "rico:hasOrHadLeader", "rico:hasOrHadLegalStatus", "rico:hasOrHadLocation", "rico:hasOrHadMainSubject", "rico:hasOrHadManager", "rico:hasOrHadMandateType", "rico:hasOrHadMember", "rico:hasOrHadMostMembersWithCreationDate", "rico:hasOrHadName", "rico:hasOrHadOccupationOfType", "rico:hasOrHadOwner", "rico:hasOrHadPart", "rico:hasOrHadParticipant", "rico:hasOrHadPhysicalLocation", "rico:hasOrHadPlaceName", "rico:hasOrHadPlaceType", "rico:hasOrHadPosition", "rico:hasOrHadRuleType", "rico:hasOrHadSomeMembersWithCategory", "rico:hasOrHadSomeMembersWithContentType", "rico:hasOrHadSomeMembersWithCreationDate", "rico:hasOrHadSomeMembersWithLanguage", "rico:hasOrHadSomeMembersWithLegalStatus", "rico:hasOrHadSomeMembersWithRecordState", "rico:hasOrHadSomeMemberswithDocumentaryFormType", "rico:hasOrHadSpouse", "rico:hasOrHadStudent", "rico:hasOrHadSubdivision", "rico:hasOrHadSubevent", "rico:hasOrHadSubject", "rico:hasOrHadSubordinate", "rico:hasOrHadTeacher", "rico:hasOrHadTitle", "rico:hasOrHadWorkRelationWith", "rico:hasOrganicOrFunctionalProvenance", "rico:hasOrganicProvenance", "rico:hasOriginal", "rico:hasPartTransitive", "rico:hasProductionTechniqueType", "rico:hasPublicationDate", "rico:hasPublisher", "rico:hasReceiver", "rico:hasRecordSetType", "rico:hasRecordState", "rico:hasReply", "rico:hasRepresentationType", "rico:hasSender", "rico:hasSibling", "rico:hasSubdivisionTransitive", "rico:hasSubeventTransitive", "rico:hasSubordinateTransitive", "rico:hasSuccessor", "rico:hasUnitOfMeasurement", "rico:hasWithin", "rico:included", "rico:includesOrIncluded", "rico:includesTransitive", "rico:intersects", "rico:isAccumulatorOf", "rico:isActivityTypeOf", "rico:isAddresseeOf", "rico:isAgentAssociatedWithAgent", "rico:isAgentAssociatedWithPlace", "rico:isAssociatedWithDate", "rico:isAssociatedWithEvent", "rico:isAssociatedWithPlace", "rico:isAssociatedWithRule", "rico:isAuthorOf", "rico:isBeginningDateOf", "rico:isBirthDateOf", "rico:isBirthPlaceOf", "rico:isCarrierTypeOf", "rico:isChildOf", "rico:isCollectorOf", "rico:isComponentOfTransitive", "rico:isConstituentOfTransitive", "rico:isContainedByTransitive", "rico:isContentTypeOf", "rico:isCopyOf", "rico:isCreationDateOf", "rico:isCreatorOf", "rico:isDateAssociatedWith", "rico:isDateOfOccurrenceOf", "rico:isDateTypeOf", "rico:isDeathDateOf", "rico:isDeathPlaceOf", "rico:isDestructionDateOf", "rico:isDirectComponentOf", "rico:isDirectConstituentOf", "rico:isDirectPartOf", "rico:isDirectSubdivisionOf", "rico:isDirectSubeventOf", "rico:isDirectSubordinateTo", "rico:isDirectlyContainedBy", "rico:isDirectlyIncludedIn", "rico:isDocumentaryFormTypeOf", "rico:isDraftOf", "rico:isEndDateOf", "rico:isEquivalentTo", "rico:isEventAssociatedWith", "rico:isEventTypeOf", "rico:isExtentOf", "rico:isExtentTypeOf", "rico:isFamilyTypeOf", "rico:isFromUseDateOf", "rico:isFunctionallyEquivalentTo", "rico:isIdentifierTypeOf", "rico:isIncludedInTransitive", "rico:isInstantiationAssociatedWithInstantiation", "rico:isLastUpdateDateOf", "rico:isModificationDateOf", "rico:isOrWasAdjacentTo", "rico:isOrWasAffectedBy", "rico:isOrWasAgentNameOf", "rico:isOrWasAnalogueInstantiationOf", "rico:isOrWasAppellationOf", "rico:isOrWasCategoryOf", "rico:isOrWasCategoryOfAllMembersOf", "rico:isOrWasCategoryOfSomeMembersOf", "rico:isOrWasComponentOf", "rico:isOrWasConstituentOf", "rico:isOrWasContainedBy", "rico:isOrWasContentTypeOfAllMembersOf", "rico:isOrWasContentTypeOfSomeMembersOf", "rico:isOrWasControllerOf", "rico:isOrWasCoordinatesOf", "rico:isOrWasCorporateBodyTypeOf", "rico:isOrWasCreationDateOfAllMembersOf", "rico:isOrWasCreationDateOfMostMembersOf", "rico:isOrWasCreationDateOfSomeMembersOf", "rico:isOrWasDemographicGroupOf", "rico:isOrWasDerivedFromInstantiation", "rico:isOrWasDescribedBy", "rico:isOrWasDigitalInstantiationOf", "rico:isOrWasDocumentaryFormTypeOfAllMembersOf", "rico:isOrWasDocumentaryFormTypeOfSomeMembersOf", "rico:isOrWasEmployerOf", "rico:isOrWasEnforcedBy", "rico:isOrWasExpressedBy", "rico:isOrWasHolderOf", "rico:isOrWasHolderOfIntellectualPropertyRightsOf", "rico:isOrWasIdentifierOf", "rico:isOrWasIncludedIn", "rico:isOrWasInstantiationOf", "rico:isOrWasJurisdictionOf", "rico:isOrWasLanguageOf", "rico:isOrWasLanguageOfAllMembersOf", "rico:isOrWasLanguageOfSomeMembersOf", "rico:isOrWasLeaderOf", "rico:isOrWasLegalStatusOf", "rico:isOrWasLegalStatusOfAllMembersOf", "rico:isOrWasLegalStatusOfSomeMembersOf", "rico:isOrWasLocationOf", "rico:isOrWasLocationOfAgent", "rico:isOrWasMainSubjectOf", "rico:isOrWasManagerOf", "rico:isOrWasMandateTypeOf", "rico:isOrWasMemberOf", "rico:isOrWasNameOf", "rico:isOrWasOccupationTypeOf", "rico:isOrWasOccupiedBy", "rico:isOrWasOwnerOf", "rico:isOrWasPartOf", "rico:isOrWasParticipantIn", "rico:isOrWasPerformedBy", "rico:isOrWasPhysicalLocationOf", "rico:isOrWasPlaceNameOf", "rico:isOrWasPlaceTypeOf", "rico:isOrWasRecordStateOfAllMembersOf", "rico:isOrWasRecordStateOfSomeMembersOf", "rico:isOrWasRegulatedBy", "rico:isOrWasResponsibleForEnforcing", "rico:isOrWasRuleTypeOf", "rico:isOrWasSubdivisionOf", "rico:isOrWasSubeventOf", "rico:isOrWasSubjectOf", "rico:isOrWasSubordinateTo", "rico:isOrWasTitleOf", "rico:isOrWasUnderAuthorityOf", "rico:isOrganicOrFunctionalProvenanceOf", "rico:isOrganicProvenanceOf", "rico:isOriginalOf", "rico:isPartOfTransitive", "rico:isPlaceAssociatedWith", "rico:isPlaceAssociatedWithAgent", "rico:isProductionTechniqueTypeOf", "rico:isPublicationDateOf", "rico:isPublisherOf", "rico:isReceiverOf", "rico:isRecordResourceAssociatedWithRecordResource", "rico:isRecordSetTypeOf", "rico:isRecordStateOf", "rico:isRelatedTo", "rico:isReplyTo", "rico:isRepresentationTypeOf", "rico:isResponsibleForIssuing", "rico:isRuleAssociatedWith", "rico:isSenderOf", "rico:isSubdivisionOfTransitive", "rico:isSubeventOfTransitive", "rico:isSubordinateToTransitive", "rico:isSuccessorOf", "rico:isToUseDateOf", "rico:isUnitOfMeasurementOf", "rico:isWithin", "rico:issuedBy", "rico:knownBy", "rico:knows", "rico:knowsOf", "rico:migratedFrom", "rico:migratedInto", "rico:occupiesOrOccupied", "rico:occurredAtDate", "rico:overlapsOrOverlapped", "rico:performsOrPerformed", "rico:precededInSequence", "rico:precedesInSequenceTransitive", "rico:precedesInTime", "rico:precedesOrPreceded", "rico:proxyFor", "rico:proxyIn", "rico:regulatesOrRegulated", "rico:relationHasSource", "rico:relationHasTarget", "rico:resultedFromTheMergerOf", "rico:resultedFromTheSplitOf", "rico:resultsOrResultedFrom", "rico:resultsOrResultedIn", "rico:thingIsSourceOfRelation", "rico:wasComponentOf", "rico:wasConstituentOf", "rico:wasContainedBy", "rico:wasIncludedIn", "rico:wasLastUpdatedAtDate", "rico:wasMergedInto", "rico:wasPartOf", "rico:wasSplitInto", "rico:wasSubdivisionOf", "rico:wasSubeventOf", "rico:wasSubordinateTo", "rico:wasUsedFromDate", "rico:wasUsedToDate"]
        _datatype_properties = ["add:privateNote", "add:notes", "add:relatedMaterial", "add:associatedMaterial", "add:findingAidNote", "add:immediateSourceOfAcquisition", "add:custodialHistory", "add:availabilityOfOtherFormats", "add:accumulationDate", "add:howToOrder", "auth:sourceNote", "auth:functionNote", "auth:privateNote", "rdfs:label", "rico:accruals", "rico:accrualsStatus", "rico:altimetricSystem", "rico:altitude", "rico:authenticityNote", "rico:authorizingMandate", "rico:beginningDate", "rico:birthDate", "rico:carrierExtent", "rico:classification", "rico:conditionsOfAccess", "rico:conditionsOfUse", "rico:creationDate", "rico:date", "rico:dateQualifier", "rico:deathDate", "rico:destructionDate", "rico:endDate", "rico:expressedDate", "rico:generalDescription", "rico:geodesicSystem", "rico:geographicalCoordinates", "rico:height", "rico:history", "rico:identifier", "rico:instantiationExtent", "rico:instantiationStructure", "rico:integrityNote", "rico:lastModificationDate", "rico:latitude", "rico:length", "rico:location", "rico:longitude", "rico:measure", "rico:modificationDate", "rico:name", "rico:normalizedDateValue", "rico:normalizedValue", "rico:physicalCharacteristicsNote", "rico:physicalOrLogicalExtent", "rico:productionTechnique", "rico:publicationDate", "rico:qualityOfRepresentationNote", "rico:quantity", "rico:recordResourceExtent", "rico:recordResourceStructure", "rico:referenceSystem", "rico:relationCertainty", "rico:relationSource", "rico:relationState", "rico:ruleFollowed", "rico:scopeAndContent", "rico:structure", "rico:technicalCharacteristics", "rico:textualValue", "rico:title", "rico:type", "rico:unitOfMeasurement", "rico:usedFromDate", "rico:usedToDate", "rico:width"]

        for qname in _classes:
            if ":" not in qname: continue
            prefix, local_name = qname.split(':', 1)
            if prefix in namespaces:
                g.add((namespaces[prefix][local_name], RDF.type, RDFS.Class))

        for qname in _object_properties:
            if ":" not in qname: continue
            prefix, local_name = qname.split(':', 1)
            if prefix in namespaces:
                g.add((namespaces[prefix][local_name], RDF.type, OWL.ObjectProperty))

        for qname in _datatype_properties:
            if ":" not in qname: continue
            prefix, local_name = qname.split(':', 1)
            if prefix in namespaces:
                g.add((namespaces[prefix][local_name], RDF.type, OWL.DatatypeProperty))

        g.serialize(destination="tests/ontologies/rico.rdf", format="xml")

    def test_e2e_parsing_with_dynamic_ontologies(self):
        """
        End-to-end test for parsing a .drawio file with dynamically loaded
        ontologies.
        """
        self._create_mock_rico_ontology_from_hardcoded_data()
        self.maxDiff = None

        # Mock rdflib's parsing of remote graphs
        uri_map = {
            "https://www.ica.org/standards/RiC/ontology/": "tests/ontologies/rico.rdf",
            "http://www.w3.org/2000/01/rdf-schema#": "tests/ontologies/rdfs.rdf",
            "http://www.w3.org/2002/07/owl#": "tests/ontologies/owl.rdf",
        }

        original_parse = rdflib.graph.ConjunctiveGraph.parse

        def mock_parse(self, source=None, publicID=None, format=None, location=None, file=None, data=None, **kwargs):
            if source in uri_map:
                source = uri_map[source]
            return original_parse(self, source=source, publicID=publicID, format=format, location=location, file=file, data=data, **kwargs)

        with patch("rdflib.graph.ConjunctiveGraph.parse", mock_parse):
            drawio_file_path = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
            with open(drawio_file_path, "r", encoding="utf-8") as f:
                drawio_content = f.read()

            output_ontology_iri = "https://data.archives.gov.on.test.gbad.ca/Schema/Mapping"

            argv = [
                "draw_io_parser.py",
                "-m", "url",
                "-c", "none",
                "--label-disable",
                "-o", output_ontology_iri,
                "-x", "mapping", "-p", f"{output_ontology_iri}#",
                "-x", "rico", "-p", "https://www.ica.org/standards/RiC/ontology/",
                "-x", "rdfs", "-p", "http://www.w3.org/2000/01/rdf-schema#",
                "-x", "owl", "-p", "http://www.w3.org/2002/07/owl#",
                "-x", "gbad", "-p", f"file://{Path.cwd()}/gbad/schema/gbad.ttl",
                "-x", "auth", "-p", f"file://{Path.cwd()}/gbad/schema/authority.ttl",
                "-x", "add", "-p", f"file://{Path.cwd()}/gbad/schema/description-listings.ttl",
            ]

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = my_stdout = io.StringIO()

            with patch('draw_io_parser.stdin', io.StringIO(drawio_content)), patch('sys.argv', argv):
                from draw_io_parser import _main as main
                main()

            sys.stdout = old_stdout
            output = my_stdout.getvalue()

            # Assert that prefixes are present
            self.assertIn("Prefix: add:", output)
            self.assertIn("Prefix: auth:", output)
            self.assertIn("Prefix: gbad:", output)
            self.assertIn("Prefix: owl:", output)
            self.assertIn("Prefix: rdfs:", output)
            self.assertIn("Prefix: rico:", output)

            # Assert that some known terms from the ontologies are used
            self.assertIn("Types: rico:RecordSet", output)
            self.assertIn("add1:mnemonic", output)
            self.assertIn("auth1:sourceNote", output)
