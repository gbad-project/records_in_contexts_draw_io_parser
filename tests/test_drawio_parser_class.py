"""
Tests for the DrawioParser class.
"""

import unittest
from pathlib import Path
from argparse import Namespace
import unittest.mock
import rdflib

from draw_io_parser import DrawioParser, _arguments_parser

class TestDrawioParserClass(unittest.TestCase):
    """
    Tests the DrawioParser class.
    """

    def _create_mock_rico_ontology_from_hardcoded_data(self):
        """
        Generates a mock rico.rdf file from the original hardcoded lists.
        """
        g = rdflib.Graph()

        prefixes = {
            'rico': 'https://www.ica.org/standards/RiC/ontology#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }

        namespaces = {prefix: rdflib.Namespace(uri) for prefix, uri in prefixes.items()}

        for prefix, namespace in namespaces.items():
            g.bind(prefix, namespace)

        _classes = ["owl:DatatypeProperty", "rico:AccumulationRelation", "rico:Activity", "rico:ActivityDocumentationRelation", "rico:ActivityType", "rico:Agent", "rico:AgentControlRelation", "rico:AgentHierarchicalRelation", "rico:AgentName", "rico:AgentTemporalRelation", "rico:AgentToAgentRelation", "rico:Appellation", "rico:AppellationRelation", "rico:AuthorityRelation", "rico:AuthorshipRelation", "rico:CarrierExtent", "rico:CarrierType", "rico:ChildRelation", "rico:Concept", "rico:ContentType", "rico:Coordinates", "rico:CorporateBody", "rico:CorporateBodyType", "rico:CorrespondenceRelation", "rico:CreationRelation", "rico:Date", "rico:DateType", "rico:DemographicGroup", "rico:DerivationRelation", "rico:DescendanceRelation", "rico:DocumentaryFormType", "rico:Event", "rico:EventRelation", "rico:EventType", "rico:Extent", "rico:ExtentType", "rico:Family", "rico:FamilyRelation", "rico:FamilyType", "rico:FunctionalEquivalenceRelation", "rico:Group", "rico:GroupSubdivisionRelation", "rico:Identifier", "rico:IdentifierType", "rico:Instantiation", "rico:InstantiationExtent", "rico:InstantiationToInstantiationRelation", "rico:IntellectualPropertyRightsRelation", "rico:KnowingOfRelation", "rico:KnowingRelation", "rico:Language", "rico:LeadershipRelation", "rico:LegalStatus", "rico:ManagementRelation", "rico:Mandate", "rico:MandateRelation", "rico:MandateType", "rico:Mechanism", "rico:MembershipRelation", "rico:MigrationRelation", "rico:Name", "rico:OccupationType", "rico:OrganicOrFunctionalProvenanceRelation", "rico:OrganicProvenanceRelation", "rico:OwnershipRelation", "rico:PerformanceRelation", "rico:Person", "rico:PhysicalLocation", "rico:Place", "rico:PlaceName", "rico:PlaceRelation", "rico:PlaceType", "rico:Position", "rico:PositionHoldingRelation", "rico:PositionToGroupRelation", "rico:ProductionTechniqueType", "rico:Proxy", "rico:Record", "rico:RecordPart", "rico:RecordResource", "rico:RecordResourceExtent", "rico:RecordResourceGeneticRelation", "rico:RecordResourceHoldingRelation", "rico:RecordResourceToInstantiationRelation", "rico:RecordResourceToRecordResourceRelation", "rico:RecordSet", "rico:RecordSetType", "rico:RecordState", "rico:Relation", "rico:RepresentationType", "rico:RoleType", "rico:Rule", "rico:RuleRelation", "rico:RuleType", "rico:SequentialRelation", "rico:SiblingRelation", "rico:SpouseRelation", "rico:TeachingRelation", "rico:TemporalRelation", "rico:Thing", "rico:Title", "rico:Type", "rico:TypeRelation", "rico:UnitOfMeasurement", "rico:WholePartRelation", "rico:WorkRelation"]
        _object_properties = ["rdfs:subPropertyOf", "rico:affectsOrAffected", "rico:agentHasOrHadLocation", "rico:authorizedBy", "rico:authorizes", "rico:contained", "rico:containsOrContained", "rico:containsTransitive", "rico:describesOrDescribed", "rico:directlyContains", "rico:directlyFollowsInSequence", "rico:directlyIncludes", "rico:directlyPrecedesInSequence", "rico:documentedBy", "rico:documents", "rico:existsOrExistedIn", "rico:expressesOrExpressed", "rico:followedInSequence", "rico:followsInSequenceTransitive", "rico:followsInTime", "rico:followsOrFollowed", "rico:hadComponent", "rico:hadConstituent", "rico:hadPart", "rico:hadSubdivision", "rico:hadSubevent", "rico:hadSubordinate", "rico:hasAccumulator", "rico:hasActivityType", "rico:hasAddressee", "rico:hasAncestor", "rico:hasAuthor", "rico:hasBeginningDate", "rico:hasBirthDate", "rico:hasBirthPlace", "rico:hasCarrierType", "rico:hasChild", "rico:hasCollector", "rico:hasComponentTransitive", "rico:hasConstituentTransitive", "rico:hasContentOfType", "rico:hasCopy", "rico:hasCreationDate", "rico:hasCreator", "rico:hasDateType", "rico:hasDeathDate", "rico:hasDeathPlace", "rico:hasDescendant", "rico:hasDestructionDate", "rico:hasDirectComponent", "rico:hasDirectConstituent", "rico:hasDirectPart", "rico:hasDirectSubdivision", "rico:hasDirectSubevent", "rico:hasDirectSubordinate", "rico:hasDocumentaryFormType", "rico:hasDraft", "rico:hasEndDate", "rico:hasEventType", "rico:hasExtent", "rico:hasExtentType", "rico:hasFamilyAssociationWith", "rico:hasFamilyType", "rico:hasGeneticLinkToRecordResource", "rico:hasIdentifierType", "rico:hasModificationDate", "rico:hasOrHadAgentName", "rico:hasOrHadAllMembersWithCategory", "rico:hasOrHadAllMembersWithContentType", "rico:hasOrHadAllMembersWithCreationDate", "rico:hasOrHadAllMembersWithDocumentaryFormType", "rico:hasOrHadAllMembersWithLanguage", "rico:hasOrHadAllMembersWithLegalStatus", "rico:hasOrHadAllMembersWithRecordState", "rico:hasOrHadAnalogueInstantiation", "rico:hasOrHadAppellation", "rico:hasOrHadAuthorityOver", "rico:hasOrHadCategory", "rico:hasOrHadType", "rico:hasOrHadComponent", "rico:hasOrHadConstituent", "rico:hasOrHadController", "rico:hasOrHadCoordinates", "rico:hasOrHadCorporateBodyType", "rico:hasOrHadCorrespondent", "rico:hasOrHadDemographicGroup", "rico:hasOrHadDerivedInstantiation", "rico:hasOrHadDigitalInstantiation", "rico:hasOrHadEmployer", "rico:hasOrHadHolder", "rico:hasOrHadIdentifier", "rico:hasOrHadInstantiation", "rico:hasOrHadIntellectualPropertyRightsHolder", "rico:hasOrHadJurisdiction", "rico:hasOrHadLanguage", "rico:hasOrHadLeader", "rico:hasOrHadLegalStatus", "rico:hasOrHadLocation", "rico:hasOrHadMainSubject", "rico:hasOrHadManager", "rico:hasOrHadMandateType", "rico:hasOrHadMember", "rico:hasOrHadMostMembersWithCreationDate", "rico:hasOrHadName", "rico:hasOrHadOccupationOfType", "rico:hasOrHadOwner", "rico:hasOrHadPart", "rico:hasOrHadParticipant", "rico:hasOrHadPhysicalLocation", "rico:hasOrHadPlaceName", "rico:hasOrHadPlaceType", "rico:hasOrHadPosition", "rico:hasOrHadRuleType", "rico:hasOrHadSomeMembersWithCategory", "rico:hasOrHadSomeMembersWithContentType", "rico:hasOrHadSomeMembersWithCreationDate", "rico:hasOrHadSomeMembersWithLanguage", "rico:hasOrHadSomeMembersWithLegalStatus", "rico:hasOrHadSomeMembersWithRecordState", "rico:hasOrHadSomeMemberswithDocumentaryFormType", "rico:hasOrHadSpouse", "rico:hasOrHadStudent", "rico:hasOrHadSubdivision", "rico:hasOrHadSubevent", "rico:hasOrHadSubject", "rico:hasOrHadSubordinate", "rico:hasOrHadTeacher", "rico:hasOrHadTitle", "rico:hasOrHadWorkRelationWith", "rico:hasOrganicOrFunctionalProvenance", "rico:hasOrganicProvenance", "rico:hasOriginal", "rico:hasPartTransitive", "rico:hasProductionTechniqueType", "rico:hasPublicationDate", "rico:hasPublisher", "rico:hasReceiver", "rico:hasRecordSetType", "rico:hasRecordState", "rico:hasReply", "rico:hasRepresentationType", "rico:hasSender", "rico:hasSibling", "rico:hasSubdivisionTransitive", "rico:hasSubeventTransitive", "rico:hasSubordinateTransitive", "rico:hasSuccessor", "rico:hasUnitOfMeasurement", "rico:hasWithin", "rico:included", "rico:includesOrIncluded", "rico:includesTransitive", "rico:intersects", "rico:isAccumulatorOf", "rico:isActivityTypeOf", "rico:isAddresseeOf", "rico:isAgentAssociatedWithAgent", "rico:isAgentAssociatedWithPlace", "rico:isAssociatedWithDate", "rico:isAssociatedWithEvent", "rico:isAssociatedWithPlace", "rico:isAssociatedWithRule", "rico:isAuthorOf", "rico:isBeginningDateOf", "rico:isBirthDateOf", "rico:isBirthPlaceOf", "rico:isCarrierTypeOf", "rico:isChildOf", "rico:isCollectorOf", "rico:isComponentOfTransitive", "rico:isConstituentOfTransitive", "rico:isContainedByTransitive", "rico:isContentTypeOf", "rico:isCopyOf", "rico:isCreationDateOf", "rico:isCreatorOf", "rico:isDateAssociatedWith", "rico:isDateOfOccurrenceOf", "rico:isDateTypeOf", "rico:isDeathDateOf", "rico:isDeathPlaceOf", "rico:isDestructionDateOf", "rico:isDirectComponentOf", "rico:isDirectConstituentOf", "rico:isDirectPartOf", "rico:isDirectSubdivisionOf", "rico:isDirectSubeventOf", "rico:isDirectSubordinateTo", "rico:isDirectlyContainedBy", "rico:isDirectlyIncludedIn", "rico:isDocumentaryFormTypeOf", "rico:isDraftOf", "rico:isEndDateOf", "rico:isEquivalentTo", "rico:isEventAssociatedWith", "rico:isEventTypeOf", "rico:isExtentOf", "rico:isExtentTypeOf", "rico:isFamilyTypeOf", "rico:isFromUseDateOf", "rico:isFunctionallyEquivalentTo", "rico:isIdentifierTypeOf", "rico:isIncludedInTransitive", "rico:isInstantiationAssociatedWithInstantiation", "rico:isLastUpdateDateOf", "rico:isModificationDateOf", "rico:isOrWasAdjacentTo", "rico:isOrWasAffectedBy", "rico:isOrWasAgentNameOf", "rico:isOrWasAnalogueInstantiationOf", "rico:isOrWasAppellationOf", "rico:isOrWasCategoryOf", "rico:isOrWasCategoryOfAllMembersOf", "rico:isOrWasCategoryOfSomeMembersOf", "rico:isOrWasComponentOf", "rico:isOrWasConstituentOf", "rico:isOrWasContainedBy", "rico:isOrWasContentTypeOfAllMembersOf", "rico:isOrWasContentTypeOfSomeMembersOf", "rico:isOrWasControllerOf", "rico:isOrWasCoordinatesOf", "rico:isOrWasCorporateBodyTypeOf", "rico:isOrWasCreationDateOfAllMembersOf", "rico:isOrWasCreationDateOfMostMembersOf", "rico:isOrWasCreationDateOfSomeMembersOf", "rico:isOrWasDemographicGroupOf", "rico:isOrWasDerivedFromInstantiation", "rico:isOrWasDescribedBy", "rico:isOrWasDigitalInstantiationOf", "rico:isOrWasDocumentaryFormTypeOfAllMembersOf", "rico:isOrWasDocumentaryFormTypeOfSomeMembersOf", "rico:isOrWasEmployerOf", "rico:isOrWasEnforcedBy", "rico:isOrWasExpressedBy", "rico:isOrWasHolderOf", "rico:isOrWasHolderOfIntellectualPropertyRightsOf", "rico:isOrWasIdentifierOf", "rico:isOrWasIncludedIn", "rico:isOrWasInstantiationOf", "rico:isOrWasJurisdictionOf", "rico:isOrWasLanguageOf", "rico:isOrWasLanguageOfAllMembersOf", "rico:isOrWasLanguageOfSomeMembersOf", "rico:isOrWasLeaderOf", "rico:isOrWasLegalStatusOf", "rico:isOrWasLegalStatusOfAllMembersOf", "rico:isOrWasLegalStatusOfSomeMembersOf", "rico:isOrWasLocationOf", "rico:isOrWasLocationOfAgent", "rico:isOrWasMainSubjectOf", "rico:isOrWasManagerOf", "rico:isOrWasMandateTypeOf", "rico:isOrWasMemberOf", "rico:isOrWasNameOf", "rico:isOrWasOccupationTypeOf", "rico:isOrWasOccupiedBy", "rico:isOrWasOwnerOf", "rico:isOrWasPartOf", "rico:isOrWasParticipantIn", "rico:isOrWasPerformedBy", "rico:isOrWasPhysicalLocationOf", "rico:isOrWasPlaceNameOf", "rico:isOrWasPlaceTypeOf", "rico:isOrWasRecordStateOfAllMembersOf", "rico:isOrWasRecordStateOfSomeMembersOf", "rico:isOrWasRegulatedBy", "rico:isOrWasResponsibleForEnforcing", "rico:isOrWasRuleTypeOf", "rico:isOrWasSubdivisionOf", "rico:isOrWasSubeventOf", "rico:isOrWasSubjectOf", "rico:isOrWasSubordinateTo", "rico:isOrWasTitleOf", "rico:isOrWasUnderAuthorityOf", "rico:isOrganicOrFunctionalProvenanceOf", "rico:isOrganicProvenanceOf", "rico:isOriginalOf", "rico:isPartOfTransitive", "rico:isPlaceAssociatedWith", "rico:isPlaceAssociatedWithAgent", "rico:isProductionTechniqueTypeOf", "rico:isPublicationDateOf", "rico:isPublisherOf", "rico:isReceiverOf", "rico:isRecordResourceAssociatedWithRecordResource", "rico:isRecordSetTypeOf", "rico:isRecordStateOf", "rico:isRelatedTo", "rico:isReplyTo", "rico:isRepresentationTypeOf", "rico:isResponsibleForIssuing", "rico:isRuleAssociatedWith", "rico:isSenderOf", "rico:isSubdivisionOfTransitive", "rico:isSubeventOfTransitive", "rico:isSubordinateToTransitive", "rico:isSuccessorOf", "rico:isToUseDateOf", "rico:isUnitOfMeasurementOf", "rico:isWithin", "rico:issuedBy", "rico:knownBy", "rico:knows", "rico:knowsOf", "rico:migratedFrom", "rico:migratedInto", "rico:occupiesOrOccupied", "rico:occurredAtDate", "rico:overlapsOrOverlapped", "rico:performsOrPerformed", "rico:precededInSequence", "rico:precedesInSequenceTransitive", "rico:precedesInTime", "rico:precedesOrPreceded", "rico:proxyFor", "rico:proxyIn", "rico:regulatesOrRegulated", "rico:relationHasSource", "rico:relationHasTarget", "rico:resultedFromTheMergerOf", "rico:resultedFromTheSplitOf", "rico:resultsOrResultedFrom", "rico:resultsOrResultedIn", "rico:thingIsSourceOfRelation", "rico:wasComponentOf", "rico:wasConstituentOf", "rico:wasContainedBy", "rico:wasIncludedIn", "rico:wasLastUpdatedAtDate", "rico:wasMergedInto", "rico:wasPartOf", "rico:wasSplitInto", "rico:wasSubdivisionOf", "rico:wasSubeventOf", "rico:wasSubordinateTo", "rico:wasUsedFromDate", "rico:wasUsedToDate"]
        _datatype_properties = ["add:privateNote", "add:notes", "add:relatedMaterial", "add:associatedMaterial", "add:findingAidNote", "add:immediateSourceOfAcquisition", "add:custodialHistory", "add:availabilityOfOtherFormats", "add:accumulationDate", "add:howToOrder", "auth:sourceNote", "auth:functionNote", "auth:privateNote", "rdfs:label", "rico:accruals", "rico:accrualsStatus", "rico:altimetricSystem", "rico:altitude", "rico:authenticityNote", "rico:authorizingMandate", "rico:beginningDate", "rico:birthDate", "rico:carrierExtent", "rico:classification", "rico:conditionsOfAccess", "rico:conditionsOfUse", "rico:creationDate", "rico:date", "rico:dateQualifier", "rico:deathDate", "rico:destructionDate", "rico:endDate", "rico:expressedDate", "rico:generalDescription", "rico:geodesicSystem", "rico:geographicalCoordinates", "rico:height", "rico:history", "rico:identifier", "rico:instantiationExtent", "rico:instantiationStructure", "rico:integrityNote", "rico:lastModificationDate", "rico:latitude", "rico:length", "rico:location", "rico:longitude", "rico:measure", "rico:modificationDate", "rico:name", "rico:normalizedDateValue", "rico:normalizedValue", "rico:physicalCharacteristicsNote", "rico:physicalOrLogicalExtent", "rico:productionTechnique", "rico:publicationDate", "rico:qualityOfRepresentationNote", "rico:quantity", "rico:recordResourceExtent", "rico:recordResourceStructure", "rico:referenceSystem", "rico:relationCertainty", "rico:relationSource", "rico:relationState", "rico:ruleFollowed", "rico:scopeAndContent", "rico:structure", "rico:technicalCharacteristics", "rico:textualValue", "rico:title", "rico:type", "rico:unitOfMeasurement", "rico:usedFromDate", "rico:usedToDate", "rico:width"]

        for qname in _classes:
            if ":" not in qname: continue
            prefix, local_name = qname.split(':', 1)
            if prefix in namespaces:
                g.add((namespaces[prefix][local_name], rdflib.RDF.type, rdflib.RDFS.Class))

        for qname in _object_properties:
            if ":" not in qname: continue
            prefix, local_name = qname.split(':', 1)
            if prefix in namespaces:
                g.add((namespaces[prefix][local_name], rdflib.RDF.type, rdflib.OWL.ObjectProperty))

        for qname in _datatype_properties:
            if ":" not in qname: continue
            prefix, local_name = qname.split(':', 1)
            if prefix in namespaces:
                g.add((namespaces[prefix][local_name], rdflib.RDF.type, rdflib.OWL.DatatypeProperty))

        g.serialize(destination="tests/ontologies/rico.rdf", format="xml")

    def test_parser_with_file(self):
        self.maxDiff = None
        self._create_mock_rico_ontology_from_hardcoded_data()

        drawio_file_path = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
        with open(drawio_file_path, "r", encoding="utf-8") as f:
            drawio_content = f.read()

        golden_file_path = "gbad/schema/description-listings/general_add_descriptions_and_listings_to_ric-o_model_2025-06-20_pz.omn"
        with open(golden_file_path, "r", encoding="utf-8") as f:
            golden_content = f.read()

        parser = DrawioParser()

        output_ontology_iri = "https://data.archives.gov.on.test.gbad.ca/Schema/Mapping"
        argv = [
            "draw_io_parser.py",
            "-m", "url",
            "-c", "none",
            "--infer-types-disable",
            "--label-disable",
            "-o", output_ontology_iri,
            "-x", "", "-p", f"{output_ontology_iri}#",
            "-x", "rico", "-p", "https://www.ica.org/standards/RiC/ontology#",
            "-x", "add", "-p", "https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/",
            "-x", "auth", "-p", "https://data.archives.gov.on.test.gbad.ca/Schema/Authority/",
            "-x", "owl", "-p", "http://www.w3.org/2002/07/owl#",
        ]

        args = _arguments_parser().parse_args(argv[1:])

        original_parse = rdflib.graph.ConjunctiveGraph.parse
        def mock_parse(self, source=None, publicID=None, format=None, location=None, file=None, data=None, **kwargs):
            uri_map = {
                "https://www.ica.org/standards/RiC/ontology/": "tests/ontologies/rico.rdf",
                "http://www.w3.org/2000/01/rdf-schema#": "tests/ontologies/rdfs.rdf",
                "http://www.w3.org/2002/07/owl#": "tests/ontologies/owl.rdf",
                "https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/": "tests/ontologies/add.rdf",
                "https://data.archives.gov.on.test.gbad.ca/Schema/Authority/": "tests/ontologies/auth.rdf",
            }
            if source in uri_map:
                source = uri_map[source]
            return original_parse(self, source=source, publicID=publicID, format=format, location=location, file=file, data=data, **kwargs)

        with unittest.mock.patch('rdflib.graph.ConjunctiveGraph.parse', mock_parse):
            output = parser.run(drawio_content, args)
            self.assertEqual(output.strip(), golden_content.strip())
