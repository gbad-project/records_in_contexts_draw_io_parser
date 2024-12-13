# Parent Commit: bc8d032c2de00b6e9ee3b2dd7f444aa09b901379
# SHA1 Hash at Parent Commit: 91c172dc7d14a0fdc4cd757b910bb3b7f0a2adf4

### Begin logic borrowed from draw_io_parser.py
# The following version was originally copied and pasted to riconvert:
# parser v0.2.1-main.sha.1171107+gbad.17: allow custom types and subproperties
# Commit Hash: fabe7c9c974342b1aa6c255344a6d0bc76feb8d5
# Author of draw_io_parser.py: Richard Williamson @williamsonrichard, here is the original commit (2024-05-13T10:47:23.000Z):
# https://github.com/williamsonrichard/records_in_contexts_draw_io_parser/commit/11711072f11e9159f08f18866e02fffb0e060ae8
# No license indicated for original code at the time of the creation of this

# pylint: disable=too-many-lines

"""
Constructs individuals in OWL with respect to the ontology Records in Contexts
from a draw.io graph.

Intended to be run as a script: the underlying XML of the graph should be sent
into the script via stdin. A number of command line parameters are available for
configuration: run

python draw_io_parser.py --help

for a full list.

Codewise, the DrawIOXMLTree class takes a raw drawio XML string in its
constructor, and exposes only one method, 'individuals_and_arrows', which
returns as a generator all RiC-O individuals and properties (arrows) that it
finds upon parsing the tree, including, for arrows, the data of which
individuals are the source and target of the arrow. Part of the parsing is
already carried out upon calling the constructor, for effectivity.

Two further functions are exposed by the module. The method 'individual_blocks'
takes an iterator of individuals and arrows such as that outputted by the
'individuals_and_arrows' method of a DrawIOXMLTree instance, and assembles them
into a dictionary whose keys are individual IRIs. The value for a given key is
itself a dictionary, collecting together the facts and types for that individual
IRI which were defined by some Individual or Arrow instance in the iterator (the
individual IRI may occur many times in Individual instances with differing
values for the 'class' variable).

The method 'serialise' takes such a dictionary of individuals and their facts
and types, arranges each pair of a key (individual) and its values (facts and
types) into an Individual block in OWL Manchester syntax, and concatenates all
of these into one large string.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field, InitVar
from datetime import datetime
from html.parser import HTMLParser
from sys import exit as sys_exit, stdin
from typing import Generator, Iterator
from xml.etree.ElementTree import Element, fromstring
from typing import Optional
import urllib.parse
import traceback

_prefixes = {
    'rico': 'https://www.ica.org/standards/RiC/ontology#',
    'add': 'https://data.archives.gov.on.ca/Schema/Description-Listings/',
    'auth': 'https://data.archives.gov.on.ca/Schema/Authority/',
    'owl': 'http://www.w3.org/2002/07/owl#'
}

_classes = [
    "owl:DatatypeProperty",
    "rico:AccumulationRelation",
    "rico:Activity",
    "rico:ActivityDocumentationRelation",
    "rico:ActivityType",
    "rico:Agent",
    "rico:AgentControlRelation",
    "rico:AgentHierarchicalRelation",
    "rico:AgentName",
    "rico:AgentTemporalRelation",
    "rico:AgentToAgentRelation",
    "rico:Appellation",
    "rico:AppellationRelation",
    "rico:AuthorityRelation",
    "rico:AuthorshipRelation",
    "rico:CarrierExtent",
    "rico:CarrierType",
    "rico:ChildRelation",
    "rico:Concept",
    "rico:ContentType",
    "rico:Coordinates",
    "rico:CorporateBody",
    "rico:CorporateBodyType",
    "rico:CorrespondenceRelation",
    "rico:CreationRelation",
    "rico:Date",
    "rico:DateType",
    "rico:DemographicGroup",
    "rico:DerivationRelation",
    "rico:DescendanceRelation",
    "rico:DocumentaryFormType",
    "rico:Event",
    "rico:EventRelation",
    "rico:EventType",
    "rico:Extent",
    "rico:ExtentType",
    "rico:Family",
    "rico:FamilyRelation",
    "rico:FamilyType",
    "rico:FunctionalEquivalenceRelation",
    "rico:Group",
    "rico:GroupSubdivisionRelation",
    "rico:Identifier",
    "rico:IdentifierType",
    "rico:Instantiation",
    "rico:InstantiationExtent",
    "rico:InstantiationToInstantiationRelation",
    "rico:IntellectualPropertyRightsRelation",
    "rico:KnowingOfRelation",
    "rico:KnowingRelation",
    "rico:Language",
    "rico:LeadershipRelation",
    "rico:LegalStatus",
    "rico:ManagementRelation",
    "rico:Mandate",
    "rico:MandateRelation",
    "rico:MandateType",
    "rico:Mechanism",
    "rico:MembershipRelation",
    "rico:MigrationRelation",
    "rico:Name",
    "rico:OccupationType",
    "rico:OrganicOrFunctionalProvenanceRelation",
    "rico:OrganicProvenanceRelation",
    "rico:OwnershipRelation",
    "rico:PerformanceRelation",
    "rico:Person",
    "rico:PhysicalLocation",
    "rico:Place",
    "rico:PlaceName",
    "rico:PlaceRelation",
    "rico:PlaceType",
    "rico:Position",
    "rico:PositionHoldingRelation",
    "rico:PositionToGroupRelation",
    "rico:ProductionTechniqueType",
    "rico:Proxy",
    "rico:Record",
    "rico:RecordPart",
    "rico:RecordResource",
    "rico:RecordResourceExtent",
    "rico:RecordResourceGeneticRelation",
    "rico:RecordResourceHoldingRelation",
    "rico:RecordResourceToInstantiationRelation",
    "rico:RecordResourceToRecordResourceRelation",
    "rico:RecordSet",
    "rico:RecordSetType",
    "rico:RecordState",
    "rico:Relation",
    "rico:RepresentationType",
    "rico:RoleType",
    "rico:Rule",
    "rico:RuleRelation",
    "rico:RuleType",
    "rico:SequentialRelation",
    "rico:SiblingRelation",
    "rico:SpouseRelation",
    "rico:TeachingRelation",
    "rico:TemporalRelation",
    "rico:Thing",
    "rico:Title",
    "rico:Type",
    "rico:TypeRelation",
    "rico:UnitOfMeasurement",
    "rico:WholePartRelation",
    "rico:WorkRelation"
]

_object_properties = [
    "rdfs:subPropertyOf",
    "rico:affectsOrAffected",
    "rico:agentHasOrHadLocation",
    "rico:authorizedBy",
    "rico:authorizes",
    "rico:contained",
    "rico:containsOrContained",
    "rico:containsTransitive",
    "rico:describesOrDescribed",
    "rico:directlyContains",
    "rico:directlyFollowsInSequence",
    "rico:directlyIncludes",
    "rico:directlyPrecedesInSequence",
    "rico:documentedBy",
    "rico:documents",
    "rico:existsOrExistedIn",
    "rico:expressesOrExpressed",
    "rico:followedInSequence",
    "rico:followsInSequenceTransitive",
    "rico:followsInTime",
    "rico:followsOrFollowed",
    "rico:hadComponent",
    "rico:hadConstituent",
    "rico:hadPart",
    "rico:hadSubdivision",
    "rico:hadSubevent",
    "rico:hadSubordinate",
    "rico:hasAccumulator",
    "rico:hasActivityType",
    "rico:hasAddressee",
    "rico:hasAncestor",
    "rico:hasAuthor",
    "rico:hasBeginningDate",
    "rico:hasBirthDate",
    "rico:hasBirthPlace",
    "rico:hasCarrierType",
    "rico:hasChild",
    "rico:hasCollector",
    "rico:hasComponentTransitive",
    "rico:hasConstituentTransitive",
    "rico:hasContentOfType",
    "rico:hasCopy",
    "rico:hasCreationDate",
    "rico:hasCreator",
    "rico:hasDateType",
    "rico:hasDeathDate",
    "rico:hasDeathPlace",
    "rico:hasDescendant",
    "rico:hasDestructionDate",
    "rico:hasDirectComponent",
    "rico:hasDirectConstituent",
    "rico:hasDirectPart",
    "rico:hasDirectSubdivision",
    "rico:hasDirectSubevent",
    "rico:hasDirectSubordinate",
    "rico:hasDocumentaryFormType",
    "rico:hasDraft",
    "rico:hasEndDate",
    "rico:hasEventType",
    "rico:hasExtent",
    "rico:hasExtentType",
    "rico:hasFamilyAssociationWith",
    "rico:hasFamilyType",
    "rico:hasGeneticLinkToRecordResource",
    "rico:hasIdentifierType",
    "rico:hasModificationDate",
    "rico:hasOrHadAgentName",
    "rico:hasOrHadAllMembersWithCategory",
    "rico:hasOrHadAllMembersWithContentType",
    "rico:hasOrHadAllMembersWithCreationDate",
    "rico:hasOrHadAllMembersWithDocumentaryFormType",
    "rico:hasOrHadAllMembersWithLanguage",
    "rico:hasOrHadAllMembersWithLegalStatus",
    "rico:hasOrHadAllMembersWithRecordState",
    "rico:hasOrHadAnalogueInstantiation",
    "rico:hasOrHadAppellation",
    "rico:hasOrHadAuthorityOver",
    "rico:hasOrHadCategory",
    "rico:hasOrHadComponent",
    "rico:hasOrHadConstituent",
    "rico:hasOrHadController",
    "rico:hasOrHadCoordinates",
    "rico:hasOrHadCorporateBodyType",
    "rico:hasOrHadCorrespondent",
    "rico:hasOrHadDemographicGroup",
    "rico:hasOrHadDerivedInstantiation",
    "rico:hasOrHadDigitalInstantiation",
    "rico:hasOrHadEmployer",
    "rico:hasOrHadHolder",
    "rico:hasOrHadIdentifier",
    "rico:hasOrHadInstantiation",
    "rico:hasOrHadIntellectualPropertyRightsHolder",
    "rico:hasOrHadJurisdiction",
    "rico:hasOrHadLanguage",
    "rico:hasOrHadLeader",
    "rico:hasOrHadLegalStatus",
    "rico:hasOrHadLocation",
    "rico:hasOrHadMainSubject",
    "rico:hasOrHadManager",
    "rico:hasOrHadMandateType",
    "rico:hasOrHadMember",
    "rico:hasOrHadMostMembersWithCreationDate",
    "rico:hasOrHadName",
    "rico:hasOrHadOccupationOfType",
    "rico:hasOrHadOwner",
    "rico:hasOrHadPart",
    "rico:hasOrHadParticipant",
    "rico:hasOrHadPhysicalLocation",
    "rico:hasOrHadPlaceName",
    "rico:hasOrHadPlaceType",
    "rico:hasOrHadPosition",
    "rico:hasOrHadRuleType",
    "rico:hasOrHadSomeMembersWithCategory",
    "rico:hasOrHadSomeMembersWithContentType",
    "rico:hasOrHadSomeMembersWithCreationDate",
    "rico:hasOrHadSomeMembersWithLanguage",
    "rico:hasOrHadSomeMembersWithLegalStatus",
    "rico:hasOrHadSomeMembersWithRecordState",
    "rico:hasOrHadSomeMemberswithDocumentaryFormType",
    "rico:hasOrHadSpouse",
    "rico:hasOrHadStudent",
    "rico:hasOrHadSubdivision",
    "rico:hasOrHadSubevent",
    "rico:hasOrHadSubject",
    "rico:hasOrHadSubordinate",
    "rico:hasOrHadTeacher",
    "rico:hasOrHadTitle",
    "rico:hasOrHadWorkRelationWith",
    "rico:hasOrganicOrFunctionalProvenance",
    "rico:hasOrganicProvenance",
    "rico:hasOriginal",
    "rico:hasPartTransitive",
    "rico:hasProductionTechniqueType",
    "rico:hasPublicationDate",
    "rico:hasPublisher",
    "rico:hasReceiver",
    "rico:hasRecordSetType",
    "rico:hasRecordState",
    "rico:hasReply",
    "rico:hasRepresentationType",
    "rico:hasSender",
    "rico:hasSibling",
    "rico:hasSubdivisionTransitive",
    "rico:hasSubeventTransitive",
    "rico:hasSubordinateTransitive",
    "rico:hasSuccessor",
    "rico:hasUnitOfMeasurement",
    "rico:hasWithin",
    "rico:included",
    "rico:includesOrIncluded",
    "rico:includesTransitive",
    "rico:intersects",
    "rico:isAccumulatorOf",
    "rico:isActivityTypeOf",
    "rico:isAddresseeOf",
    "rico:isAgentAssociatedWithAgent",
    "rico:isAgentAssociatedWithPlace",
    "rico:isAssociatedWithDate",
    "rico:isAssociatedWithEvent",
    "rico:isAssociatedWithPlace",
    "rico:isAssociatedWithRule",
    "rico:isAuthorOf",
    "rico:isBeginningDateOf",
    "rico:isBirthDateOf",
    "rico:isBirthPlaceOf",
    "rico:isCarrierTypeOf",
    "rico:isChildOf",
    "rico:isCollectorOf",
    "rico:isComponentOfTransitive",
    "rico:isConstituentOfTransitive",
    "rico:isContainedByTransitive",
    "rico:isContentTypeOf",
    "rico:isCopyOf",
    "rico:isCreationDateOf",
    "rico:isCreatorOf",
    "rico:isDateAssociatedWith",
    "rico:isDateOfOccurrenceOf",
    "rico:isDateTypeOf",
    "rico:isDeathDateOf",
    "rico:isDeathPlaceOf",
    "rico:isDestructionDateOf",
    "rico:isDirectComponentOf",
    "rico:isDirectConstituentOf",
    "rico:isDirectPartOf",
    "rico:isDirectSubdivisionOf",
    "rico:isDirectSubeventOf",
    "rico:isDirectSubordinateTo",
    "rico:isDirectlyContainedBy",
    "rico:isDirectlyIncludedIn",
    "rico:isDocumentaryFormTypeOf",
    "rico:isDraftOf",
    "rico:isEndDateOf",
    "rico:isEquivalentTo",
    "rico:isEventAssociatedWith",
    "rico:isEventTypeOf",
    "rico:isExtentOf",
    "rico:isExtentTypeOf",
    "rico:isFamilyTypeOf",
    "rico:isFromUseDateOf",
    "rico:isFunctionallyEquivalentTo",
    "rico:isIdentifierTypeOf",
    "rico:isIncludedInTransitive",
    "rico:isInstantiationAssociatedWithInstantiation",
    "rico:isLastUpdateDateOf",
    "rico:isModificationDateOf",
    "rico:isOrWasAdjacentTo",
    "rico:isOrWasAffectedBy",
    "rico:isOrWasAgentNameOf",
    "rico:isOrWasAnalogueInstantiationOf",
    "rico:isOrWasAppellationOf",
    "rico:isOrWasCategoryOf",
    "rico:isOrWasCategoryOfAllMembersOf",
    "rico:isOrWasCategoryOfSomeMembersOf",
    "rico:isOrWasComponentOf",
    "rico:isOrWasConstituentOf",
    "rico:isOrWasContainedBy",
    "rico:isOrWasContentTypeOfAllMembersOf",
    "rico:isOrWasContentTypeOfSomeMembersOf",
    "rico:isOrWasControllerOf",
    "rico:isOrWasCoordinatesOf",
    "rico:isOrWasCorporateBodyTypeOf",
    "rico:isOrWasCreationDateOfAllMembersOf",
    "rico:isOrWasCreationDateOfMostMembersOf",
    "rico:isOrWasCreationDateOfSomeMembersOf",
    "rico:isOrWasDemographicGroupOf",
    "rico:isOrWasDerivedFromInstantiation",
    "rico:isOrWasDescribedBy",
    "rico:isOrWasDigitalInstantiationOf",
    "rico:isOrWasDocumentaryFormTypeOfAllMembersOf",
    "rico:isOrWasDocumentaryFormTypeOfSomeMembersOf",
    "rico:isOrWasEmployerOf",
    "rico:isOrWasEnforcedBy",
    "rico:isOrWasExpressedBy",
    "rico:isOrWasHolderOf",
    "rico:isOrWasHolderOfIntellectualPropertyRightsOf",
    "rico:isOrWasIdentifierOf",
    "rico:isOrWasIncludedIn",
    "rico:isOrWasInstantiationOf",
    "rico:isOrWasJurisdictionOf",
    "rico:isOrWasLanguageOf",
    "rico:isOrWasLanguageOfAllMembersOf",
    "rico:isOrWasLanguageOfSomeMembersOf",
    "rico:isOrWasLeaderOf",
    "rico:isOrWasLegalStatusOf",
    "rico:isOrWasLegalStatusOfAllMembersOf",
    "rico:isOrWasLegalStatusOfSomeMembersOf",
    "rico:isOrWasLocationOf",
    "rico:isOrWasLocationOfAgent",
    "rico:isOrWasMainSubjectOf",
    "rico:isOrWasManagerOf",
    "rico:isOrWasMandateTypeOf",
    "rico:isOrWasMemberOf",
    "rico:isOrWasNameOf",
    "rico:isOrWasOccupationTypeOf",
    "rico:isOrWasOccupiedBy",
    "rico:isOrWasOwnerOf",
    "rico:isOrWasPartOf",
    "rico:isOrWasParticipantIn",
    "rico:isOrWasPerformedBy",
    "rico:isOrWasPhysicalLocationOf",
    "rico:isOrWasPlaceNameOf",
    "rico:isOrWasPlaceTypeOf",
    "rico:isOrWasRecordStateOfAllMembersOf",
    "rico:isOrWasRecordStateOfSomeMembersOf",
    "rico:isOrWasRegulatedBy",
    "rico:isOrWasResponsibleForEnforcing",
    "rico:isOrWasRuleTypeOf",
    "rico:isOrWasSubdivisionOf",
    "rico:isOrWasSubeventOf",
    "rico:isOrWasSubjectOf",
    "rico:isOrWasSubordinateTo",
    "rico:isOrWasTitleOf",
    "rico:isOrWasUnderAuthorityOf",
    "rico:isOrganicOrFunctionalProvenanceOf",
    "rico:isOrganicProvenanceOf",
    "rico:isOriginalOf",
    "rico:isPartOfTransitive",
    "rico:isPlaceAssociatedWith",
    "rico:isPlaceAssociatedWithAgent",
    "rico:isProductionTechniqueTypeOf",
    "rico:isPublicationDateOf",
    "rico:isPublisherOf",
    "rico:isReceiverOf",
    "rico:isRecordResourceAssociatedWithRecordResource",
    "rico:isRecordSetTypeOf",
    "rico:isRecordStateOf",
    "rico:isRelatedTo",
    "rico:isReplyTo",
    "rico:isRepresentationTypeOf",
    "rico:isResponsibleForIssuing",
    "rico:isRuleAssociatedWith",
    "rico:isSenderOf",
    "rico:isSubdivisionOfTransitive",
    "rico:isSubeventOfTransitive",
    "rico:isSubordinateToTransitive",
    "rico:isSuccessorOf",
    "rico:isToUseDateOf",
    "rico:isUnitOfMeasurementOf",
    "rico:isWithin",
    "rico:issuedBy",
    "rico:knownBy",
    "rico:knows",
    "rico:knowsOf",
    "rico:migratedFrom",
    "rico:migratedInto",
    "rico:occupiesOrOccupied",
    "rico:occurredAtDate",
    "rico:overlapsOrOverlapped",
    "rico:performsOrPerformed",
    "rico:precededInSequence",
    "rico:precedesInSequenceTransitive",
    "rico:precedesInTime",
    "rico:precedesOrPreceded",
    "rico:proxyFor",
    "rico:proxyIn",
    "rico:regulatesOrRegulated",
    "rico:relationHasTarget",
    "rico:resultedFromTheMergerOf",
    "rico:resultedFromTheSplitOf",
    "rico:resultsOrResultedFrom",
    "rico:resultsOrResultedIn",
    "rico:thingIsSourceOfRelation",
    "rico:wasComponentOf",
    "rico:wasConstituentOf",
    "rico:wasContainedBy",
    "rico:wasIncludedIn",
    "rico:wasLastUpdatedAtDate",
    "rico:wasMergedInto",
    "rico:wasPartOf",
    "rico:wasSplitInto",
    "rico:wasSubdivisionOf",
    "rico:wasSubeventOf",
    "rico:wasSubordinateTo",
    "rico:wasUsedFromDate",
    "rico:wasUsedToDate"
]

_datatype_properties = [
    "add:hiddenNotes",
    "auth:sourceNote",
    "auth:functionNote",
    "auth:privateNote",
    "rico:accruals",
    "rico:accrualsStatus",
    "rico:altimetricSystem",
    "rico:altitude",
    "rico:authenticityNote",
    "rico:authorizingMandate",
    "rico:beginningDate",
    "rico:birthDate",
    "rico:carrierExtent",
    "rico:classification",
    "rico:conditionsOfAccess",
    "rico:conditionsOfUse",
    "rico:creationDate",
    "rico:date",
    "rico:dateQualifier",
    "rico:deathDate",
    "rico:destructionDate",
    "rico:endDate",
    "rico:expressedDate",
    "rico:generalDescription",
    "rico:geodesicSystem",
    "rico:geographicalCoordinates",
    "rico:height",
    "rico:history",
    "rico:identifier",
    "rico:instantiationExtent",
    "rico:instantiationStructure",
    "rico:integrityNote",
    "rico:lastModificationDate",
    "rico:latitude",
    "rico:length",
    "rico:location",
    "rico:longitude",
    "rico:measure",
    "rico:modificationDate",
    "rico:name",
    "rico:normalizedDateValue",
    "rico:normalizedValue",
    "rico:physicalCharacteristicsNote",
    "rico:physicalOrLogicalExtent",
    "rico:productionTechnique",
    "rico:publicationDate",
    "rico:qualityOfRepresentationNote",
    "rico:quantity",
    "rico:recordResourceExtent",
    "rico:recordResourceStructure",
    "rico:referenceSystem",
    "rico:relationCertainty",
    "rico:relationSource",
    "rico:relationState",
    "rico:ruleFollowed",
    "rico:scopeAndContent",
    "rico:structure",
    "rico:technicalCharacteristics",
    "rico:textualValue",
    "rico:title",
    "rico:type",
    "rico:unitOfMeasurement",
    "rico:usedFromDate",
    "rico:usedToDate",
    "rico:width"
]

Blocks = dict[tuple[str, str], dict[str, set[str]]]
Cell = Element
CellID = str
XCoordinate = float
YCoordinate = float
Width = float
Height = float
ArrowStart = tuple[XCoordinate, YCoordinate]
ArrowEnd = tuple[XCoordinate, YCoordinate]
Label = str
ArrowData = tuple[Cell, Optional[ArrowStart], Optional[ArrowEnd], Label]
Dimensions = tuple[XCoordinate, YCoordinate, Width, Height]
Paragraph = str
Metacharacter = str
Replacement = str

DEFAULT_CAPITALISATION_SCHEME = "upper-camel"
DEFAULT_INDENTATION = 2
DEFAULT_MAX_GAP = 10
OWL_METACHARACTERS = ["(", ")", "[", "]", "{", "}", "/", ",", ":", ".", "'", '"', ' ', '#']


class NothingToParseException(Exception):
    """
    Can be thrown when calling the constructor of the DrawIOXMLTree class if the
    passed-in XML appears to define an empty graph
    """


class NotInKnownException(Exception):
    """
    Can be thrown if an arrow has a label which does not correspond to an
    object or datatype property in RiC-O, and if it has been specified that this
    is not to be permitted
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class _NoCellCloseEnoughException(Exception):
    pass


class NoSourceException(Exception):
    """
    Can be thrown when calling the 'individuals_and_arrows' function if a given
    arrow has no source that can be identified
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class NoTargetException(Exception):
    """
    Can be thrown when calling the 'individuals_and_arrows' function if a given
    arrow has no target that can be identified
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class _NoValueException(Exception):
    pass


class _SourceNotIndividualException(Exception):
    pass


class ArrowWithoutIndividualAsSourceException(Exception):
    """
    Can be thrown when calling the 'individuals_and_arrows' function if a given
    arrow has a source that appears not to be an individual node
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class _MetacharacterSubstituteParseException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MetacharacterException(Exception):
    """
    Can be thrown when calling the 'individual_blocks' function if an individual
    has an identifier (the text in the upper half of an individual node)
    containing an OWL metacharacter
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class _InvalidCapitalisationSchemeException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ParseException(Exception):
    """
    Can be thrown if the XML being parsed does not have the anticipated
    structure in some respect
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class Individual:
    """
    Represents an OWL individual with type a RiC-O class, coming from a node in
    the parsed graph
    """
    identifier: str
    ric_class: str


@dataclass(frozen=True)
class Arrow:
    """
    Represents an OWL object or datatype property with type a RiC-O class,
    coming from an arrow in the parsed graph
    """
    identifier: str
    source: str
    target: str


class NodeHTMLParser(HTMLParser):
    """
    Subclasses HTMLParser to define its behaviour with respect to 'handle_data',
    'handle_starttag', and 'handle_endtag' (this is the usage pattern expected
    by HTMLParser). It seems that text, including multi-line text, in draw.io
    may come in three forms: as a simple string; as a string within a blockquote
    element; or as a sequence of strings inside divs inside a blockquote. In
    the simple string case, our subclassing of the three afore-mentioned methods
    is such as to discard all information except these strings, and to collect
    them, in the sequence they are encountered in, into a list.

    The 'content' function takes such a list and collects the strings together
    into paragraphs. Single line-breaks in the original graph (corresponding
    usually to three consecutive divs, the middle one of which contains no
    string) are ignored; two or more line-breaks in the original graph will lead
    to a paragraph break.

    The 'clear' function resets the internal state of the class, and should be
    called before parsing a new chunk of HTML.
    """

    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_starttag(self, tag: str, _: list[tuple[str, str | None]]) -> None:
        if tag in ["div", "blockquote", "p", "br"]:
            # Otherwise words stick together in place of a single line break
            self._chunks.append(' ')

    def handle_endtag(self, tag: str) -> None:
        if tag in ["div", "blockquote", "p"]:
            # Otherwise words stick together in place of a single line break
            self._chunks.append(' ')

    def handle_data(self, data: str) -> None:
        """
        Overrides a function in HTMLParser, storing the raw data (text) inside
        a HTML element in the instance variable 'raw_data'.
        """
        # Implementing chunks universally seems to fix lost data with single <br> tags
        self._chunks.append(data)

    def _prettify_linebreaks(self) -> Generator[Paragraph, None, None]:
        # This method is unsafe because can also generate line breaks in Individuals
        previous_was_empty = False
        paragraph_already_handled = False
        current = ""
        for chunk in self._chunks:
            if not chunk:
                if current:
                    yield current
                current = ""
                if previous_was_empty and not paragraph_already_handled:
                    yield "\n\n"
                    paragraph_already_handled = True
                else:
                    previous_was_empty = True
                continue
            current += chunk
            previous_was_empty = False
            paragraph_already_handled = False
        if current:
            yield current

    def content(self) -> str:
        """
        Takes all of the string chunks (within divs and blockquotes) obtained
        during the current run of the parser, and collects them together
        into paragraphs, handling line breaks as described in the docstring
        for this class
        """
        return "".join(self._prettify_linebreaks()).strip()

    def clear(self) -> None:
        """
        Clears the internal state of the parser so that it is as though newly
        constructed
        """
        self._chunks = []


@dataclass(frozen=True)
class SerialisationConfig:
    """
    Holds various user-configurable parameters for configuring the serialisation
    to OWL outputted by the 'serialise' function
    """
    infer_type_of_literals: bool
    include_preamble: bool
    ontology_iri: str | None
    prefix: str | None
    prefix_iri: str | None
    indentation: int
    include_label: bool


@dataclass(frozen=True)
class DrawIOXMLTree:
    """
    The purpose of this class is to parse a raw draw.io XML to a list of
    instances of the Individual and Arrow classes, corresponding respectively to
    nodes and arrows in the graph which the  XML defines. The constructor takes
    such an XML string, and part of the parsing is already carried out upon
    calling the constructor, for effectivity (elements which will be looped
    over are extracted once and for all). The method
    'individuals_and_arrows' can then be called to complete the parsing and
    return the obtained Individual and Arrow instances as a generator
    """
    draw_io_xml_tree: Element = field(init=False)
    literal_node_html_parser: NodeHTMLParser = field(init=False)
    individual_cells: list[
        tuple[Cell, Individual, Dimensions]] = field(init=False)
    arrow_cells: list[ArrowData] = field(init=False)
    literal_cells: list[tuple[Cell, Dimensions]] = field(init=False)

    raw_xml: InitVar[str]

    def __post_init__(self, raw_xml):
        object.__setattr__(self, "literal_node_html_parser", NodeHTMLParser())
        object.__setattr__(self, "draw_io_xml_tree", fromstring(raw_xml))
        object.__setattr__(self, "individual_cells", [])
        object.__setattr__(self, "arrow_cells", [])
        object.__setattr__(self, "literal_cells", [])
        self._extract_individual_and_arrow_and_literal_cells()

    def _cell_with_id(self, _id: str) -> Element:
        cell = self.draw_io_xml_tree.find(f".//*[@id='{_id}']")
        if cell is None:
            raise ValueError(f"No cell with id: {_id}")
        return cell

    def _value_of(self, cell: Element) -> str:
        try:
            value = cell.attrib["value"].strip()
        except KeyError as key_error:
            raise _NoValueException from key_error
        self.literal_node_html_parser.clear()
        self.literal_node_html_parser.feed(value)
        return self.literal_node_html_parser.content()

    def _parent_of(self, cell: Element) -> Element:
        try:
            parent_id = cell.attrib["parent"]
        except KeyError as key_error:
            raise ParseException(
                "Could not parse XML tree: found an 'mxCell' element with "
                "the following id which has value beginning with 'rico:' but "
                f"with no parent: {cell.attrib['id']}"
            ) from key_error
        return self._cell_with_id(parent_id)

    def _child_of(self, parent_id: str) -> Generator[Element, None, None]:
        yield from self.draw_io_xml_tree.findall(f".//*[@parent='{parent_id}']")

    @staticmethod
    def _geometry(cell: Element) -> Element:
        try:
            for element in cell:
                if element.tag == "mxGeometry":
                    return element
        except IndexError as index_error:
            raise ParseException(
                "Expecting the cell with the following id to have an "
                "mxGeometry sub-element, but has no sub-elements at all: "
                f"{cell.attrib['id']}"
            ) from index_error
        raise ParseException(
            "Expecting the cell with the following id to have an mxGeometry "
            f"sub-element: {cell.attrib['id']}")

    @staticmethod
    def _x_and_y_in_geometry(geometry: Element, cell_id: str) -> tuple[
            XCoordinate, YCoordinate]:
        try:
            x = float(geometry.attrib["x"])
        except KeyError as key_error:
            raise ParseException(
                "Encountered an mxGeometry element of the cell with the "
                f"following id without an 'x' attribute: {cell_id}"
            ) from key_error
        try:
            y = float(geometry.attrib["y"])
        except KeyError as key_error:
            raise ParseException(
                "Encountered an mxGeometry element of the cell with the "
                f"following id without a 'y' attribute: {cell_id}"
            ) from key_error
        return x, y

    @staticmethod
    def _has_correct_as_attribute(
            element: Element, as_attribute: str, cell_id: str) -> bool:
        try:
            return element.attrib["as"] == as_attribute
        except KeyError as key_error:
            raise ParseException(
                "Encountered an mxPoint element of the cell with the "
                f"following id without an 'as' attribute: {cell_id}"
            ) from key_error

    @staticmethod
    def _is_locked(cell: Element, as_attribute: str) -> bool:
        if as_attribute == "sourcePoint" and ("source" in cell.attrib):
            return True
        if as_attribute == "targetPoint" and ("target" in cell.attrib):
            return True
        return False

    def _start_or_end(self, cell: Element, as_attribute: str | None) -> tuple[
            XCoordinate, YCoordinate] | None:
        """
        The cell can be part of a group (have another 'parent' than that of the
        top-level graph), in which case the immediate x and y coordinates will
        be relative to the parent in the group rather than absolute; recursion
        is used here to obtain absolute coordinates
        """
        geometry = DrawIOXMLTree._geometry(cell)
        if as_attribute is None:
            return self._x_and_y_in_geometry(geometry, cell.attrib["id"])
        if not geometry:
            raise ParseException(
                "Expecting the mxGeometry element of the cell with the "
                "following id to have sub-elements, but has no sub-elements "
                f"at all: {cell.attrib['id']}")
        for element in geometry:
            if element.tag != "mxPoint" or not self._has_correct_as_attribute(
                    element, as_attribute, cell.attrib["id"]):
                continue
            try:
                x = float(element.attrib["x"])
            except KeyError as key_error:
                if self._is_locked(cell, as_attribute):
                    return None
                raise ParseException(
                    "Encountered an mxPoint element of the cell with the "
                    "following id without an 'x' attribute: "
                    f"{cell.attrib['id']}"
                ) from key_error
            try:
                y = float(element.attrib["y"])
            except KeyError as key_error:
                if self._is_locked(cell, as_attribute):
                    return None
                raise ParseException(
                    "Encountered an mxPoint element of the cell with the "
                    "following id without a 'y' attribute: "
                    f"{cell.attrib['id']}"
                ) from key_error
            parent_id = cell.attrib["parent"]
            if parent_id == "1":
                return x, y
            parent_coordinates = self._start_or_end(
                self._parent_of(cell), None)
            if parent_coordinates is None:
                raise ValueError
            parent_x, parent_y = parent_coordinates
            return x + parent_x, y + parent_y
        raise ParseException(
            "Expecting the mxGeometry element of the cell with the following "
            "id to have an mxPoint sub-element with 'as' attribute having "
            f"value 'sourcePoint', but it does not: {cell.attrib['id']}")

    def _arrow_start(self, arrow_cell: Element) -> ArrowStart | None:
        return self._start_or_end(arrow_cell, "sourcePoint")

    def _arrow_end(self, arrow_cell: Element) -> ArrowEnd | None:
        return self._start_or_end(arrow_cell, "targetPoint")

    @staticmethod
    def _dimensions(individual_cell: Element) -> Dimensions:
        geometry = DrawIOXMLTree._geometry(individual_cell)
        try:
            x = float(geometry.attrib["x"])
        except KeyError as key_error:
            x = 0.0
            #raise ParseException(
            #    "Expecting the mxGeometry element of the cell with the "
            #    "following id to have an 'x' attribute, but it does not: "
            #    f"{individual_cell.attrib['id']}"
            #) from key_error
        try:
            y = float(geometry.attrib["y"])
        except KeyError as key_error:
            y = 0.0
            #raise ParseException(
            #    "Expecting the mxGeometry element of the cell with the "
            #    "following id to have a 'y' attribute, but it does not: "
            #    f"{individual_cell.attrib['id']}"
            #) from key_error
        try:
            width = float(geometry.attrib["width"])
        except KeyError as key_error:
            raise ParseException(
                "Expecting the mxGeometry element of the cell with the "
                "following id to have a 'width' attribute, but it does not: "
                f"{individual_cell.attrib['width']}"
            ) from key_error
        try:
            height = float(geometry.attrib["height"])
        except KeyError as key_error:
            raise ParseException(
                "Expecting the mxGeometry element of the cell with the "
                "following id to have a 'height' attribute, but it does not: "
                f"{individual_cell.attrib['height']}"
            ) from key_error
        return x, y, width, height

    @staticmethod
    def _is_possible_literal(cell: Element) -> bool:
        try:
            if cell.attrib["parent"] != "1":
                return False
            return "rounded=1" in cell.attrib["style"]
        except KeyError:
            return False

    def _arrow_label(self, arrow_cell: Element) -> str:
        for cell in self._child_of(arrow_cell.attrib["id"]):
            try:
                style = cell.attrib["style"]
            except KeyError:
                continue
            if "edgeLabel" in style:
                return self._value_of(cell)
        raise _NoValueException

    def _add_arrow_if_find_label(self, cell: Element) -> None:
        try:
            label = self._arrow_label(cell)
            arrow_data = (
                cell,
                self._arrow_start(cell),
                self._arrow_end(cell),
                label
            )
            self.arrow_cells.append(arrow_data)
        except _NoValueException:
            pass

    def _extract_individual_and_arrow_and_literal_cells(self) -> None:
        try:
            if not self.draw_io_xml_tree[0][0][0]:
                raise NothingToParseException
        except IndexError as key_error:
            raise NothingToParseException from key_error
        for cell in self.draw_io_xml_tree[0][0][0]:
            if cell.tag != "mxCell":
                raise ParseException(
                    "Could not parse XML tree: expecting an element with tag "
                    f"'mxCell', but had tag '{cell.tag}'")
            try:
                cell_value = self._value_of(cell)
            except _NoValueException:
                continue
            if not cell_value:
                self._add_arrow_if_find_label(cell)
                continue
            if not cell_value.split(":")[0] in _prefixes.keys():
                if self._is_possible_literal(cell):
                    self.literal_cells.append((cell, self._dimensions(cell)))
                continue
            try:
                parent = self._parent_of(cell)
                individual_identifier = self._value_of(parent)
            except _NoValueException:
                try:
                    arrow_data = (
                        cell,
                        self._arrow_start(cell),
                        self._arrow_end(cell),
                        cell.attrib["value"]
                    )
                    self.arrow_cells.append(arrow_data)
                except _NoValueException:
                    pass
                continue
            if not individual_identifier:
                continue
            for prefix in _prefixes.keys():
                for ric_class in cell_value.split(f"{prefix}:")[1:]:
                    ric_class = f"{prefix}:" + ric_class.strip()
                    _verify_is_ric_class(ric_class)
                    individual = Individual(individual_identifier, ric_class)
                    self.individual_cells.append(
                        (cell, individual, self._dimensions(parent)))
            #for ric_class in cell_value.split("rico:")[1:]:
            #    ric_class = ric_class.strip()
            #    _verify_is_ric_class(ric_class)
            #    individual = Individual(individual_identifier, ric_class)
            #    self.individual_cells.append(
            #        (cell, individual, self._dimensions(parent)))

    @staticmethod
    def _close_enough(
            arrow_endpoint: ArrowStart | ArrowEnd,
            cell_dimensions: Dimensions,
            max_gap: float) -> bool:
        endpoint_x, endpoint_y = arrow_endpoint
        cell_x, cell_y, cell_width, cell_height = cell_dimensions
        return (
            cell_x - max_gap <= endpoint_x <= cell_x + cell_width + max_gap
        ) and (
            cell_y - max_gap <= endpoint_y <= cell_y + cell_height + max_gap
        )

    def _cell_close_to(
            self,
            arrow_endpoint: ArrowStart | ArrowEnd,
            max_gap: float) -> Element:
        for cell, _, dimensions in self.individual_cells:
            if self._close_enough(arrow_endpoint, dimensions, max_gap):
                return cell
        for cell, dimensions in self.literal_cells:
            if self._close_enough(arrow_endpoint, dimensions, max_gap):
                return cell
        raise _NoCellCloseEnoughException

    def _defines_individual(self, identifier: str) -> bool:
        for _, individual, _ in self.individual_cells:
            if individual.identifier == identifier:
                return True
        return False

    def _source_or_target(
            self,
            source_or_target_cell: Element,
            must_be_individual: bool) -> str:
        try:
            value = self._value_of(source_or_target_cell)
        except KeyError as key_error:
            raise _NoValueException from key_error
        if value.split(":")[0] in _prefixes.keys():
            return self._value_of(self._parent_of(source_or_target_cell))
        if must_be_individual and not self._defines_individual(value):
            raise _SourceNotIndividualException
        return value

    def _arrow(
            self,
            arrow_data: ArrowData,
            strict_mode: bool,
            max_gap: float) -> Arrow:
        arrow_cell, arrow_start, arrow_end, arrow_label = arrow_data
        try:
            source_cell = self._cell_with_id(arrow_cell.attrib["source"])
        except KeyError as key_error:
            if strict_mode or arrow_start is None:
                raise NoSourceException(
                    f"The mxCell element with label '{arrow_label}' and id "
                    f"{arrow_cell.attrib['id']} seems to be an arrow, but its "
                    "source was not able to be determined"
                ) from key_error
            try:
                source_cell = self._cell_close_to(arrow_start, max_gap)
            except _NoCellCloseEnoughException as not_close_enough_exception:
                raise NoSourceException(
                    f"The mxCell element with label '{arrow_label}' and id "
                    f"{arrow_cell.attrib['id']} seems to be an arrow, but its "
                    "source was not able to be determined"
                ) from not_close_enough_exception
        try:
            source = self._source_or_target(source_cell, True)
        except _SourceNotIndividualException as exception:
            raise ArrowWithoutIndividualAsSourceException(
                f"The arrow with id {arrow_cell.attrib['id']} and label "
                f"{arrow_label} has a source which appears not to be a node "
                "defining a RiC-O individual"
            ) from exception
        try:
            target_cell = self._cell_with_id(arrow_cell.attrib["target"])
        except KeyError as key_error:
            if strict_mode or arrow_end is None:
                raise NoSourceException(
                    f"The mxCell element with label '{arrow_label}' and id "
                    f"{arrow_cell.attrib['id']} seems to be an arrow, but its "
                    "target was not able to be determined"
                ) from key_error
            try:
                target_cell = self._cell_close_to(arrow_end, max_gap)
            except _NoCellCloseEnoughException as not_close_enough_exception:
                raise NoSourceException(
                    f"The mxCell element with label '{arrow_label}' and id "
                    f"{arrow_cell.attrib['id']} seems to be an arrow, but its "
                    "target was not able to be determined"
                ) from not_close_enough_exception
        target = self._source_or_target(target_cell, False)
        return Arrow(str(arrow_label.strip()), source, target)

    def individuals_and_arrows(
            self, strict_mode: bool, max_gap: float) -> Generator[
            Individual | Arrow, None, None]:
        """
        Returns as a generator all Individual and Arrow instances obtained
        when parsing the nodes and arrows of the draw.io XML graph fed into the
        DrawIOXMLTree instance upon its construction
        """
        for _, individual, _ in self.individual_cells:
            yield individual
        for arrow_data in self.arrow_cells:
            yield self._arrow(arrow_data, strict_mode, max_gap)


def _verify_is_ric_class(ric_class: str):
    if not ric_class in _classes:
        raise NotInKnownException(f"Not a known class: {ric_class}")


def _handle_spaces(
        identifier: str,
        space_substitute: Replacement,
        capitalisation_scheme: str) -> str:
    if capitalisation_scheme == "upper-camel":
        return f"{space_substitute}".join(
            word[0].upper() + word[1:] for word in identifier.split())
    if capitalisation_scheme == "lower-camel":
        words = identifier.split()
        return f"{space_substitute}".join(
            [words[0][0].lower() + words[0][1:]] + [
                word[0].upper() + word[1:] for word in words[1:]])
    if capitalisation_scheme == "flat":
        return f"{space_substitute}".join(
            word[0].lower() + word[1:] for word in identifier.split())
    if capitalisation_scheme == "none":
        return f"{space_substitute}".join(identifier.split())
    raise ValueError


def _replace_metacharacter(
        metacharacter: str, identifier: str, metacharacter_substitutes: list[
        tuple[Metacharacter, Replacement]]) -> str:
    if metacharacter not in identifier:
        return identifier
    for to_replace, replacement in metacharacter_substitutes:
        if metacharacter == to_replace:
            return identifier.replace(to_replace, replacement)
    raise MetacharacterException(
        f"The following contains the OWL metacharacter '{metacharacter}': "
        f"'{identifier}'. Use the -m/--metacharacter-substitute option, more "
        "than once if necessary, to define a character or string to substitute "
        "it with, or to specify that it should be removed")


def _replace_metacharacters(
        identifier: str,
        metacharacter_substitutes: list[tuple[Metacharacter, Replacement]],
        space_substitute: Replacement | None,
        capitalisation_scheme: str) -> str:
    if ' ' in identifier:
        if space_substitute is None:
            raise MetacharacterException(
                "The following contains a space, but how to handle spaces in "
                "individual nodes has not been specified (spaces cannot be "
                f"used in OWL IRIs): '{identifier}'. Use the "
                "-m/--metacharacter-substitute and -c/--capitalisation-scheme "
                "options to define how to handle spaces")
        identifier = _handle_spaces(
            identifier, space_substitute, capitalisation_scheme)
    elif capitalisation_scheme in ["lower-camel", "flat"]:
        identifier = identifier[0].lower() + identifier[1:]
    for metacharacter in OWL_METACHARACTERS:
        identifier = _replace_metacharacter(
            metacharacter, identifier, metacharacter_substitutes)
    return identifier


def _add_individual_type(
        blocks: Blocks,
        individual: Individual,
        metacharacter_substitutes: list[tuple[Metacharacter, Replacement]],
        space_substitute: Replacement | None,
        capitalisation_scheme: str) -> None:
    individual_id = _replace_metacharacters(
        individual.identifier,
        metacharacter_substitutes,
        space_substitute,
        capitalisation_scheme)
    try:
        block = blocks[(individual_id, individual.identifier)]
    except KeyError:
        blocks[(individual_id, individual.identifier)] = {
            "Types": {individual.ric_class}}
        return
    try:
        block["Types"].add(individual.ric_class)
    except KeyError:
        block["Types"] = {individual.ric_class}


def individual_blocks(
        individuals_and_arrows: Iterator[Individual | Arrow],
        metacharacter_substitutes: list[tuple[Metacharacter, Replacement]],
        space_substitute: Replacement | None,
        capitalisation_scheme: str) -> Blocks:
    """
    Takes an iterator of Individual and Arrow instances, such as that outputted
    by the 'individuals_and_arrows' method of a DrawIOXMLTree instance, and
    assembles them into adictionary whose keys are individual IRIs. The value
    for a given key is itself a dictionary, collecting together the facts and
    types for that individual IRI which were defined by some Individual or Arrow
    instance in the iterator (the individual IRI may occur many times in
    Individual instances with differing values for the 'class' variable).
    """
    blocks: Blocks = {}
    for individual_or_arrow in individuals_and_arrows:
        if isinstance(individual_or_arrow, Individual):
            _add_individual_type(
                blocks,
                individual_or_arrow,
                metacharacter_substitutes,
                space_substitute,
                capitalisation_scheme)
            continue
        if individual_or_arrow.identifier in _object_properties:
            target_identifier = _replace_metacharacters(
                individual_or_arrow.target,
                metacharacter_substitutes,
                space_substitute,
                capitalisation_scheme)
        elif individual_or_arrow.identifier in _datatype_properties:
            target_identifier = individual_or_arrow.target
        else:
            raise NotInKnownException(
                f"An arrow has label '{individual_or_arrow.identifier}', "
                "which is not a known object property or datatype property")
        source_identifier = _replace_metacharacters(
            individual_or_arrow.source,
            metacharacter_substitutes,
            space_substitute,
            capitalisation_scheme)
        try:
            block = blocks[(source_identifier, individual_or_arrow.source)]
        except KeyError:
            blocks[(source_identifier, individual_or_arrow.source)] = {
                individual_or_arrow.identifier: {target_identifier}}
            continue
        try:
            block[individual_or_arrow.identifier].add(target_identifier)
        except KeyError:
            block[individual_or_arrow.identifier] = {target_identifier}
    return blocks


def _infer_type(literal: str) -> str:
    if literal.isnumeric():
        return "\"" + literal + "\"^^xsd:integer"
    try:
        datetime.strptime(literal, "%Y-%m-%d")
        return "\"" + literal + "\"^^xsd:date"
    except ValueError:
        pass
    try:
        if literal[-1] == "Z":
            try:
                datetime.strptime(literal[-1], "%Y-%m-%dT%H-%M-%S")
                return "\"" + literal + "\"^^xsd:dateTime"
            except ValueError:
                pass
        elif literal[-6] == "+" or literal[-6] == "-":
            try:
                datetime.strptime(literal[:-6], "%Y-%m-%dT%H-%M-%S")
                datetime.strptime(literal[-5:], "%H:%M")
                return "\"" + literal + "\"^^xsd:dateTime"
            except ValueError:
                pass
        else:
            try:
                datetime.strptime("%Y-%m-%dT%H-%M-%S", literal)
                return "\"" + literal + "\"^^xsd:dateTime"
            except ValueError:
                pass
    except IndexError:
        # Short literals
        pass
    literal = literal.replace('"', r'\"')
    return "\"" + literal + "\""


def _serialise_facts(
        facts: dict[str, set[str]],
        infer_type_of_literals: bool = True,
        prefix: str | None = None) -> Generator[str, None, None]:
    if prefix:
        prefix_string = prefix + ":"
    else:
        prefix_string = ""
    for _property, values in facts.items():
        for value in sorted(values):
            if _property in _datatype_properties:
                if infer_type_of_literals:
                    formatted_value = _infer_type(value)
                else:
                    formatted_value = "\"" + value + "\""
            else:
                formatted_value = prefix_string + value
            yield f"{_property} {formatted_value}"


def _serialise_block(
        individual_identifier: str,
        individual_label: str,
        types_and_facts: dict[str, set[str]],
        serialisation_config: SerialisationConfig) -> str:
    prefix = serialisation_config.prefix
    indentation = serialisation_config.indentation
    infer_type_of_literals = serialisation_config.infer_type_of_literals
    include_label = serialisation_config.include_label
    if prefix:
        prefix_string = prefix + ":"
    else:
        prefix_string = ""
    if any(str(x).startswith('owl:') for x in types_and_facts["Types"]):
        keyword = 'DataProperty'
    else:
        keyword = 'Individual'
    header = f"{keyword}: {prefix_string}{individual_identifier}"
    if include_label:
        header += f"\n{' '*indentation}Annotations:"
        header += f"\n{' '*(indentation*2)}rdfs:label \"{individual_label}\""
    types_string = ", ".join(
        _type for _type in sorted(types_and_facts["Types"]) if not _type.startswith('owl:'))
        #f"rico:{_type}" for _type in sorted(types_and_facts["Types"]))
    facts = types_and_facts.copy()
    del facts["Types"]
    types_string = f"{' '*indentation}Types: {types_string}" if types_string else ''
    if not facts:
        return f"""{header}
{types_string}

"""
    subproperties = facts.get('rdfs:subPropertyOf', None)
    if subproperties:
        subproperties_string = f"{' '*indentation}SubPropertyOf:\n{' '*(indentation*2)}"
        subproperties_string += f",\n{' '*(indentation*2)}".join(
            subproperty for subproperty in sorted(subproperties))
        return f"""{header}
{subproperties_string}

"""
    serialised_facts = list(_serialise_facts(
        facts, infer_type_of_literals, prefix))
    facts_string = f"{' '*indentation}Facts:\n{' '*(indentation*2)}"
    facts_string += f",\n{' '*(indentation*2)}".join(serialised_facts[:-1])
    facts_string += f",\n{' '*(indentation*2)}{serialised_facts[-1]}" if len(
        serialised_facts) > 1 else f"{serialised_facts[-1]}"
    return f"""{header}
{types_string}
{facts_string}

"""


def _preamble(serialisation_config: SerialisationConfig) -> str:
    ontology_iri = serialisation_config.ontology_iri
    prefix = serialisation_config.prefix
    prefix_iri = serialisation_config.prefix_iri
    indentation = serialisation_config.indentation
    include_label = serialisation_config.include_label
    if ontology_iri:
        ontology_iri_string = ontology_iri
    else:
        current_time = datetime.strftime(datetime.now(), "%Y-%m-%dT%H-%M-%S")
        ontology_iri_string = f"ontology://generated-from-draw-io/{current_time}"
    if prefix:
        prefix_string = prefix
    else:
        prefix_string = ""
    if prefix_iri:
        prefix_iri = f"<{prefix_iri}>"
    else:
        prefix_iri = f"<{ontology_iri_string}#>"
    if include_label:
        preamble = "Prefix: rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    else:
        preamble = ""
    preamble_lines = "\n".join([f"Prefix: {prefix}: <{uri}>" for prefix, uri in _prefixes.items()])

    non_rico_object_properties = [prop for prop in _object_properties if not prop.startswith('rico:')]
    objectproperty_lines = ''
    if len(non_rico_object_properties) > 0:
        for non_rico_object_property in non_rico_object_properties:
            objectproperty_lines += f"\nObjectProperty:\n{' '*indentation}" + non_rico_object_property + "\n"
    
    non_rico_datatype_properties = [prop for prop in _datatype_properties if not prop.startswith('rico:')]
    dataproperty_lines = ''
    if len(non_rico_datatype_properties) > 0:
        for non_rico_datatype_property in non_rico_datatype_properties:
            dataproperty_lines += f"\nDataProperty:\n{' '*indentation}" + non_rico_datatype_property + "\n"
    
    return preamble + f"""{preamble_lines}
Prefix: {prefix_string}: {prefix_iri}
Ontology: <{ontology_iri_string}>
{' '*indentation}Import: <{_prefixes['rico']}>
{objectproperty_lines}{dataproperty_lines}
"""


def serialise(blocks: Blocks, serialisation_config: SerialisationConfig) -> str:
    """
    Takes such a dictionary of individuals and their facts and types such as
    that outputted by the 'individual_blocks' function, arranges each pair of a
    key (individual) and its values (facts and types) into an Individual block
    in OWL Manchester syntax, and concatenates all of these into one large
    string.
    """
    if serialisation_config.include_preamble:
        serialised = _preamble(serialisation_config)
    else:
        serialised = ""
    for individual, types_and_facts in blocks.items():
        individual_id, individual_label = individual
        # Ensure that quotation marks are escaped properly in values for rdfs:label
        individual_label = individual_label.replace('"', r'\"')
        serialised += _serialise_block(
            individual_id,
            individual_label,
            types_and_facts,
            serialisation_config)
    return serialised


def _parse_space_substitute(
        metacharacter_substitutes: list[str]) -> str | None:
    has_remove = False
    has_url = False
    for substitution_definition in metacharacter_substitutes:
        if substitution_definition == "remove":
            has_remove = True
            if not has_url:
                continue
        if substitution_definition == "url":
            has_url = True
            if not has_remove:
                continue
        if substitution_definition[0] != ' ':
            if not has_url:
                continue
        if substitution_definition[1] != "=":
            raise _MetacharacterSubstituteParseException(
                "The second character of a string other than 'remove' or 'url' "
                "passed into the -m/--metadata-substitute option must be '='. This is "
                f"not the case for: {substitution_definition}")
        return substitution_definition.split("=")[1]
    if has_remove:
        return ""
    elif has_url:
        return "%20"
    return None


def _parse_metacharacter_substitutes(
        metacharacter_substitutes: list[str]) -> Generator[
        tuple[Metacharacter, Replacement], None, None]:
    has_remove = False
    has_url = False
    handled = []
    for substitution_definition in metacharacter_substitutes:
        if substitution_definition[0] == ' ':
            continue
        if substitution_definition == "remove":
            has_remove = True
            if not has_url:
                continue
        if substitution_definition == "url":
            has_url = True
            if not has_remove:
                continue
        if substitution_definition[0] not in OWL_METACHARACTERS:
            metacharacters = ', '.join(
                f"'{character}'" for character in OWL_METACHARACTERS)
            raise _MetacharacterSubstituteParseException(
                "The first character of a string other than 'remove' or 'url' "
                "passed into the -m/--metadata-substitute option must be an OWL "
                f"metacharacter, namely one of the following: {metacharacters}"
                f". This is not the case for: {substitution_definition}")
        if substitution_definition[1] != "=":
            raise _MetacharacterSubstituteParseException(
                "The second character of a string other than 'remove' passed "
                "into the -m/--metadata-substitute option must be '='. This is "
                f"not the case for: {substitution_definition}")
        metacharacter, replacement = substitution_definition.split("=", 1)
        handled.append(metacharacter)
        yield metacharacter, replacement
    for metacharacter in OWL_METACHARACTERS:
        if metacharacter not in handled:
            if has_url:
                yield metacharacter, urllib.parse.quote(metacharacter, safe='')
            else:
                yield metacharacter, ""
    if not has_remove:
        return


def _parse_capitalisation_scheme(capitalisation_scheme: str) -> None:
    if capitalisation_scheme not in [
            "upper-camel", "lower-camel", "flat", "none"]:
        raise _InvalidCapitalisationSchemeException(
            "The following was passed into the -c/--capitalisation-scheme "
            f"option, which is not a permitted value: "
            f"{capitalisation_scheme}. See the documentation of the "
            "-c/--capitalisation-scheme option for the permitted values")


def _arguments_parser():
    argument_parser = ArgumentParser(
        description=(
            "Constructs individuals in OWL with respect to the ontology "
            "Records in Contexts from a draw.io graph. The underlying XML of "
            "the graph should be sent into the script via stdin.")
    )
    argument_parser.add_argument(
        "-d",
        "--preamble-disable",
        action="store_true",
        help=(
            "Disable inclusion of a preamble (defining prefix and ontology "
            "IRIs and imports)"))
    argument_parser.add_argument(
        "-g",
        "--max-gap",
        type=float,
        default=DEFAULT_MAX_GAP,
        help=(
            "only taken into account if the '-s/--strict-mode' flag is not "
            "used. In this case, when parsing an arrow whose source or target "
            "is not locked to a node, the geometry of the graph will be taken "
            "into consideration, and a node regarded as the source or target "
            "respectively if the gap (in pixels) between the node and the "
            "start or target of the arrow is less than the max gap defined "
            "here. Can be an integer or a decimal. If not specified, a default "
            f"value of {DEFAULT_MAX_GAP} will be used"))
    argument_parser.add_argument(
        "-i",
        "--infer-types-disable",
        action="store_true",
        help="disable attempted inference of the type of literals")
    argument_parser.add_argument(
        "-n",
        "--indentation",
        type=int,
        default=DEFAULT_INDENTATION,
        help=(
            "the number of spaces to indent by in the outputted OWL syntax. "
            f"If not specified, a default value of {DEFAULT_INDENTATION} will "
            "be used"))
    argument_parser.add_argument(
        "-o",
        "--ontology-iri",
        type=str,
        help=(
            "an IRI to use to define the ontology. By default an IRI, a priori "
            "non-dereferenceable, will be generated, and will include a "
            "current timestamp"))
    argument_parser.add_argument(
        "-p",
        "--prefix-iri",
        type=str,
        help=(
            "an IRI to use with the prefix used for generated individuals, or "
            "the default one if none is specified using the '-x/--prefix' "
            "flag. By default, the ontology IRI will be used with the symbol "
            "'#' appended"))
    argument_parser.add_argument(
        "-s",
        "--strict-mode",
        action="store_true",
        help=(
            "parse arrows in 'strict mode': both the source and the target "
            "must be locked to a node, and no attempt will made to guess them "
            "from the graph geometry if they are not present"))
    argument_parser.add_argument(
        "-x",
        "--prefix",
        type=str,
        help=(
            "a prefix to use with all generated individuals when defining "
            "their IRIs. By default no prefix is used"))
    metacharacters = ', '.join(
        f"'{character}'" for character in OWL_METACHARACTERS)
    argument_parser.add_argument(
        "-m",
        "--metacharacter-substitute",
        type=str,
        nargs='*',
        default=[],
        action="extend",
        help=(
            "defines a substitute for an OWL metacharacter, namely for a space "
            f"character ' ' or one of the following: {metacharacters}. This "
            "option can be used multiple times, for each metacharacter one "
            "wishes to handle. The string passed into the option must "
            "be 'remove' or 'url'; otherwise., the syntax 'c=d' must be used, "
            "where c is the metacharacter and d is its substitute, which can "
            "consist of zero, one, or more characters. The case of zero characters, "
            "that is to say when the syntax reads 'c=', has the effect of simply "
            "removing any occurrence of c. In several cases it will be "
            "necessary to include the quotation marks in the syntax, and "
            "indeed doing so in all cases will not harm. In the special case "
            "of the metacharacter ' ', that is to say a space, any consecutive "
            "chain of spaces will be treated as one, i.e. the entire chain "
            "will be replaced by the character/string d or removed. If the "
            "special string 'remove' is used, all metacharacters will simply "
            "be removed except for those for which a replacement has been "
            "defined by means of a separate use of the "
            "-m/--metacharacter-substitute option. "
            "If the special string 'url' is used, all metacharacters will simply "
            "be replaced with corresponding URL entities except for those "
            "for which a replacement has been defined by means of a separate use "
            "of the -m/--metacharacter-substitute option."))
    argument_parser.add_argument(
        "-l",
        "--label-disable",
        action="store_true",
        help=(
            "disable the inclusion in the outputted OWL individual blocks of "
            "an rdfs:label annotation property recording the original text "
            "in a node of the graph from which the IRI of the individual is "
            "constructed (if spaces and other metacharacters are present in "
            "the original text, these will need to be handled by means of the "
            "-m/--metacharacter-substitute and -c/--capitalisation-scheme "
            "options"))
    argument_parser.add_argument(
        "-c",
        "--capitalisation-scheme",
        type=str,
        default=DEFAULT_CAPITALISATION_SCHEME,
        help=(
            "spaces are not permitted in OWL individual IRIs, and thus a "
            "choice of how to separate multiple words is needed. The "
            "-m/--metacharacter-substitute option allows for specification "
            "of which character or string to replace spaces by, or whether "
            "to simply remove spaces. The option documented here allows in "
            "addition for adjusting the capitalisation of the words now "
            "combined by the replacing of/removal of spaces. The option "
            "accepts one of the following strings: 'upper-camel', "
            "'lower-camel', 'flat', 'none', of which the default is "
            "'upper-camel'. Here 'upper-camel' capitalises the first letter "
            "of every word; 'lower-camel' capitalises the first letter of "
            "every word except the first, which is made lower-case; 'flat' "
            "makes every word lower-case; and 'none' leaves the words "
            "untouched"))
    return argument_parser


def _run() -> None:
    arguments = _arguments_parser().parse_args()
    serialisation_config = SerialisationConfig(
        infer_type_of_literals=not arguments.infer_types_disable,
        include_preamble=not arguments.preamble_disable,
        ontology_iri=arguments.ontology_iri,
        prefix=arguments.prefix,
        prefix_iri=arguments.prefix_iri,
        indentation=arguments.indentation,
        include_label=not arguments.label_disable)
    max_gap = arguments.max_gap
    strict_mode = arguments.strict_mode
    capitalisation_scheme = arguments.capitalisation_scheme
    try:
        space_substitute = _parse_space_substitute(
            arguments.metacharacter_substitute)
        metacharacter_substitutes = list(_parse_metacharacter_substitutes(
            arguments.metacharacter_substitute))
        _parse_capitalisation_scheme(capitalisation_scheme)
    except (
            _MetacharacterSubstituteParseException,
            _InvalidCapitalisationSchemeException) as exception:
        sys_exit(f"{exception}")
    try:
        draw_io_xml_tree = DrawIOXMLTree(stdin.read())
    except NothingToParseException:
        sys_exit("The draw IO XML graph passed in appears to be empty")
    except NotInKnownException as exception:
        sys_exit(f"{exception}")
    try:
        blocks = individual_blocks(
            draw_io_xml_tree.individuals_and_arrows(strict_mode, max_gap),
            metacharacter_substitutes,
            space_substitute,
            capitalisation_scheme)
    except NoSourceException as exception:
        if arguments.strict_mode:
            message = (
                f"{exception}. If so, try to lock the arrow to an individual "
                "node in the original graph; or the underlying XML could be "
                "edited to indicate the source. Alternatively, try running the "
                "parser in non-strict mode (without the '-s/--strict-mode' "
                "flag), optionally making use of the '-g/--max-gap' option")
        else:
            message = (
                f"{exception}. If so, consider using the '-g/--max gap' option "
                "when running the script to increase the max recognised gap "
                "between a node and an arrow end; or try to lock the arrow to "
                "an individual node in the original graph; or the underlying "
                "XML could be edited")
        sys_exit(message)
    except (
            NotInKnownException,
            ArrowWithoutIndividualAsSourceException,
            MetacharacterException) as exception:
        sys_exit(f"{exception}")
    print(serialise(blocks, serialisation_config).rstrip())

def draw_io_parser_run(args) -> None:
    # Skip the first argument (input file) and pass the rest to the argument parser
    arguments = _arguments_parser().parse_args(args[1:])
    serialisation_config = SerialisationConfig(
        infer_type_of_literals=not arguments.infer_types_disable,
        include_preamble=not arguments.preamble_disable,
        ontology_iri=arguments.ontology_iri,
        prefix=arguments.prefix,
        prefix_iri=arguments.prefix_iri,
        indentation=arguments.indentation,
        include_label=not arguments.label_disable)
    max_gap = arguments.max_gap
    strict_mode = arguments.strict_mode
    capitalisation_scheme = arguments.capitalisation_scheme
    try:
        space_substitute = _parse_space_substitute(
            arguments.metacharacter_substitute)
        metacharacter_substitutes = list(_parse_metacharacter_substitutes(
            arguments.metacharacter_substitute))
        _parse_capitalisation_scheme(capitalisation_scheme)
    except (
            _MetacharacterSubstituteParseException,
            _InvalidCapitalisationSchemeException) as exception:
        sys_exit(f"{exception}")
    try:
        with open(args[0], 'r') as f:
            draw_io_xml_tree = DrawIOXMLTree(f.read())
    except NothingToParseException:
        sys_exit("The draw IO XML graph passed in appears to be empty")
    except NotInKnownException as exception:
        sys_exit(f"{exception}")
    try:
        blocks = individual_blocks(
            draw_io_xml_tree.individuals_and_arrows(strict_mode, max_gap),
            metacharacter_substitutes,
            space_substitute,
            capitalisation_scheme)
    except NoSourceException as exception:
        if arguments.strict_mode:
            message = (
                f"{exception}. If so, try to lock the arrow to an individual "
                "node in the original graph; or the underlying XML could be "
                "edited to indicate the source. Alternatively, try running the "
                "parser in non-strict mode (without the '-s/--strict-mode' "
                "flag), optionally making use of the '-g/--max-gap' option")
        else:
            message = (
                f"{exception}. If so, consider using the '-g/--max gap' option "
                "when running the script to increase the max recognised gap "
                "between a node and an arrow end; or try to lock the arrow to "
                "an individual node in the original graph; or the underlying "
                "XML could be edited")
        sys_exit(message)
    except (
            NotInKnownException,
            ArrowWithoutIndividualAsSourceException,
            MetacharacterException) as exception:
        sys_exit(f"{exception}")
    print(serialise(blocks, serialisation_config).rstrip())
    
def _main() -> None:
    try:
        _run()
    except ParseException as exception:
        sys_exit(str(exception))
    except Exception as exception:  # pylint: disable=broad-exception-caught
        error_type = type(exception).__name__
        error_traceback = traceback.format_exc()
        sys_exit(f"An unexpected error occurred: {error_type}: {exception}\n\nTraceback:\n{error_traceback}")

def draw_io_parser_main():
    _main()
### End logic borrowed from draw_io_parser.py

### Begin logic from map_schema.py
# The following version was originally copied and pasted to riconvert:
# Fix parent commit where triples with custom predicates would not be mapped
# Commit Hash: eeb77846da811d08c8ca5ec9ff44231cac00862e
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
import uuid

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
    'CorporateBody': ('Agent', r'/(Corporate Name|[ABC] Ontario Government Name)/'),
    'Family': ('Agent', r'/(Family Name)/'),
    'Place': ('Place', r'/(Geographic Name)/'),
    'Person': ('Agent', r'/(Personal Name)/')
}
uuid_label = 'UUID'
triplesmap_pattern = r'[^0-9a-z_-]'

triplesmap_label = 'TriplesMap'
uriref_str_label = 'uriref_str'
map_predicate_label = 'map_predicate'
map_object_label = 'map_object'
increment_number_label = 'increment_number'

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

def map_schema_init(graph_path, csv_path, output_dir, sanitized_filename, schema_code='auth', source_filename=None):
    # Define GBAD schema ontology
    base_data_uri = 'https://data.archives.gov.on.ca'
    #base_gbad_uri = URIRef(f"{base_data_uri}/RiC-O_1-0-1")
    base_gbad_uri = base_data_uri
    base_schema_uri = URIRef(f"{base_data_uri}/Schema")
    #base_kb_uri = URIRef(f"{base_data_uri}/KB")
    base_auth_uri = URIRef(f"{base_schema_uri}/Authority")
    base_add_uri = URIRef(f"{base_schema_uri}/Description-Listings")
    base_mapping_uri = URIRef(f"{base_schema_uri}/Mapping")
    MAPPING_NS_UUID = uuid.uuid5(uuid.NAMESPACE_URL, f"{base_mapping_uri}#")
    print(f"Namespace UUID v5 for <{base_mapping_uri}#>: {MAPPING_NS_UUID}\n")

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
    schema_regex_str = rf'^({auth_term}|{add_term}|{maps_term})/([A-Za-z_]+)(#.*|/.*)?$'
    schema_regex = re.compile(schema_regex_str, flags=re.IGNORECASE)

    # Any mnemonic-based URIs in GBAD URI syntax
    mnemonic_pattern = r"\{([A-Z:_\d\.]+)\}"
    mnemonic_regex = re.compile(rf"([a-zA-Z]+)/({mnemonic_pattern})/?(.*)")
    # Camel case separation
    camelcase_pattern = r"(?<=[a-z])(?=[A-Z])"
    camelcase_regex = re.compile(camelcase_pattern)
    def decamelize(s): s = camelcase_regex.sub(' ', s); return s[0].upper() + s[1:] # uppercase custom properties
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
    # Define UUID replacement logic - support any prefix or suffix (to diversify UUIDs) but only allowed chars
    uuid_pattern = f"{{({uuid_label}({triplesmap_pattern}*)|({triplesmap_pattern}*){uuid_label})}}" # unencoded curly brackets because regex applied before encoding
    uuid_regex = re.compile(uuid_pattern, flags=re.IGNORECASE) # let's make it case-insensitive to allow flexibility for different URI formats
    def generate_uuid_str(entity_name, show_message=True):
        uuid_str = str(uuid.uuid5(MAPPING_NS_UUID, entity_name))
        if show_message:
            print(f"UUID v5 generated from namespace and '{entity_name}': {uuid_str}")
        return uuid_str

    def substitute_uuid(uriref, uuid_str):
        uriref = uuid_regex.sub(uuid_str, uriref)
        return URIRef(uriref) if isinstance(uriref, URIRef) else Literal(uriref)

    def prettify_rdfs_label(literal_str):
        # Make sure no encoded chars remain, in particular those can come from rr:constant
        literal_str = urllib.parse.unquote(literal_str)

        # Remove base data prefix
        if literal_str.startswith(base_uri_prefix):
            literal_str = str(literal_str[len(base_uri_prefix):])

        # Schema entities
        if literal_str.lower().startswith(schema_term.lower() + '/'):
            literal_str = str(literal_str[len(schema_term)+1:])
            match = schema_regex.search(literal_str)
            if match:
                literal_str = match.group(0)
                #auth_add_maps_term = match.group(1)
                rico_class_ish_term = decamelize(match.group(2))
                last_term =  match.group(3)
                if last_term is None:
                    literal_str = rico_class_ish_term
                elif str(last_term).startswith('#'):
                    literal_str = f"{rico_class_ish_term}: {last_term[1:]}"
                elif str(last_term).startswith('/'):
                    literal_str = last_term[1:]
                else:
                    pass
                #literal_str = str(literal_str[len(auth_add_maps_term)+1:])
                #literal_str = literal_str + f' ({auth_add_maps_term} Schema Entity)'

        # KB entities
        if literal_str.lower().startswith(kb_term.lower() + '/'):
            literal_str = str(literal_str[len(kb_term)+1:])
            #literal_str = literal_str + ' (Knowledge Base Entity'
            match = re.search(mnemonic_regex, literal_str)
            if match:
                rico_ish_class = decamelize(match.group(1))
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

    # Block that overrides everything for riconvert
    graph_path = graph_path
    source_path = csv_path
    ttl_to_rml_prefix = "rml_from_drawio_turtle_"
    rml_path = os.path.join(output_dir, f"{ttl_to_rml_prefix}{sanitized_filename}.rml")

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
    auth = ('auth', Namespace(URIRef(f"{base_auth_uri}/")))
    add = ('add', Namespace(URIRef(f"{base_add_uri}/")))

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
    g.namespace_manager.bind(*auth)
    g.namespace_manager.bind(*add)
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
            for authtp_rico_class, rico_authtp_tuple in rico_authtp_dict.items():
                rico_authtp_uri_term, authtp_value = rico_authtp_tuple
                # If contains {RICO_AUTHTP}
                # Add two subjects for easy separate triplesmap creation later on
                for authtp_i in [1, 2]:
                    authtp_column_name = f"{auth_authtp_label}_{authtp_i}"
                    replacement = f"{authtp_rico_class}_{authtp_column_name}" # class is used for uniqueness
                    rico_disaggregated_subject_uri = str(subject_uri).replace(subject_mask, replacement)

                    if isinstance(subject_uri, URIRef):
                        rico_disaggregated_subject_uri = URIRef(rico_disaggregated_subject_uri)
                    else:
                        rico_disaggregated_subject_uri = Literal(rico_disaggregated_subject_uri)
                    rico_disaggregated_subjects.append(rico_disaggregated_subject_uri)
                    # Keep an external list of these for future use
                    if not rico_disaggregated_subject_uri in rico_authtp_subjects.keys():
                        true_rico_disaggregated_subject_uri = str(subject_uri).replace(subject_mask, rico_authtp_uri_term) # actual term
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
                for authtp_rico_class, rico_authtp_tuple in rico_authtp_dict.items():
                    rico_authtp_uri_term, authtp_value = rico_authtp_tuple
                    # If contains {RICO_AUTHTP}
                    # Add two subjects for easy separate triplesmap creation later on
                    for authtp_i in [1, 2]:
                        authtp_column_name = f"{auth_authtp_label}_{authtp_i}"
                        replacement = f"{authtp_rico_class}_{authtp_column_name}" # class is used for uniqueness
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
    allowed_non_rico_classes = [
        owl[1].DatatypeProperty
    ]
    subjects_df = parsed_df[
        (parsed_df['predicate'].apply(lambda x: str(normalize_uri(x, g.namespace_manager))) == 'rdf:type') &
        (parsed_df['object'].apply(lambda x:
                                   (str(normalize_uri(x, g.namespace_manager)).startswith(f"{rico[0]}:") or
                                    x in allowed_non_rico_classes)))
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
        triplesmap_name = re.sub(triplesmap_pattern, '_', str, flags=re.IGNORECASE)
        return triplesmap_name

    def generate_triplesmap_name(row, show_message=False):
        def series(triplesmap_name, uuid_str):
            map_series = pd.Series({
                triplesmap_label: triplesmap_name,
                uuid_label: uuid_str
            })
            return map_series
        
        # This implementation assumes that subject URIs are unique
        subject_str = row[uriref_str_label]
        cleaned_subject = triplesmap_clean(subject_str)
        # Not showing message because we would only need to see it for rows with {UUID},
        # and this is implemented in UUID substitution logic
        uuid_str = generate_uuid_str(cleaned_subject, show_message=show_message)
        return series(cleaned_subject, uuid_str)
    
    # Necessary to init namespace manager for uriref_str_to_map
    # Initialize an RDF graph
    mapping = Graph(base = URIRef(f"{base_gbad_uri}/"))
    
    def uriref_str_to_map(uriref_str, uuid_str=None):
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
            # This is where the actual UUID substitution happens when UUID accompanies mnemonics
            if uuid_str:
                cleaned_uri = substitute_uuid(cleaned_uri, uuid_str)
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
            # This is where the actual UUID substitution happens when UUID is the only mask
            if uuid_str:
                cleaned_uri = substitute_uuid(cleaned_uri, uuid_str)
            encoded_uri = URIRef(urllib.parse.quote(cleaned_uri, safe=":/#?&="))
            if isinstance(encoded_uri, URIRef): # check if true URI or constant literal
                map_object = URIRef(encoded_uri)
            else:
                map_object = Literal(cleaned_uri)
        # Treat anything else as a literal
        else:
            map_object = Literal(uriref_str)

        return series(map_predicate, map_object)
    
    def extract_mnemonic(row):
        map_predicate = row[map_predicate_label]
        map_object = row[map_object_label]
        uuid_str = row.get(uuid_label, None)
        if map_object:
            if map_predicate == rml[1].reference:
                return map_object
            elif map_predicate != rr[1].template:
                return None
            # Consider replacing this with more robust, findall logic
            # later on to allow for true multiple masks
            #map_object = substitute_rico_version_mask(map_object)
            if uuid_str:
                map_object = substitute_uuid(map_object, uuid_str)
            matches = re.findall(mnemonic_pattern, map_object)
            if matches:
                if len(matches) > 1:
                    # If there is a predicate column set in row, then we are iterating over parsed_df objects,
                    # which means we already saw the warning when iterating over disaggregated subjects_df
                    # Then if there is no triplesmap name set, then this is subjects_df before disaggregation,
                    # and we do not want to see the warning yet because URIs are not yet final
                    show_warning = (row.get('predicate', None) is None and
                                    row.get(triplesmap_label, None) is not None)
                    if show_warning:
                        other_mnemonics = ", ".join([f"{{{match}}}" for match in matches[1:]])
                        if uuid_str:
                            generate_triplesmap_name(row, show_message=True)  # just to show the message
                        print("At most one rr:template is allowed per subject map ",
                            f"whereas multiple are given in: '{map_object}'. ",
                            f"By default logic, the leftmost mnemonic is deliberately chosen as the main one.",
                            f"Thus, {{{matches[0]}}} will be processed as the main mnemonic, "
                            f"and all the others will be passed to RML as is: {other_mnemonics}", "\n")
                    #return None
                return matches[0]
        return None
    
    def generate_rico_name(row):
        object_uri = row['object']
        object_str = str(normalize_uri(object_uri, g.namespace_manager))
        cleaned_object = object_str
        return cleaned_object
    
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

        # If increment number in row, then disaggregation already done (e.g., for subject)...
        increment_number = row.get(increment_number_label, None)
        if isinstance(increment_number, int): # ...so will only generate one row with the inherited increment number (e.g., for object)
            mnemonic_i_from = increment_number; mnemonic_i_to = increment_number
        for mnemonic_i in range(mnemonic_i_from, mnemonic_i_to + 1):
            new_row = row.copy()
            new_row[increment_number_label] = mnemonic_i
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
    # Mnemonic is necessary for disaggregation logic that follows
    subjects_df[mnemonic_label] = subjects_df.apply(extract_mnemonic, axis=1)\
    
    # Now that we have mnemonics generated, let's honor any increment requests
    disaggregated_subject_rows = []
    def collect_incremented_subject_uri(row): return collect_incremented_uri(row, 'subject', disaggregated_subject_rows)
    subjects_df = subjects_df.apply(collect_incremented_subject_uri, axis=1)
    # Creating new frame so that there is no duplication wih previous
    subjects_df = pd.DataFrame(disaggregated_subject_rows)
    # Let's regenerate cols above for simplicity now that rows are disaggregated
    subjects_df[uriref_str_label] = subjects_df['subject'].apply(extract_uriref_str)
    # Knowing TriplesMap name is necessary for UUID substitution at uriref_str_to_map and mnemonic extraction
    subjects_df[[triplesmap_label, uuid_label]] = subjects_df.apply(generate_triplesmap_name, axis=1)
    subjects_df[[map_predicate_label, map_object_label]] = subjects_df.apply(lambda row: uriref_str_to_map(row[uriref_str_label], row[uuid_label]), axis=1)
    subjects_df[mnemonic_label] = subjects_df.apply(extract_mnemonic, axis=1)
    # Now that all cols have been disaggregated, let's generate remaining useful cols
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
    parsed_df = pd.merge(parsed_df, subjects_df[['original_subject', 'subject', rico_name_label, increment_number_label]], on='original_subject', how='left')
    # The below line is necessary because np.nan in merged df force this col into float
    parsed_df[increment_number_label] = parsed_df[increment_number_label].astype('Int64')
    #parsed_df[increment_number_label] = parsed_df[increment_number_label].astype(int)  # Convert to int
    #print(parsed_df[increment_number_label])
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
    # TriplesMap string will be necessary later on to filter out non-matching objects
    parsed_df[[triplesmap_label, uuid_label]] = parsed_df.apply(generate_triplesmap_name, axis=1)
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
    mapping.namespace_manager.bind(*auth)
    mapping.namespace_manager.bind(*add)
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
            true_subject_po = uriref_str_to_map(extract_uriref_str(true_subject_uri),
                                                generate_uuid_str(triplesmap_name,
                                                                  show_message=False)) # already saw these UUIDs
            subject_map_predicate = true_subject_po[map_predicate_label]
            uri_mask = true_subject_po[map_object_label]
        else:
            subject_map_predicate = subject_row[map_predicate_label]
            uri_mask = subject_row[map_object_label]
        #URIRef(urllib.parse.unquote(str(subject)))
        #uri_mask = construct_uri_mask(subjects_df, i)
        
        # Define an empty Subject Map
        subject_map = BNode()
        mapping.add((triples_map, rr[1].subjectMap, subject_map))

        # Remove prefix from RiC-O name from subject df and add to graph
        rico_name = subject_row[rico_name_label]
        rico_class = rico_name.split(':')[1]
        class_uri = rico[1][rico_class]
        for non_rico_class_uri in allowed_non_rico_classes:
            if rico_name == str(normalize_uri(non_rico_class_uri, mapping.namespace_manager)):
                class_uri = non_rico_class_uri; break
        # So this adds the rdf:type definition
        mapping.add((subject_map, rr[1]['class'], class_uri))

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
                rico_authtp_uri_term, authtp_value = rico_authtp_dict[rico_class]
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
            is_auth = (norm_predicate.startswith(f"{auth[0]}:"))
            is_add = (norm_predicate.startswith(f"{add[0]}:"))
            if (is_rico | is_rdfs | is_auth | is_add):
                # Now we can actually iterate over objects
                object = parsed_result['object']
                original_object = parsed_result['original_object']
                if object in rico_authtp_subjects.keys(): # checking if the object is a subject among rico_authtp_subjects
                    true_object_uri, authtp_column_name = rico_authtp_subjects[object]
                    object_triplesmap_name = parsed_result[triplesmap_label]
                    true_object_po = uriref_str_to_map(extract_uriref_str(true_object_uri),
                                                        generate_uuid_str(object_triplesmap_name,
                                                                          show_message=False)) # already saw these UUIDs
                    object_map_predicate = true_object_po[map_predicate_label]
                    object_map_object = true_object_po[map_object_label]
                else:
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

                    # This concerns only constant literals, meaning nodes
                    # in drawio graph for which no mapping logic is defined
                    if not object_map_predicate:
                        # So these are simply added as predicate and object, no predicate-object map
                        if object_map_object: # sometimes it may be empty
                            mapping.add((object_map, rr[1].constant, object_map_object)) 
                        else:
                            mapping.add((object_map, rr[1].constant, Literal(object))) # point to constant URI
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

#def map_schema_main():
#    parser = argparse.ArgumentParser(description="Map schema of choice")
#    parser.add_argument("schema", help="Choose one: add or auth.")
#    parser.add_argument("source", nargs='?', help="Filename of source CSV without extension. Defaults to the head=6 version for chosen schema.")
#
#    args = parser.parse_args()
#
#    __init__(str(args.schema).lower(), args.source)
### End logic from map_schema.py

### Begin logic from map_rml.py
# The following version was originally copied and pasted to riconvert:
# Tweak/fix general ADD somewhat more plus ->
# Commit Hash: 9cea34bbed043c607cfcc27d586fa52ec61aebb6

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

def map_rml(script_dir, output_dir, schema_code='auth'):
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

    # Override for riconvert
    rml_dir = output_dir
    ttl_root = output_dir
    rmlmapper_dir = script_dir

    # Find the .rml file
    rml_files = glob.glob(os.path.join(os.path.normpath(rml_dir), "*.rml"))
    rmlmapper_files = glob.glob(os.path.join(os.path.normpath(rmlmapper_dir), "rmlmapper*"))

    return_tuple = (None, None, None)
    if (rml_files and rmlmapper_files):
        rml = rml_files[0]  # Assuming you want the first .rml file found
        rmlmapper = rmlmapper_files[0] # Same assumption for mapper jar
        rml_filename = os.path.splitext(os.path.basename(rml))[0]

        # Create target directory if it does not exist
        ttl_dir = os.path.join(os.path.normpath(ttl_root), f"mapped_from_{rml_filename}")
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
                        large_ttl = os.path.join(ttl_dir, mapped_large_filename)
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
            print(f"Failed to run mapper jar: '{e}'")
    else:
        print("No .rml and/or mapper files found in specified paths.")
    
    return return_tuple
    
def postprocess(graph_path):
    # Create the input RDF graph
    base_uri = 'https://data.archives.gov.on.ca'
    base_kb_uri = URIRef(f"{base_uri}/KB")
    base_schema_uri = URIRef(f"{base_uri}/Schema")
    base_auth_uri = URIRef(f"{base_schema_uri}/Authority")
    base_add_uri = URIRef(f"{base_schema_uri}/Description-Listings")
    base_mapping_uri = URIRef(f"{base_schema_uri}/Mapping")
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

    total_count = 0
    def print_total_count(): print(f"\nNumber of triples in the graph: {total_count}")

    try:    
        g.parse(graph_path,
                format=format)
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

    def remove_false_agentcontrolrelation(g):
        # Parametrized query to find all rico:AgentControlRelation instances that are not
        # objects of rico:thingIsSourceOfRelation (empty, false entities generated from
        # drawio logic), and remove any triples where these are subjects or objects
        triples_to_remove = []
        removed_graph = Graph()
        removed_list_path = os.path.join(os.path.dirname(graph_path), 'removed_triples.nt')
        for s, p, o in g.triples((None, RDF.type, rico[1].AgentControlRelation)):
            if not (s, None, None) in g.triples((None, rico[1].thingIsSourceOfRelation, s)):
                for triple in g.triples((s, None, None)):
                    triples_to_remove.append(triple)
                # This part below is not needed really because none should exist
                #for triple in g.triples((s, None, None)):
                #   triples_to_remove.append(triple)
        removed_count = len(triples_to_remove)
        for triple in triples_to_remove:
            g.remove(triple)
            #print(*triple)
            removed_graph.add(triple)
        removed_graph.serialize(destination=removed_list_path, format="nt")
        pseudo_sparql = """
        PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>

        DELETE WHERE {
            ?s a rico:AgentControlRelation .
            FILTER NOT EXISTS {
                ?subject rico:thingIsSourceOfRelation ?s .
            }
        }
        """ # generated with ChatGPT based on parametrized
        print("Executed a parametrized alternative of the following query:", pseudo_sparql)
        print(f"{removed_count} triples were removed and dumped to: '{removed_list_path}'")
        return removed_count
    
    def remove_false_authtp(g):
        # Set config
        triples_to_remove = []
        removed_graph = Graph()
        removed_triples_output_format = 'nt'
        removed_triples_output_encoding = 'utf-8'
        pseudo_sparql = """
        PREFIX rico: <https://www.ica.org/standards/RiC/ontology#>
        PREFIX authtp: <https://data.archives.gov.on.ca/Schema/Authority/AuthorityType#>
        DELETE WHERE {
            ?s1 rico:hasOrHadCorporateBodyType authtp:Geographic%20Name .
            ?s2 rico:hasOrHadCorporateBodyType authtp:Family%20Name .
            ?s3 rico:hasOrHadCorporateBodyType authtp:Personal%20Name .
            authtp:Geographic%20Name ?p1 ?o1 .
            authtp:Family%20Name ?p2 ?o2 .
            authtp:Personal%20Name ?p3 ?o3 .
        }
        """

        # Run parametrized query
        authtp = ('authtp', Namespace(URIRef(f"{base_auth_uri}/AuthorityType#")))
        g.namespace_manager.bind(*authtp)
        authtp_list = [
            authtp[1]['Geographic%20Name'],
            authtp[1]['Family%20Name'],
            authtp[1]['Personal%20Name']
        ]
        for authtp_name in authtp_list:
            for s, p, o in g.triples((None, rico[1].hasOrHadCorporateBodyType, authtp_name)):
                triples_to_remove.append((s, p, o))
            for s, p, o in g.triples((authtp_name, None, None)):
                triples_to_remove.append((s, p, o))
        removed_count = len(triples_to_remove)
        for triple in triples_to_remove:
            g.remove(triple)
            #print(*triple)
            removed_graph.add(triple)

        # Save removed triples
        print("Executed a parametrized alternative of the following query:", pseudo_sparql)
        if removed_count > 0:
            ttl_filename = os.path.basename(graph_path)
            removed_triples_filename = f'{ttl_filename[:-4]}_removed_triples.{removed_triples_output_format}'
            removed_list_path = os.path.join(os.path.dirname(graph_path), removed_triples_filename)
            removed_graph.serialize(destination=removed_list_path,
                                    format=removed_triples_output_format,
                                    encoding=removed_triples_output_encoding)
            print(f"{removed_count} triples were removed and dumped to: '{removed_list_path}'")
        else:
            print(f"No triples were removed.")
        return removed_count

    def run_postprocessing():
        nonlocal total_count
        original_set = set(g)
        print("Postprocessing...")
        #total_count = total_count - remove_false_agentcontrolrelation(g)
        total_count = total_count - remove_false_authtp(g)
        print_total_count()
        return set(g) != original_set
    has_changed = run_postprocessing()

    return g, has_changed

def save_postprocessed_graph(
        graph,
        output_format = 'nt', # assumed to be quickest
        ttl_path = None):
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

#def map_rml_main():
#    parser = argparse.ArgumentParser(description="Map schema of choice")
#    parser.add_argument("schema", help="Choose one: add or auth.")
#
#    args = parser.parse_args()
#
#    rml_path, rmlmapper_path, ttl_path = map_rml(str(args.schema).lower())
#    graph, has_changed = postprocess(ttl_path)
#
#    if has_changed:
#        postprocessed_ttl_content = save_postprocessed_graph(ttl_path)
#    #print(postprocessed_ttl_content)
### End logic from map_rml.py

### Begin own riconvert logic
# Originally generated with ChatGPT-4o on 2024-12-13, modified
import os
import sys
import subprocess
import re
import io
import contextlib
import shutil
from datetime import datetime, timezone

def sanitize_filename(filename):
    """
    Sanitize filename using the specified algorithm:
    - Convert to lowercase
    - Replace spaces with underscores
    - Remove special characters
    """
    # Convert to lowercase
    sanitized = filename.lower()
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Remove specified special characters
    sanitized = re.sub(r'[()[\]/,:."\']', '', sanitized)
    
    return sanitized

def find_input_file(script_dir):
    # Find the single .drawio file
    drawio_files = [
        f for f in os.listdir(script_dir) 
        if f.lower().endswith('.drawio')
    ]
    
    if len(drawio_files) == 0:
        print("No .drawio file found in the directory.")
        sys.exit(1)
    
    if len(drawio_files) > 1:
        print("Multiple .drawio files found. Only one file is expected.")
        sys.exit(1)
    
    input_file = os.path.join(script_dir, drawio_files[0])
    return input_file

def generate_output_dir(script_dir, input_file):
    # Create output directory name based on DrawIO filename
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    sanitized_filename = sanitize_filename(base_filename)
    riconverted_prefix = "riconverted_"
    output_dir_name = riconverted_prefix + sanitized_filename
    output_dir = os.path.join(script_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, sanitized_filename

def convert_drawio_file(script_dir, input_file, output_dir, sanitized_filename):
    """
    Find and convert a single DrawIO file in the script directory
    
    Args:
        script_dir (str): Directory to search for .drawio file
    """
    print("Welcome to riconvert, a Records in Contexts Ontology (RiC-O) conversion utility.")
    print("It detects a *.drawio and *.csv file in the same folder (only one of each must exist).")
    print("It uses the DrawIO file to create a Manchester OWL, Turtle, and RML (RDF Mapping language) file.")
    print("It uses the RML file to map the CSV file into triples (Turtle or N3).")

    print("Input drawio file:")
    print(input_file)
    
    print("Output directory:")
    print(output_dir)

    # Prepare output file paths
    drawio_to_owl_prefix = "owl_from_drawio_"
    drawio_to_ttl_prefix = "turtle_from_drawio_owl_"
    output_file = os.path.join(output_dir, f'{drawio_to_owl_prefix}{sanitized_filename}.owl')
    ttl_file = os.path.join(output_dir, f'{drawio_to_ttl_prefix}{sanitized_filename}.ttl')

    # Make sure we do not inadvertently overwrite anything
    if os.path.exists(output_file) or os.path.exists(ttl_file) or os.path.exists(os.path.join(output_dir, os.path.basename(os.path.abspath(__file__)))) or os.path.exists(os.path.join(output_dir, os.path.basename(input_file))):
        response = input("Some existing files already found in the output directory. EVERYTHING in the output directory will be overwritten! Do you really want to proceed? (y/N): ").lower().strip()
        if response != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    # Create a copy of the script for reproducibility
    shutil.copy2(os.path.abspath(__file__), output_dir)

    # Create a copy of the drawio for reproducibility
    shutil.copy2(input_file, output_dir)
    
    try:
        # Prepare default parser commands
        parser_commands = [
            '-m', 'url',
            '-c', 'none',
            '-o', 'http://gbad.archives.gov.on.ca', 
            '-p', 'http://gbad.archives.gov.on.ca/'
        ]
        
        # Prepare full argument list for _run
        full_args = [input_file] + parser_commands
        
        # Call _run function (assuming it's defined in the parent context)
        #sys.argv = full_args
        #import draw_io_parser  # Assuming this is imported in parent context
        
        # Capture stdout to write OWL file
        with open(output_file, 'w') as owl_out:
            with contextlib.redirect_stdout(owl_out):
                try:
                    draw_io_parser_run(full_args)
                except ParseException as exception:
                    sys_exit(str(exception))
                except Exception as exception:  # pylint: disable=broad-exception-caught
                    error_type = type(exception).__name__
                    error_traceback = traceback.format_exc()
                    sys_exit(f"An unexpected error occurred: {error_type}: {exception}\n\nTraceback:\n{error_traceback}")
        
        print(f"OWL Output saved to: {output_file}")
        
        # Convert OWL to TTL using robot
        robot_cmd = [
            'java', 
            '-jar', 
            os.path.join(script_dir, 'robot.jar'), 
            'convert', 
            '-i', output_file, 
            '-o', ttl_file
        ]
        
        subprocess.run(robot_cmd, check=True)
        print(f"TTL Output saved to: {ttl_file}")
    
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

    return ttl_file

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()  # Ensure the output is flushed to both streams

    def flush(self):
        for stream in self.streams:
            stream.flush()

def main():
    """
    Main function to convert a single DrawIO file
    """
    # Read script dir and find input file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = find_input_file(script_dir)
    
    # Set up output directory and log file
    output_dir, sanitized_filename = generate_output_dir(script_dir, input_file)
    datetime_now = datetime.now(timezone.utc)
    log_file_path = os.path.join(output_dir, f"riconversion_{datetime_now.strftime('%Y-%m-%d_%H-%M-%S')}.log")
    
    # Open log file in write mode
    with open(log_file_path, 'w') as log_file:
         # Log the start time and command
        log_file.write(f"Execution started at: {datetime_now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

        # Create a Tee object to capture both stdout and the log file
        tee = Tee(sys.stdout, log_file)

        # Redirect stdout and stderr to Tee
        sys.stdout = tee
        sys.stderr = tee
        
        # Convert the DrawIO file
        graph_path = convert_drawio_file(script_dir, input_file, output_dir, sanitized_filename)

        # Convert schema to RML
        csv_path = os.path.join(script_dir, 'authority_tailshuf_100.csv')
        map_schema_init(graph_path, csv_path, output_dir, sanitized_filename, schema_code='auth', source_filename=None)

        # Map RML
        rml_path, rmlmapper_path, ttl_path = map_rml(script_dir, output_dir, schema_code='auth')
        graph, has_changed = postprocess(ttl_path)

        if has_changed:
            postprocessed_ttl_content = save_postprocessed_graph(graph, ttl_path)
        #print(postprocessed_ttl_content)

        # Log the end time
        log_file.write(f"\nExecution ended at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

if __name__ == '__main__':
    try:
        # Your main script logic goes here
        main()
    except Exception as e:
        # Print the exception traceback to the console
        print("An error occurred:")
        traceback.print_exc()  # This will print the full error traceback
        input("Press Enter to exit...")  # Wait for user to press Enter
        #os.system("pause")
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        # Ensure the log captures the normal termination
        # Restore stdout and stderr so that they are printed to the console again

        # Prevent the command window from closing automatically
        input("Press Enter to exit...")
        #os.system("pause")
### End own riconvert logic
