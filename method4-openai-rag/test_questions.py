import rag_queries


def test_question_1():
    question = "What cotton weeds have reported glyphosate resistance?"
    expected_response = """
Glyphosate (Group 9(M)) resistance has been confirmed and is widespread in the following cotton weeds:
Windmill grass
Awnless barnyard grass
Fleabane
Sowthistle
Feathertop Rhodes grass
Liverseed grass
Annual ryegrass is a significant issue in Southern valleys and is emerging as a problem in Northern NSW. There are reports
of cross resistance to glyphosate and Group 1(A) herbicides.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_2():
    question = "What are the advantages of retaining cotton stubble?"
    expected_response = """
adds organic matter to the soil
improves soil tilth
decreases soil bulk density
creates greater biological activity in the soil
maintains active populations of soil organisms
supplies energy to the soil microbial biomass
enhances nutrient cycling
improves fertiliser use efficiency
improves moisture infiltration
reduces wind and water erosion
incorporating stubble forms part of the pupae-busting operation
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_3():
    question = "What kinds of soil are there in cotton growing regions?"
    expected_response = """
The soils on which cotton is grown in Australia are inherently
fertile relative to the majority of rangeland soils used for
grazing. They are dominated by cracking clays (vertosols),
which are naturally fertile, alkaline, with high clay content
(>35%) and, initially, where the soils that supported brigalow/
belah vegetation associations with relatively high organic
matter content. These soils were formed from fertile alluvium
and wind-blown dust under conditions of relatively low rainfall.
Other cotton-growing soils include chromosols (in the
Macquarie, Namoi, Gwydir, Lachlan and Murrumbidgee
valleys), and in many of the Queensland districts, sodosols
form a part of the soil spectrum.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_4():
    question = "Tell me what are the major nutrients taken up by cotton?"
    expected_response = """
Nitrogen (N)
Phosphorous (P)
Potassium (K)
Calcium (Ca)
Magnesium (Mg)
Sulphur (S)
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_5():
    question = "What factors can restrict nutrient uptake in cotton?"
    expected_response = """
poor physical soil structure (e.g. compaction) or soil
chemical toxicities (e.g. salinity, sodicity, pH) limit root
growth, reducing nutrient uptake, even where sufficient
nutrients are available
a deficiency of one nutrient limits crop growth, reducing
the capacity of the plant to take up or metabolise other
nutrients
as the crop matures, nutrients and sugars within the
plant are diverted from vegetative (including roots) to
reproductive organs
oxygen supply, needed by roots to maintain metabolic
processes, including nutrient uptake, is restricted, i.e.
through waterlogging.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_6():
    question = "How does nutrient deficiency show in cotton?"
    expected_response = """
Nitrogen deficiency symptoms
Deficiency symptoms include small, pale yellow leaves.
Nitrogen-deficient plants are stunted, and produce fewer
nodes through a combination of fewer vegetative branches
and fruiting branches that will also be reduced and shorter.
As N deficiency progresses, older leaves become yellow,
as N is remobilised to new growth. Leaves with severe N
deficiency turn various shades of autumn colours as tannins
in the leaves are expressed.

Phosphorus deficiency symptoms
P deficiency symptoms for cotton include stunted plants
with dark green foliage, which may later become discoloured
(reddish, purple on some plant parts).
When the deficiency is not corrected, fruiting is delayed and
restricted.

Potassium deficiency symptoms
In the plant, the mobile potassium ion will move to new growth
from the older leaves. Hence, K deficiency first appears in
older leaves. Initially, the leaf margins and interveinal areas
show a yellowish-white mottling, then a rusty bronze colour.
Necrotic spots between the leaf veins cause the leaf to
appear rusted or dotted with brown specks at the leaf tip,
margins, and between the leaf veins. As the leaf breaks down,
the margins and leaf tip shrivel. Eventually, the whole leaf dies
and is shed as the condition moves up the plant. In severe
deficiencies, young leaves are affected and the terminal dies.
Premature shedding of leaves prevents boll development,
resulting in small immature bolls, many of which fail to open.
The symptoms of severe deficiencies are likely to occur only
in soils with low K reserves, where dry weather has restricted
root activity in relatively K-rich topsoils, or where a sudden
waterlogging event has restricted root activity in a proportion
of the soil volume. The latter situation can induce sudden
and severe onset of premature senescence, particularly if
the waterlogging events coincide with periods of high plant K
demand (e.g. during boll loading and filling).
When deficiencies are experienced later in the season, as
the developing boll load is a strong and competitive sink for
available K, the youngest mature leaf (YML) at the top of the
canopy is often the first to show symptoms.

Zinc deficiency symptoms
Zinc is relatively immobile in the plant. First deficiency
symptoms can be seen shortly after the first true leaves
appear. Plants lack vigour and appear unthrifty.
They are often shorter with thin stems, and have less
branching, flowering and boll set. In young plants, symptoms
appear as dark brown interveinal necrotic lesions (bronzing)
on the older true leaves. They develop without prior chlorosis,
and leaf margins are often cupped upwards. Eventually, the
lesions join up and the leaf dies. If the deficiency persists,
young leaves develop a pale yellow, blotchy chlorosis
(yellowing between the leaf veins) appearance. Leaves
become very small and are malformed, having holes or torn
margins (Grundon 1987).

 Iron deficiency symptoms
Iron deficiencies are mainly confined to the young growth,
as Fe is immobile within the plant. Crops lack vigour
and yield poorly but are only slightly smaller than normal
crops. The young leaves become yellow between the veins
(chlorosis) while the veins usually remain green. Under severe
deficiencies, veins fade and eventually the whole leaf may
turn white (Grundon, 1987). Leaves may appear limp, with
the tips and margins hanging down as if wilted. Severely
Fe-deficient plants show significant reductions in plant and
root growth, roots thicken and do not develop root hairs
(Vretta-Kouskoleka and Kallinis 1968). Although the plant
may contain high concentrations of Fe, most of it is in an
unavailable form, in which case chlorophyll production stops
and the leaves lose their green colour.

Copper deficiency symptoms
Initial indications of Cu deficiency are unthrifty and poor
yielding crops, stunted with short stems and dull green
leaves. Branching is reduced, and fewer flowers produced and
bolls set. Leaves initially appear limp and wilted, but as the
deficiency progresses, a faint, dull yellow interveinal chlorosis
develops in the older leaves. In severe cases, dieback of the
terminal bud is preceded by peculiar distortions, and tissues
die at the tip or base of the terminal (Grundon, 1987).

Boron deficiency symptoms
Boron deficiency symptoms vary with the stage of growth and
the severity of the deficiency. The problem is most commonly
found in sandy soils prone to leaching, although may also
occur during prolonged dry periods, or in alkaline soils when B
availability is reduced.
Mildly deficient crops lack vigour and yield poorly. The
plants appear bushy and stunted, with shorter branches
and internodes, dark green leaves and stout stems. Flower
production and boll set is significantly reduced. In severely
deficient crops, the plants often die before any flowers are
formed.
The first symptoms of B deficiency appear in new growth, as
B is relatively immobile in plants. The youngest leaves are
the most severely affected; they hang down and margins are
cupped under. The upper internodes are very short, resulting
in the new developing leaves in the apical bud to crowd
together and eventually die, preventing further growth of the
stem. If the deficiency persists, the apical buds in lateral
branches also die and, eventually, the whole plant (Grundon
1987). Other symptoms that have been described include
deformed and small bolls, boll shedding, hard locks, sepals
around the bolls are hard and fail to open, root growth can be
severely inhibited and secondary roots stunted (Stevens and
Dunn 2008, Gupta 1979).
The range between B deficiency and toxicity is narrow. Toxic
concentrations of B result in leaf cupping, chlorosis and death
of leaf tissue in localised spots.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_7():
    question = "What are the common ways of applying N to a cotton crop?"
    expected_response = """
Anhydrous ammonia (NH3)
Urea (CO[NH2]2)
Ammonium sulphate ([NH4]2SO4)
Starter fertilisers, such as mono-ammonium phosphate
(MAP) and di-ammonium phosphate (DAP), supply only a
small amount of N to cotton seedlings.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_8():
    question = "When and where should I soil sample in a cotton field?"
    expected_response = """
the preferred time to sample soil
is from July to September when significant changes in soil
nutrient content before sowing is unlikely. When fertiliser is
to be applied before this, a small, unfertilised area should be
left from where soil samples can be collected, or the decision
support tools used to interpret the soil test should be able
to provide an estimate of likely net N mineralisation from
sampling to sowing.
In a trouble-shooting situation, soil and plant tissue samples
should be taken from the good and poor areas at the same
time to ensure direct comparability.
The sample collection strategy for a paddock is a function of
the purpose of the sampling, the degree of variability in crop
performance across the area to be sampled, and the ability to
apply fertiliser products and rates to meet existing variability.
In the first instance, aim for a comprehensive soil sampling
spacing of approximately 400 m across cotton fields, i.e. one
sampling site per 16 ha, approximately. Use the sampling grid
or management zones determined by yield maps in a flexible
manner that allows the soil sampling plan to be adjusted
to include high- and low-yielding areas and other zones of
interest. In summary, sampling locations in developed cotton
fields should include:
• high-yielding zone
• average-yielding zone
• low-yielding zone.
Avoid collecting samples on sites such as old fence lines,
filled in irrigation channels, near trees or old stumps, or if
the soil is excessively wet. It is important to avoid sampling
fertiliser bands from previous (or current) years as this
can seriously affect laboratory analyses. This is especially
important where phosphorus (P), zinc (Zn) or potassium (K)
fertilisers have been applied. Sample soil close to the plant
line or from the middle of the bed, but avoid fertiliser bands.
This problem is not normally encountered where fertilisers
have been broadcast and incorporated.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_9():
    question = "How can you optimise spray appliation?"
    expected_response = """
Appropriate nozzle selection for product/adjuvants and target;
Appropriate operating parameters, inluding nozzle spacing, operating
pressure, travel speed and spray release height;
Meeting label requirements for spray quality and drift risk;
Consideration of most suitable timing for mobile targets (insects);
Careful calculation of actual field rates for product and water;
Ensuring the water used is of suitable quality;
Considering the potential for incompatibilities during mixing; and,
Regular calibration and rig maintenance.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_10():
    question = "What are the key cotton diseases?"
    expected_response = """
Alternaria leaf spot
Black root rot
Boll rot, tight lock and seed rot
Cotton bunchy top
Fusarium wilt
Nematodes
Ramularia leaf spot
Reoccurring wilt
Seedling diseases
Verticillium wilt
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_11():
    question = "What are the principles of the Resistance Management Plan? What are the elements of the Resistance Management Plan?"
    expected_response = """
The Resistance Management Plan is based on three basic
principles: (1) minimising the exposure of Helicoverpa spp. to the
Bacillus thuringiensis (Bt) proteins Cry1Ac, Cry2Ab and Vip3A;
(2) providing a population of susceptible individuals that can
mate with any resistant individuals, hence diluting any potential
resistance; and (3) removing resistant individuals at the end of
the cotton season. These principles are supported through the
implementation of five elements that are the key components of
the Resistance Management Plan. These elements are:
1. Planting timing restrictions;
2. Refuge crops;
3. Control of volunteers and ratoon cotton;
4. Pupae destruction/trap crops; and
5. Spray limitations
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_12():
    question = "At what crop stage should square retention assessment commence, and what actions should be taken for pest control up to the first 5 fruiting branches?"
    expected_response = """
Square retention assessment should commence when the crop has 3-4 fruiting branches (9-12 nodes), with a count of all positions. Regarding pest control actions:
If square retention is less than 40%, control actions should only be taken if pest insects are still present at the threshold and the crop has reached 4-5 fruiting branches.
If square retention is between 40% and 80%, no action is required except to continue monitoring for insects.
If square retention is more than 80%, no action is required, but monitoring for insects should continue.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_13():
    question = "What are the key insect pests?"
    expected_response = """
Aphids
Armyworm (Spodoptera species)
Helicopverpa
Mealybug
MiridsPage
Mites
Soil and establishment pests
Stink bugs and stainers
Thrips
Whitefly
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_14():
    question = "How does the visual soil assessment method determine the health of soil?"
    expected_response = """
The visual soil assessment is a method of conducting soil health measurement
internationally endorsed by the FAO... to track changes over time to know if
we’re going backwards or forwards... using a copy of the FAO’s Visual Soil 
ssessment and a scorecard... you'll need a spade, a box, and a tarpaulin... 
watch videos for a complete guide through the process
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_15():
    question = "What are the main components of soil biology according to current research?"
    expected_response = """
Living components include plants and animals, microflora like bacteria and fungi,
and macrofauna like earthworms, ants, etc. Non-living
organic matter includes dead plant and animal residue.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_16():
    question = "How does soil biology influence nutrient cycles in crop systems?"
    expected_response = """
Soil organic matter...decomposed by soil organisms... provides them with energy to grow
and reproduce, thus contributing to the nutrient
cycles essential for growing plants.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_17():
    question = "What are the potential effects of reconfiguring field layouts on crop productivity?"
    expected_response = """
Redesigning farm layouts can lead to variability across the field... Removing topsoil can
expose less fertile subsoil, affecting soil health and productivity.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_18():
    question = "What is the importance of assessing soil texture in visual soil assessments?"
    expected_response = """
The texture of the soil is crucial as it affects moisture retention and root penetration,
which are vital for crop growth.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_19():
    question = "How can soil biology be measured effectively in the field?"
    expected_response = """
Soil biology is typically assessed through measurements of soil organic carbon, which is
indicative of biological activity and soil health.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_20():
    question = "What are the common soil constraints affecting cotton crops in Australia?"
    expected_response = """
High magnesium content, low calcium-to-magnesium ratios, and high soil sodicity are common 
oil constraints in Australian cotton production.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_21():
    question = "Discuss the assessment methods for porosity, mottling, and soil color in visual soil assessments."
    expected_response = """
Assessment includes examining soil crumbliness for porosity, checking the extent of mottling for
signs of anaerobic conditions, and comparing soil color against reference soils to infer organic content.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_22():
    question = "How does soil structure affect the overall health of the soil?"
    expected_response = """
Soil structure affects water infiltration, root penetration, and aeration, which are crucial for healthy
plant growth.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_23():
    question = "What are the best practices to manage soil compaction in wet conditions?"
    expected_response = """
Allow soils to recover naturally, use deep ripping and crop rotation to alleviate compaction, and adjust
field traffic to minimize further compaction.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_24():
    question = "What methods are used to measure soil organic carbon, and why is it important?"
    expected_response = """
Soil organic carbon is measured using the Walkley-Black method or loss on ignition... important for soil
structure, nutrient cycling, and as a baseline for carbon trading.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_25():
    question = "What roles do earthworms play in Australian cotton systems?"
    expected_response = """
Earthworms, although beneficial for soil aeration and nutrient cycling, are less influential in Australian
cotton systems due to typically lower organic matter and moisture.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_26():
    question = "Explain the visual soil assessment’s approach to evaluating rooting depth, surface ponding, and erosion."
    expected_response = """
Evaluates rooting depth using probe data, checks for surface ponding to assess soil drainage, and examines
soil for signs of erosion to assess structural stability.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_27():
    question = "What does the 2018 edition of NUTRIpak include that differs from earlier versions?"
    expected_response = """
Includes updated information on nutrient management, especially nitrogen and phosphorus, reflecting higher
average yields and new insights into nutrient cycling
and soil health.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_28():
    question = "How do the soil properties of clay influence cotton production in Australia?"
    expected_response = """
Clay soils, which dominate cotton-growing areas in Australia, are naturally fertile but can pose challenges
such as compaction and reduced aeration.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_29():
    question = "What are the strategies for managing nutrient cycling effectively in cotton production?"
    expected_response = """
Strategies include the proper management of soil organic matter, using cover crops to enhance nutrient cycling,
and maintaining balanced soil chemistry.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response


def test_question_30():
    question = "Discuss the implications of long-fallow disorder on mycorrhizal populations in cotton soils."
    expected_response = """
Long-fallow disorder leads to reduced mycorrhizal populations, which can affect nutrient uptake and overall
soil health in cotton production systems.
"""
    actual_response = rag_queries.query_openai_rag(question)
    return actual_response, expected_response
