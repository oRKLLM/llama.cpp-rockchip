#pragma once

// Built-in calibration corpus for the .orkpack GPTQ pass.
//
// WHY THIS EXISTS AT ALL. GPTQ is error feedback against a Hessian H = A^T A accumulated over real
// activations, so it needs text pushed through the model. Every other rung of the quality ladder
// (Hadamard rotation, SmoothQuant, int8 promotion) is a pure function of what the packer already has
// in hand; this one is not. A pack is built lazily on first load, with nobody around to supply a
// corpus, so "the ladder is on by default" requires a default corpus to exist.
//
// WHAT IT IS AND IS NOT. Ordinary English prose across a deliberately wide spread of registers --
// narrative, technical, legal, culinary, scientific, conversational -- because H is an input-channel
// covariance and what matters is that the channels a real workload uses are actually excited. It is
// NOT a substitute for calibrating on the distribution you will serve: --pack-calib <file> overrides
// it and should be used whenever a representative corpus exists.
//
// ITS LIMIT, STATED PLAINLY. rank(H) <= the number of calibration rows, and a weight of input width K
// wants at least K rows; below that the damping term owns the null space and GPTQ degrades toward
// round-to-nearest in exactly those directions. This text is roughly 2.5k tokens. That is full rank
// for K <= 2048 and short of it for the widest FFN weights of a mid-sized model, so the driver prints
// rows-available against rows-needed on every build rather than letting a short corpus pass silently.
// Enlarging the default is cheap; pretending a short one is sufficient is not.
//
// CALIBRATE AND EVALUATE ON DISJOINT TEXT. This corpus is fixed and shipped, so it is disjoint from
// any evaluation set by construction -- which is the one property a built-in default gets for free
// and a hand-picked corpus usually gets wrong.

static const char * const ORK_CALIB_TEXT = R"CALIB(
The river had carved its way through the limestone for longer than anyone could reasonably imagine, and the result was a gorge whose walls recorded, in bands of ochre and grey, the slow arithmetic of deposition. Visitors came in summer to look down from the iron bridge, and most of them saw only a pretty drop; a few, usually the ones who had read the plaque, understood that they were looking at several million years arranged vertically.

Economic forecasting is an unusual discipline in that its practitioners are judged almost entirely on outcomes they cannot control. A model may be internally coherent, carefully specified, and honest about its assumptions, and still be wrong because a shipping lane closed or a central bank changed its mind on a Thursday. The useful forecasters are not the accurate ones but the ones who say clearly which of their assumptions, if broken, would break everything else.

To braise a shoulder of lamb properly you want a heavy pot with a lid that actually fits. Brown the meat hard on every surface, in batches if you must, because crowding the pan traps steam and steam is the enemy of colour. Soften onions and carrots in the rendered fat, deglaze with a glass of something you would drink, return the lamb, add stock to come halfway up, and then leave it alone in a low oven for three hours. The temptation to check it every twenty minutes is the main reason people think braising is difficult.

The court held that the clause was unenforceable not because its terms were unclear but because the party seeking to rely on it had, by its own conduct over eighteen months, represented that it would not. Estoppel is often described as a shield rather than a sword, and the distinction matters here: the defendant was not asking for damages, only to be relieved of an obligation the claimant had spent a year and a half implying did not exist.

Photosynthesis is frequently taught as a single equation, which is a convenient lie. In practice the light-dependent reactions and the carbon-fixing cycle are separated in both space and chemistry, linked by two carriers whose regeneration sets the whole pace. Plants in bright, dry conditions face a genuine conflict: the stomata that admit carbon dioxide also lose water, and several lineages have independently evolved ways of opening them at night and storing the carbon until morning.

She had not expected the letter, and certainly not in that handwriting. For a moment she stood in the hallway with the envelope unopened, aware of the dog scratching at the kitchen door and the kettle beginning to complain, and she thought about how a single rectangle of paper can reorganise an afternoon. Then she opened it, read four lines, and sat down on the stairs.

Distributed systems fail in ways that single machines do not, and the interesting failures are almost never the ones where a component stops. A component that stops is easy; everyone notices. The expensive failures are partial: a node that answers health checks but serves stale data, a network partition that heals asymmetrically, a clock that drifts just far enough to reorder two events that must not be reordered. Designing for these means accepting that the system will spend much of its life in a state no one drew on the whiteboard.

The second movement is where most amateur orchestras come apart, not because it is technically harder but because it is slow. Slow music exposes intonation, exposes ragged entries, exposes the fact that half the section is counting and the other half is following. A conductor who spends rehearsal time on the fast passages because they sound more impressive is usually storing up a disaster for the concert.

Soil carbon is one of the few climate levers that is simultaneously enormous, cheap, and extremely difficult to verify. The global stock held in the top metre of soil dwarfs the atmospheric pool, and management changes can move it in either direction within a decade. The verification problem is that measuring a small percentage change in a very large and spatially heterogeneous quantity requires either a great many samples or a model you are prepared to defend.

He maintained, with the patience of a man who had made the argument many times, that the bicycle was the most efficient machine ever built for converting human effort into distance, and that everything invented since had mainly been an elaboration on a solved problem. His workshop supported the thesis: frames hanging from the ceiling, a wall of spoke keys, and a tea-stained notebook recording every wheel he had ever built and whether it had stayed true.

In the eighteenth century the problem of longitude at sea was not academic. A ship could establish its latitude from the sun without much difficulty, but without knowing how far east or west it had travelled a navigator was reduced to dead reckoning and hope. The eventual solution was mechanical rather than astronomical, which disappointed a great many astronomers, and it required a clock that would keep time on a rolling deck through heat, cold, and salt.

The first thing to understand about the immune system is that it is not primarily a defence against the unfamiliar; it is a system for distinguishing self from other, and most of its sophistication is spent on not attacking the body it lives in. Autoimmune disease is therefore not a failure of strength but a failure of discrimination, which is why treatments that simply suppress the whole apparatus tend to trade one set of problems for another.

Markets for perishable goods behave differently from markets for durable ones in a way that surprises people the first time they see it. A fish merchant at four in the afternoon is not negotiating about value; he is negotiating about the difference between some money and no money. Prices in the last hour of a wholesale market are a better illustration of marginal cost than most textbooks manage.

The cathedral took one hundred and forty years to build, which means that essentially nobody who laid its foundations saw its roof. Mediaeval masons worked within a tradition that made this tolerable: the drawings were held in the lodge, the proportions were transmitted by rule and by apprenticeship, and a change of master did not mean a change of plan. It is a model of institutional memory that modern organisations, which struggle to retain intent across a two-year project, might usefully study.

Rust forms when iron, oxygen, and water meet, but the rate at which it forms depends enormously on what else is present. Chloride ions, for instance, disrupt the thin passive oxide layer that would otherwise slow the reaction to a crawl, which is why a car driven on salted winter roads corrodes at a pace that would astonish someone from a dry climate. Protection strategies either exclude the water, exclude the oxygen, or supply a more reactive metal to be consumed instead.

The interview went badly in a way that took him two days to understand. Nothing had been said that was obviously wrong; the questions were answered, the examples were relevant, the tone was correct. But he had spent the whole hour describing what he had done rather than what he had decided, and the difference, he realised on the train, was the entire point of the conversation.

Antarctic sea ice behaves quite differently from Arctic sea ice, and treating them as two instances of the same phenomenon has led to a good deal of confused commentary. The Arctic is an ocean surrounded by land, so its ice is constrained, thick, and increasingly multi-year. The Antarctic is land surrounded by ocean, so its ice is free to expand outward every winter and almost entirely melts away every summer. The two respond to warming on different timescales and through different mechanisms.

Good technical writing has an unglamorous property: it tells the reader what they will be able to do after reading it, and then it does exactly that and stops. The common failure is not obscurity but generosity -- the author knows a great deal, and every paragraph offers an aside. The reader, who came for one specific thing, now has to perform search as well as comprehension.

The grammar of a language is not a set of rules imposed from above but a description of regularities that speakers already observe without being able to state them. This is why native speakers can reject a sentence as wrong while being entirely unable to say which rule it breaks, and why prescriptive rules that were genuinely invented -- the prohibition on splitting an infinitive, for example -- never quite take hold in speech.

Bridge inspection is largely a matter of looking for the consequences of water. Water gets into joints, freezes, and levers concrete apart; it carries chloride to reinforcing steel, which then rusts, expands, and spalls the cover off; it pools where drainage was designed on paper and never cleared in practice. An inspector with a hammer and a good torch finds most of what matters, and the expensive instrumentation is usually confirming a suspicion rather than generating one.

Sourdough is a managed ecology rather than a recipe. The culture is a stable association of wild yeasts and lactic acid bacteria whose balance shifts with temperature, hydration, and feeding schedule, and the flavour of the bread is mostly a readout of that balance. Bakers who move house and find their starter behaves differently are not imagining it; the flour has changed, the water has changed, and the ambient population has changed.

The treaty was signed on terms that both delegations privately regarded as unsustainable, which is a more common outcome than diplomatic history usually admits. Each side had a domestic audience that would reject anything better for the other, and a negotiating team that understood, correctly, that an imperfect agreement signed now would constrain behaviour for at least a decade while a perfect one negotiated later would not exist.

Birdsong is learned in most songbird species, and learned within a sensitive period that closes. A young bird exposed to no song, or to the wrong song, produces an impoverished version for life. The parallels with human language acquisition are close enough to be useful and loose enough to be dangerous, and the literature contains a good deal of both careful comparison and overreach.

Cache invalidation is difficult less for technical reasons than for organisational ones. The code that writes the underlying data and the code that caches a derived view of it are usually owned by different teams, written at different times, and deployed on different schedules. A correct invalidation protocol requires the writer to know what views exist, which is exactly the coupling the cache was introduced to avoid.

The volcano had been quiet for two centuries, which the surrounding towns had reasonably interpreted as safety. Geologists reading the deposits saw something else: a pattern of long repose followed by large eruption, with the repose length roughly proportional to the size of what followed. Communicating this is one of the hardest problems in the field, because the honest statement -- low probability, enormous consequence, unknown timing -- is precisely the shape of warning that people discount.

She kept accounts the way her mother had, in a cloth-bound ledger with a column for everything, and she could tell you what the shop had taken on any Tuesday for twenty years. When the accountant suggested software she listened politely and then explained that the ledger was not for the accountant, it was for her, and that the act of writing the number down was how she noticed when something was wrong.

Nuclear reactor control depends on a happy accident of physics: a small fraction of the neutrons from fission are emitted not instantly but seconds to minutes later, by the decay of certain fission products. Without those delayed neutrons the response time of a reactor would be microseconds and no mechanical control system could follow it. With them, the effective response time is on the order of seconds, which is comfortably within human and mechanical reach.

The argument for standardised containers was never really about the boxes. It was about the interface: once the box is a fixed size with fixed lifting points, the crane, the ship, the lorry, the train wagon, and the warehouse can all be designed independently against the same specification. The box is the least interesting part of containerisation, and the part everyone remembers.

Memory in the colloquial sense -- recalling what happened -- is reconstructive rather than reproductive, which is why confident recollection and accurate recollection are so weakly correlated. Each retrieval is also an encoding, so a memory discussed often is a memory edited often. This is inconvenient for the legal system, which continues to treat eyewitness confidence as evidence of reliability despite several decades of results to the contrary.

Pruning a fruit tree is an argument with the tree about where it should put next year's energy. Cut in winter and the response is vigorous regrowth; cut in summer and the response is muted, because the leaves that would have funded the regrowth are gone. Most beginners prune too timidly in the first year and then, alarmed by the thicket this produces, far too hard in the second.

Compilers spend most of their effort not on translation but on deciding which of many legal translations to emit. Register allocation alone is a graph colouring problem that is formally hard and practically solved by heuristics that work well on code humans actually write. The interesting consequence is that performance is often less a property of the source than of how closely the source resembles the shapes the optimiser was tuned against.

The storm arrived two hours earlier than forecast and from a slightly different bearing, which was enough to put the moored boats on a lee shore. By morning three had dragged their anchors onto the shingle and one had gone through the harbour wall. The subsequent inquiry found no single failure, only a forecast within its stated error bars, a harbourmaster following standing instructions, and a set of moorings specified for conditions that had been typical when they were laid in 1974.

Antibiotic resistance is an evolutionary problem disguised as a clinical one. Every course of treatment is a selection event, and the resistant survivors are not a hypothetical future population but the actual organisms left behind this week. Stewardship programmes that reduce unnecessary prescribing work, measurably, which is encouraging; the discouraging part is that resistance genes move horizontally between species, so the accounting cannot be done one pathogen at a time.

Translation between languages with different information structure is less a matter of finding equivalent words than of deciding what to make explicit. A language that marks evidentiality obliges the speaker to say whether they witnessed an event or heard about it; rendering that into a language without the category means either adding a clause that was not emphasised in the original or dropping a distinction that was obligatory.

The second-hand bookshop had no catalogue and no apparent system, and the owner could nevertheless find anything within a minute. Asked how, he said that he shelved by when he had acquired a book rather than by subject, and that he remembered acquisitions. It was, he conceded, a system with exactly one user, and he had no plan for what would happen to it afterwards.

Groundwater moves slowly, which has two consequences that are easy to state and hard to internalise. The first is that contamination arriving at a well today may have entered the aquifer decades ago, so stopping the source does not stop the problem. The second is that abstraction can exceed recharge for a very long time before the water table announces it, which is why the collapse, when it comes, feels sudden.

In the theatre the lighting plot is written before the actors have fully blocked the scene, which means the designer is predicting where people will stand. Experienced designers over-light the probable positions and leave the edges soft, so that a late change in blocking costs a focus note rather than a re-rig. It is a discipline of leaving oneself options, and it transfers surprisingly well to other kinds of planning.

Statistical significance was never intended as a decision rule and has been used as one for a century. A p-value below a threshold tells you something about the probability of data under a null hypothesis and nothing about the probability of the hypothesis, the size of the effect, or whether it matters. The replication difficulties of the last decade are less a crisis of fraud than the predictable result of promoting a diagnostic to a verdict.

The engine note changed about a mile outside the village, dropping half an octave and acquiring a faint periodic knock. He pulled in beside a gate, lifted the bonnet, and found exactly what he expected: a belt that had been on his list since spring, frayed through on one side, still turning but not for much longer. There is a particular satisfaction in a failure you predicted and a particular irritation in having predicted it and done nothing.

Urban heat islands arise from three effects that compound. Dark, dense materials absorb and re-radiate more solar energy than vegetation; the geometry of streets traps that radiation between surfaces; and the removal of vegetation removes evaporative cooling. The remedies therefore also come in three kinds, and the cheapest of them -- lighter surfaces -- is the one that requires no maintenance and is consistently the hardest to get adopted.

A well-run meeting has a decision at the end of it, and everyone knows before it starts what the decision will be about. Meetings that exist to share information are usually a document that nobody wanted to write. Meetings that exist to generate ideas work only if someone has prepared the constraints, because unconstrained brainstorming produces volume rather than options.
)CALIB";
