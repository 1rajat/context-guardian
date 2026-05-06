"""Filler text generator using public-domain prose, chunked to exact token counts."""

import tiktoken

# Public domain prose — Paul Graham essay style, Gutenberg excerpts blended together.
# Multiple paragraphs so we can stitch them to arbitrary lengths.
_CORPUS = """
It is a truth universally acknowledged that a startup in possession of a good idea
must be in want of funding. But this framing gets things backwards. The best startups
are not the ones that raised the most money — they are the ones that needed the least
and moved the fastest. Capital is a multiplier, not an engine. If you have no engine,
adding more fuel merely fills the tank. The engine is the people.

What makes a great programmer? Not talent alone. Talent without taste produces code
that technically works but is unpleasant to read, fragile to modify, and expensive to
extend. Taste is the sense that tells you when a solution is beautiful versus merely
correct. You can teach someone to write code; teaching taste is harder. It comes from
reading vast amounts of code, noticing what delights you versus what makes you wince,
and caring enough to push past the first solution that works toward the one that sings.

The history of technology is largely the history of making things smaller, cheaper, and
faster. The vacuum tube gave way to the transistor; the transistor gave way to the
integrated circuit; the integrated circuit spawned the microprocessor. Each transition
compressed enormous complexity into a smaller, more manageable, more reliable package.
We are in the middle of a similar compression in artificial intelligence. Models that
once required warehouses now run in a pocket. The endpoint of this compression is
unclear, but the direction is not.

In the beginning was the word, and the word was stored in RAM. Memory is the substrate
of thought — both biological and computational. A mind without memory is not a mind at
all; it is a reflex arc. Long-term memory lets you build on what you learned yesterday.
Short-term memory lets you hold the thread of an argument while constructing its next
step. Context windows are the computational analogue of working memory: the space inside
which current reasoning must occur. Expanding that space changes what kinds of reasoning
are possible.

Cities are the original social networks. Before fiber optic cables, before satellites,
before the printing press, ideas propagated through proximity. You had to be in Florence
to absorb the Florentine synthesis of classical learning and mercantile ambition. You had
to be in 17th-century London to encounter the first coffeehouses where the news of the
day was debated aloud. Silicon Valley is merely the latest instance of the same ancient
pattern: dense clustering of ambitious people around a scarce resource — in this case,
venture capital and world-class engineering talent — produces disproportionate output.

There are two kinds of knowledge: knowing that something is true, and knowing why it is
true. The first kind is cheap to acquire and quickly forgotten. The second kind is
expensive — it requires going back to first principles, working through the derivation
yourself, making and correcting errors — but it sticks, and it transfers. When you
understand why a sorting algorithm works, you can derive it again if you forget it.
When you only know that it works, forgetting it leaves you stranded.

The sea does not care about your boat. This is the essential lesson of ocean sailing, and
it generalizes. Nature is indifferent to human intentions. Markets are indifferent to
your business plan. Your competitors are indifferent to how hard you worked. Indifference
is not hostility — it is something colder and more useful to understand. It means that
results are the only currency, and that excuses, however valid, buy nothing.

Language is the user interface of thought. The words you have access to determine the
thoughts you can think with precision. This is why vocabulary matters — not because
big words impress people, but because precise words let you manipulate precise ideas.
When Eskimo languages have many words for snow, it is not a quirk; it is an adaptation
to an environment where the distinctions between types of snow carry survival information.
The vocabulary of any domain encodes the distinctions that practitioners have found worth
tracking.

Measurement is the beginning of improvement. You cannot optimize what you do not measure.
But this truth has a shadow: not everything worth doing is easily measurable, and systems
that optimize only for what is measured tend to degrade on what is not. Schools that
teach to the test produce students who test well. Companies that optimize for quarterly
earnings produce quarters that look good in earnings calls. The art is in choosing what
to measure carefully enough that the metric captures the underlying thing you care about.

Every abstraction leaks. This is Spolsky's law, and it has not been repealed. A leaky
abstraction is one whose implementation details intrude on the surface you were promised
you could ignore. TCP/IP is supposed to hide the unreliable behavior of the underlying
network, but when your connection drops mid-stream, the unreliable network is suddenly
very visible. SQL is supposed to hide the complexity of data storage, but when your query
plan is catastrophically wrong, you are suddenly in the internals. Abstractions are
useful, indispensable even, but they are not magic. Underneath every abstraction is a
real thing, and that real thing always reserves the right to surprise you.

The best writers read voraciously. The best musicians listen constantly. The best
programmers read code — other people's code, old code, code in languages they don't use
professionally. Reading code is how you absorb idioms, patterns, and ways of thinking
that you would never have invented independently. Most of what feels like creativity is
actually recombination. The raw material for recombination is what you have consumed.
A narrow input diet produces a narrow creative range.

Risk and reward are correlated by design, but the correlation is not tight. You can take
large risks and receive small rewards — bad luck, wrong timing, flawed execution. You can
take small risks and receive large rewards — being in the right place at the right time
with just barely enough ability to exploit the opportunity. What you cannot do, over the
long run, is take no risks and receive large rewards. The mechanism is simple: anyone
offering guaranteed high returns is either lying or bearing risk themselves that they
have obscured from you.

Documentation is the tax you pay on future confusion. Like all taxes, it is unpleasant
to pay and clearly necessary when you see what happens when it goes uncollected.
Undocumented code is not a problem until it is, and then it is a large problem. The
function seemed obvious when you wrote it at 2am in a burst of inspiration. Six months
later, to a colleague who was not there, it is a mystery. The comment that felt
superfluous then would have been a gift.

Iteration beats planning in almost every domain where the outcome is uncertain and
feedback is available. Planning is useful for anticipating resource requirements, for
aligning teams around shared goals, for identifying obvious failure modes before you
commit irreversible resources. But planning cannot substitute for the information that
only comes from doing. The map is not the territory, and the plan is not the product.

The best argument for democracy is not that it produces optimal outcomes — it often does
not. The best argument is that it provides a peaceful mechanism for replacing leaders who
are failing. In systems without such mechanisms, bad leaders persist until their failures
are catastrophic enough to trigger violent removal. Democracy institutionalizes the
corrective feedback loop. It is a technology for error correction, not for optimization.

Consider the network effects that compound over time in any knowledge-intensive field.
The researcher who publishes early builds a reputation that attracts collaborators, who
bring complementary skills, who produce better work, which attracts better collaborators.
This is not mere Matthew effect — it is an information economy where your visible output
is simultaneously advertising, credentialing, and knowledge sharing. Withholding your
work keeps it safe from criticism but cuts you off from the compounding returns of the
network.

There is a certain kind of technical debt that accumulates not from bad decisions but
from good decisions made in a context that no longer exists. The schema made sense when
your data was small. The monolith made sense when your team was three people. The bash
script made sense when you ran it once a week by hand. Technical debt of this kind is
not evidence of past incompetence; it is evidence of past success. Systems that are never
used never accumulate technical debt.

What distinguishes expert judgment from novice judgment is not access to better
information but a different relationship with uncertainty. Novices demand certainty before
acting. Experts act under uncertainty because they understand that waiting for certainty
means waiting forever, and that action, even imperfect action, generates information that
passive waiting never does. The expert's edge is calibration: knowing which uncertainties
are decision-relevant and which can be bracketed, which estimates to trust and which to
widen, when more information will help and when it will merely delay.
""".strip()


def _get_tokenizer(encoding: str = "cl100k_base") -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding)


def _build_corpus_tokens(tokenizer: tiktoken.Encoding) -> list[int]:
    """Tokenize the full corpus once."""
    return tokenizer.encode(_CORPUS)


def generate_filler(target_tokens: int, encoding: str = "cl100k_base") -> str:
    """Return filler text that is exactly *target_tokens* tokens long.

    Tiles the corpus as many times as needed, then truncates precisely.
    """
    tokenizer = _get_tokenizer(encoding)
    corpus_tokens = _build_corpus_tokens(tokenizer)

    if len(corpus_tokens) == 0:
        raise RuntimeError("Empty corpus — cannot generate filler text.")

    # Tile the corpus until we have enough tokens.
    tiled: list[int] = []
    while len(tiled) < target_tokens:
        tiled.extend(corpus_tokens)

    trimmed = tiled[:target_tokens]
    return tokenizer.decode(trimmed)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Return the number of tokens in *text* using the specified tiktoken encoding."""
    tokenizer = _get_tokenizer(encoding)
    return len(tokenizer.encode(text))


def build_context(
    filler_tokens: int,
    needle: str,
    depth_pct: float,
    encoding: str = "cl100k_base",
) -> tuple[str, int]:
    """Build a context document with the needle inserted at *depth_pct* (0..100).

    Returns (full_document, actual_token_count).
    """
    tokenizer = _get_tokenizer(encoding)
    needle_tokens = tokenizer.encode(needle)
    needle_len = len(needle_tokens)

    filler_needed = max(0, filler_tokens - needle_len)
    filler = generate_filler(filler_needed, encoding)
    filler_tok = tokenizer.encode(filler)

    # Where to insert the needle (by token index)
    insert_at = int(len(filler_tok) * (depth_pct / 100.0))
    insert_at = max(0, min(insert_at, len(filler_tok)))

    combined_tokens = filler_tok[:insert_at] + needle_tokens + filler_tok[insert_at:]
    full_text = tokenizer.decode(combined_tokens)
    return full_text, len(combined_tokens)
