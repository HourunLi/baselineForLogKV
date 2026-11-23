"""RULER's niah_multiquery with its four keys asked one per turn on one cache (SCBench's multi-turn mode).

niah_multiquery hides the values of four keys in an essay haystack and asks for all four at the end, so the
question is in the prompt when the context is cached (and the query-aware cue sees it). Here the samples come
from the same generator, with the same haystack, needles and string-match scoring, but are cut before the
question: eval.py caches the context with no question known, then asks the keys one per turn, each answer
greedy to the end of its line and kept in the cache before the next question (LogKVLM._generate_turns).
turn1 ... turn4 score each turn and acc their mean; next to niah_multiquery at the same length, the drop is
what the cache loses when it cannot know the question, and turn1 against turn4 what later turns add.
"""
from lm_eval.tasks.ruler.common_utils import get_tokenizer
from lm_eval.tasks.ruler.niah_utils import TEMPLATE, download_dataset
from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

SEPARATOR = "<|turn|>"  # between the context and each question; eval.py splits the prompt on it
TURNS = 4
OPENING = "\nWhat are all the special magic numbers for "
QUESTION = ("\nWhat is the special magic number for {key} mentioned in the provided text? "
            "The special magic number for {key} mentioned in the provided text is")


def keys_of(gen_prefix):
    """The queried keys, in order, from niah_multiquery's generation prefix."""
    query = gen_prefix.split(" for ", 1)[1].rsplit(" mentioned in the provided text", 1)[0]
    return query.replace(", and ", ", ").split(", ")


def turns_of(sample):
    """One niah_multiquery sample as a context and one question per key."""
    keys = keys_of(sample["gen_prefix"])
    assert len(keys) == len(sample["outputs"]) == TURNS, sample["gen_prefix"]
    context = sample["input"][:sample["input"].rindex(OPENING)]
    return dict(input=SEPARATOR.join([context] + [QUESTION.format(key=key) for key in keys]),
                outputs=sample["outputs"], keys=keys, length=sample["length"], max_length=sample["max_length"])


def build(length, kwargs, num_samples=100):
    # lm-eval caches get_tokenizer by its arguments: pass the tokenizer's only, not the rest of the metadata
    # (max_seq_lengths is a list, which cannot be a cache key).
    tokenizer = get_tokenizer(**{k: kwargs[k] for k in ("tokenizer", "pretrained") if kwargs.get(k)})
    # Room for four questions and answers instead of one question and its answer.
    samples = generate_samples(get_haystack(type_haystack="essay"), max_seq_length=length, template=TEMPLATE,
                               type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                               num_needle_q=TURNS, num_samples=num_samples, tokens_to_generate=384,
                               TOKENIZER=tokenizer)
    return download_dataset([[turns_of(sample) for sample in samples]])


def niah_turns_32k(**kwargs):
    return build(32768, kwargs)


def niah_turns_64k(**kwargs):
    return build(65536, kwargs)


def niah_turns_128k(**kwargs):
    return build(131072, kwargs)


def niah_turns_256k(**kwargs):
    return build(262144, kwargs)


def niah_turns_512k(**kwargs):
    return build(524288, kwargs)


def niah_turns_1m(**kwargs):
    return build(1048576, kwargs)


def process_results(doc, results):
    answers = results[0].split(SEPARATOR)
    answers += [""] * (TURNS - len(answers))
    scores = [float(value.lower() in answer.lower()) for value, answer in zip(doc["outputs"], answers)]
    return {"acc": sum(scores) / TURNS, **{f"turn{i + 1}": score for i, score in enumerate(scores)}}
