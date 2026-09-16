"""Separate string-content tokens from tokens containing quotes."""


def split_string_tokens(vocab: dict[str, int]) -> tuple[set[int], set[int]]:
    """Return sets of content-token IDs and quote-bearing token IDs.

    Skip empty token text and classify every other token by whether it
    contains a double quote. Do not validate JSON escapes or control text.

    Args:
        vocab: Mapping of vocabulary token text to token IDs.

    Returns:
        Pair of ID sets: tokens without quotes, then tokens containing quotes.
            Empty token text is excluded.
    """
    content_ids = set()
    quote_ids = set()

    for text, token in vocab.items():
        if not text:
            continue
        if '"' in text:
            quote_ids.add(token)
        else:
            content_ids.add(token)
    return content_ids, quote_ids
