def split_string_tokens(vocab: dict[str, int]) -> tuple[set[int], set[int]]:
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
