import codecs

def _load_vocab(vocab_path, name, counters, min_freq):
        # counters changes in place
        vocab = _read_vocab_file(vocab_path, name)
        vocab_size = len(vocab)
        logger.info('Loaded %s vocab has %d tokens.' % (name, vocab_size))
        for i, token in enumerate(vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counters[name][token] = vocab_size - i + min_freq
        return vocab, vocab_size


def _read_vocab_file(vocab_path, tag):
    """Loads a vocabulary from the given path.

    Args:
        vocab_path (str): Path to utf-8 text file containing vocabulary.
            Each token should be on a line by itself. Tokens must not
            contain whitespace (else only before the whitespace
            is considered).
        tag (str): Used for logging which vocab is being read.
    """

    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]
