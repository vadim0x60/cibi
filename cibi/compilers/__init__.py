class Language:
    # We are enforcing the norm that EOS = 0 in all languages
    # This is assumption is important for genetic operators
    eos_int = 0

    def int_to_char(self, code_indices):
        code = ''.join(self.token_space[i] for i in code_indices)
        return code

    def char_to_int(self, code):
        code_indices = [self.token_ids[c] for c in code]
        return code_indices

    def prune(self, code):
        return code

    def __init__(self, alphabet, eos_char):
        assert eos_char not in alphabet

        # eos_int = 0
        self.eos_char = eos_char
        self.alphabet = alphabet
        self.token_space = eos_char + alphabet
        self.token_ids = dict([(c, i) for i, c in enumerate(self.token_space)])