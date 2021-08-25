class Language:
    def bf_int_to_char(self, code_indices):
        code = ''.join(self.__int_to_char__[i] for i in code_indices)
        return code

    def bf_char_to_int(self, code):
        code_indices = [self.__char_to_int__[c] for c in code]
        return code_indices

    def __init__(self, alphabet, eos_char):
        assert eos_char not in alphabet

        # We are enforcing the norm that EOS = 0 in all languages
        # This is assumption is important for genetic operators
        self.__alphabet__ = alphabet
        self.__int_to_char__ = eos_char + alphabet
        self.__char_to_int__ = dict([(c, i) for i, c in enumerate(self.__int_to_char__)])