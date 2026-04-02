from typing import List, Dict

class WordPieceTokenizer:
    """
    WordPiece tokenizer for BERT.
    """
    
    def __init__(self, vocab: Dict[str, int], unk_token: str = "[UNK]", max_word_len: int = 100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_word_len = max_word_len
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into WordPiece tokens.
        """
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word into subwords.
        """
        
        out = []
        n = len(word)
        lt = 0
        rt = n
        prefix = "##"
        cur = ""

        while lt<rt:

            cur = word[lt:rt]
            if lt != 0:
                cur = prefix + word[lt:rt]
                
            if cur in self.vocab:
                out.append(cur)

                lt = rt
                rt = n
            else:
                rt -= 1

        return ["[UNK]"] if "".join(out).replace("##", "") != word else out
                