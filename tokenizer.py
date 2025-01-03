# Llama 3.1 Tokenizer (with tiktoken)

import os
from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

# The tiktoken tokenizer can handle <=400k chars without
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000

class Tokenizer:
    """
    A tokenizer for converting between strings and tokenized integer sequences.

    This implementation uses `tiktoken` and includes special tokens for specific use cases.
    """

    # Define special tokens and regular expression for tokenization
    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256  # Number of reserved special tokens
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # Tokenization pattern

    def __init__(self, model_path: str):
        """
        Initialize the tokenizer with a model file.

        Args:
            model_path: Path to the BPE model file.
        """
        # Ensure the model file exists
        assert os.path.isfile(model_path), model_path

        # Load mergeable ranks for BPE encoding
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)

        # Define a set of special tokens
        special_tokens = [
            "<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|step_id|>",
            "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>",
        ]
        # Add reserved tokens for additional special cases
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        # Map special tokens to their IDs
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        # Create the tiktoken encoding model
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        # Set tokenizer properties
        self.n_words: int = num_base_tokens + len(special_tokens)
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]  # Beginning of sequence
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]  # End of sequence
        self.eot_id: int = self.special_tokens["<|eot_id|>"]  # End of turn
        self.eom_id: int = self.special_tokens["<|eom_id|>"]  # End of message
        self.python_tag_id = self.special_tokens["<|python_tag|>"]  # Python code tag
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]  # Padding token
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s: The input string to encode.
            bos: Whether to prepend the beginning-of-sequence token.
            eos: Whether to append the end-of-sequence token.
            allowed_special: Special tokens allowed in the input string.
            disallowed_special: Special tokens that raise an error if present in the input.

        Returns:
            A list of token IDs representing the encoded string.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        # Split input into manageable substrings
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            # Encode each substring and add to token list
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:  # Add beginning-of-sequence token
            t.insert(0, self.bos_id)
        if eos:  # Add end-of-sequence token
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            t: Sequence of token IDs to decode.

        Returns:
            The decoded string.
        """
        return self.model.decode(cast(List[int], t))  # Safe typecasting

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits a string into substrings with limited consecutive whitespace or non-whitespace.

        Args:
            s: Input string to split.
            max_consecutive_slice_len: Maximum length of consecutive whitespaces or non-whitespaces.

        Yields:
            Substrings with limited consecutive whitespace or non-whitespace characters.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:  # Check if type changes
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]  # Yield the substring
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]  # Yield the final substring

