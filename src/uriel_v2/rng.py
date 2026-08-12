from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable


MASK_64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    """Small deterministic PRNG with stable behavior across Python versions."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK_64

    def next_u64(self) -> int:
        self.state = (self.state + SPLITMIX_GAMMA) & MASK_64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK_64
        return (value ^ (value >> 31)) & MASK_64

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper는 양수여야 합니다")
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper


def stable_seed(namespace: str, *parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=list)
    digest = hashlib.blake2b(digest_size=8, person=b"uriel-v2-seed")
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(payload.encode("utf-8"))
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def generate_numbers(seed: int, *, count: int = 6, maximum: int = 45) -> tuple[int, ...]:
    if count <= 0 or maximum < count:
        raise ValueError("count는 1 이상이며 maximum 이하여야 합니다")
    rng = SplitMix64(seed)
    pool = list(range(1, maximum + 1))
    for index in range(count):
        selected = index + rng.randbelow(maximum - index)
        pool[index], pool[selected] = pool[selected], pool[index]
    return tuple(sorted(pool[:count]))


def numbers_mask(numbers: Iterable[int]) -> int:
    mask = 0
    for number in numbers:
        mask |= 1 << number
    return mask
