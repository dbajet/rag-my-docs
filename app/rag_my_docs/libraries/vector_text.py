from __future__ import annotations


class VectorText:
    """A vector as pgvector reads it, `[0.1,-0.2,…]`, passed as a parameter and
    cast with `::vector` in the SQL: no adapter package between the code and
    the database."""

    @classmethod
    def of(cls, values: list[float]) -> str:
        return "[" + ",".join(f"{value:.8g}" for value in values) + "]"
