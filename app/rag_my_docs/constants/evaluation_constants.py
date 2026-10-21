from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _EvaluationConstants:
    # The test set, as (question, documents, facts, expects_answer):
    # - documents: retrieval is right when a passage comes from any of them —
    #   Publication 17 repeats much of what the shorter publications say;
    # - facts: each must be in the answer, case aside; "a|b" accepts either;
    # - expects_answer: False for a question the publications cannot answer,
    #   where the right answer says so.
    # Every fact was read in the publication's 2025 revision (CorpusConstants
    # pins them): most are figures for 2025, which a model alone does not know.
    cases: tuple[tuple[str, tuple[str, ...], tuple[str, ...], bool], ...] = (
        (
            "What is the 2025 standard deduction for a single filer?",
            (
                "Publication 501 — Dependents, Standard Deduction, and Filing Information",
                "Publication 17 — Your Federal Income Tax",
            ),
            ("15,750",),
            True,
        ),
        (
            "What is the standard mileage rate for business use of a car in 2025?",
            (
                "Publication 463 — Travel, Gift, and Car Expenses",
                "Publication 334 — Tax Guide for Small Business",
                "Publication 17 — Your Federal Income Tax",
            ),
            ("70 cents|0.70",),
            True,
        ),
        (
            "How much can I deduct for a business gift to one client?",
            ("Publication 463 — Travel, Gift, and Car Expenses", "Publication 334 — Tax Guide for Small Business"),
            ("25",),
            True,
        ),
        (
            "What share of business meal expenses can I deduct?",
            ("Publication 463 — Travel, Gift, and Car Expenses", "Publication 334 — Tax Guide for Small Business"),
            ("50%|50 percent",),
            True,
        ),
        (
            "What is the IRA contribution limit for 2025 if I am 50 or older?",
            (
                "Publication 590-A — Contributions to Individual Retirement Arrangements (IRAs)",
                "Publication 17 — Your Federal Income Tax",
            ),
            ("8,000",),
            True,
        ),
        (
            "What will the IRA contribution limit be in 2026?",
            ("Publication 590-A — Contributions to Individual Retirement Arrangements (IRAs)",),
            ("7,500",),
            True,
        ),
        (
            "Above what share of my AGI can medical expenses be deducted?",
            ("Publication 502 — Medical and Dental Expenses", "Publication 17 — Your Federal Income Tax"),
            ("7.5%|7.5 percent",),
            True,
        ),
        (
            "What is the most child care expense I can count for two children?",
            ("Publication 503 — Child and Dependent Care Expenses", "Publication 17 — Your Federal Income Tax"),
            ("6,000",),
            True,
        ),
        (
            "With the simplified method for a home office, what is the rate and the largest area allowed?",
            ("Publication 587 — Business Use of Your Home", "Publication 334 — Tax Guide for Small Business"),
            ("$5", "300 square feet"),
            True,
        ),
        (
            "What is the maximum American opportunity credit per student?",
            ("Publication 970 — Tax Benefits for Education", "Publication 17 — Your Federal Income Tax"),
            ("2,500",),
            True,
        ),
        (
            "What is the maximum lifetime learning credit?",
            ("Publication 970 — Tax Benefits for Education", "Publication 17 — Your Federal Income Tax"),
            ("2,000",),
            True,
        ),
        (
            "For a single filer, between which MAGI amounts is the student loan interest deduction phased out?",
            ("Publication 970 — Tax Benefits for Education", "Publication 17 — Your Federal Income Tax"),
            ("85,000", "100,000"),
            True,
        ),
        (
            "From what amount of net earnings from self-employment must I file a return?",
            (
                "Publication 334 — Tax Guide for Small Business",
                "Publication 17 — Your Federal Income Tax",
                "Publication 501 — Dependents, Standard Deduction, and Filing Information",
            ),
            ("400",),
            True,
        ),
        ("When is the 2025 Form 1040 due?", ("Publication 17 — Your Federal Income Tax",), ("April 15, 2026",), True),
        (
            "What is the general limit on deducting charitable contributions, as a share of AGI?",
            ("Publication 526 — Charitable Contributions", "Publication 17 — Your Federal Income Tax"),
            ("60%|60 percent",),
            True,
        ),
        (
            "When is a vacation home I rent out considered used as a home?",
            ("Publication 527 — Residential Rental Property",),
            ("14 days", "10%|10 percent"),
            True,
        ),
        # Pages of numbers: the row's figures say nothing a question resembles,
        # so retrieval by meaning rarely finds the line. Expected to fail.
        (
            "From the 2025 tax table, what is the tax for a single filer with taxable income of $52,320?",
            ("Publication 1040 — Tax and Earned Income Credit Tables",),
            ("6,426",),
            True,
        ),
        ("What is the California state income tax rate?", (), (), False),
        ("How do I renew a U.S. passport?", (), (), False),
    )
    run_out_of_range: str = "This test question does not exist."


EvaluationConstants = _EvaluationConstants()
