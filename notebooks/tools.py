# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.64.0",
#     "llamabot==0.13.5",
#     "marimo",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from llamabot.components.tools import function_to_dict
    import llamabot as lmb

    return function_to_dict, lmb


@app.cell
def _(lmb):
    from typing import Dict

    @lmb.tool
    def calculate_compound_interest(
        principal: float, rate: float, time: float, compound_frequency: int = 1
    ) -> Dict[str, float]:
        """
        Calculate compound interest using the standard compound interest formula.

        This function computes the final amount after compound interest is applied
        to a principal amount over a specified time period.

        If we have no other notes...

        Parameters
        ----------
        principal : float
            The initial amount of money (principal) invested or borrowed.
        rate : float
            The annual interest rate as a decimal (e.g., 0.05 for 5%).
        time : float
            The time period in years for which the money is invested or borrowed.
        compound_frequency : int, optional
            The number of times interest is compounded per year, by default 1.

        Returns
        -------
        Dict[str, float]
            A dictionary containing:
            - 'final_amount': The total amount after compound interest
            - 'interest_earned': The total interest earned
            - 'effective_rate': The effective annual interest rate

        Raises
        ------
        ValueError
            If any of the input parameters are negative or if compound_frequency is zero.

        Examples
        --------
        >>> result = calculate_compound_interest(1000, 0.05, 2, 12)
        >>> print(f"Final amount: ${result['final_amount']:.2f}")
        Final amount: $1105.16

        >>> result = calculate_compound_interest(5000, 0.08, 10, 4)
        >>> print(f"Interest earned: ${result['interest_earned']:.2f}")
        Interest earned: $6020.20

        Notes
        -----
        The compound interest formula used is:
        A = P(1 + r/n)^(nt)

        Where:
        - A = final amount
        - P = principal
        - r = annual interest rate
        - n = compound frequency
        - t = time in years
        """
        # Validate inputs
        if principal < 0 or rate < 0 or time < 0:
            raise ValueError("Principal, rate, and time must be non-negative")
        if compound_frequency <= 0:
            raise ValueError("Compound frequency must be positive")

        # Calculate compound interest
        final_amount = principal * (1 + rate / compound_frequency) ** (
            compound_frequency * time
        )
        interest_earned = final_amount - principal
        effective_rate = (final_amount / principal) ** (1 / time) - 1 if time > 0 else 0

        return {
            "final_amount": round(final_amount, 2),
            "interest_earned": round(interest_earned, 2),
            "effective_rate": round(effective_rate, 4),
        }

    return (calculate_compound_interest,)


@app.cell
def _(calculate_compound_interest, function_to_dict):
    function_to_dict(calculate_compound_interest)
    return


@app.cell
def _(calculate_compound_interest):
    calculate_compound_interest.json_schema
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
