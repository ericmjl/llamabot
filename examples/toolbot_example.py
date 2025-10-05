#!/usr/bin/env python3
"""
ToolBot Example: Data Analysis Workflow

This example demonstrates how to use ToolBot for data analysis tasks.
It shows how ToolBot can execute Python code and work with global variables.
"""

import llamabot as lmb
import pandas as pd
import numpy as np
from llamabot.components.tools import write_and_execute_code


def main():
    """Demonstrate ToolBot usage for data analysis."""

    # Create sample data
    print("Creating sample data...")
    customer_data = pd.DataFrame(
        {
            "customer_id": range(1, 101),
            "age": np.random.randint(18, 80, 100),
            "income": np.random.randint(20000, 150000, 100),
            "purchase_amount": np.random.randint(10, 1000, 100),
            "region": np.random.choice(["North", "South", "East", "West"], 100),
        }
    )

    print(f"Created dataset with {len(customer_data)} customers")
    print(f"Columns: {list(customer_data.columns)}")
    print()

    # Create a ToolBot for data analysis
    print("Initializing ToolBot...")
    bot = lmb.ToolBot(
        system_prompt="""
        You are a data analyst assistant. You have access to customer_data DataFrame
        and can execute Python code to perform analysis. Focus on providing insights
        about customer demographics, purchasing patterns, and regional differences.
        """,
        model_name="gpt-4.1",
        tools=[write_and_execute_code(globals_dict=globals())],
    )

    # Perform various analyses
    analyses = [
        "Calculate the average age and income by region",
        "Find the top 10 customers by purchase amount",
        "Create a correlation matrix between age, income, and purchase amount",
        "Generate summary statistics for all numeric columns",
    ]

    print("Performing analyses...")
    for i, analysis in enumerate(analyses, 1):
        print(f"\n--- Analysis {i}: {analysis} ---")
        try:
            response = bot(analysis)
            print("Tool calls returned:", response)
        except Exception as e:
            print(f"Error during analysis: {e}")

    print("\n--- Example completed ---")


if __name__ == "__main__":
    main()
