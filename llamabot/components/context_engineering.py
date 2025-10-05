"""Convenience functions to engineer context for the LLM."""


def describe_dataframes_in_globals(globals_dict):
    """Describe DataFrames and nested data structures in a globals dictionary.

    This function analyzes all variables in the provided globals dictionary
    and generates descriptive summaries for DataFrames and nested data structures
    (dictionaries, lists, tuples). For DataFrames, it provides detailed column
    analysis including data types, unique ratios, and statistical summaries.

    :param globals_dict: Dictionary containing global variables to analyze
    :type globals_dict: dict
    :return: String containing formatted descriptions of all found data structures
    :rtype: str
    """
    import pandas as pd

    def describe_df(df):
        """Generate detailed description of a DataFrame.

        Analyzes each column in the DataFrame to determine its data type
        (numeric, datetime, identifier, or categorical) and provides
        statistical summaries.

        :param df: DataFrame to analyze
        :type df: pandas.DataFrame
        :return: Formatted string description of the DataFrame
        :rtype: str
        """
        desc = []
        desc.append(f"\nDataFrame Summary:\n{df.describe(include='all')}")
        for column in df.columns:
            non_null_series = df[column].dropna()

            # More robust check for numeric conversion
            if pd.to_numeric(non_null_series, errors="coerce").notnull().all():
                desc.append(f"\nColumn '{column}' is numeric.")
                continue

            # Stricter datetime check allowing reasonable conversion
            try:
                pd.to_datetime(non_null_series, errors="raise")
                desc.append(f"\nColumn '{column}' is a datetime.")
                continue
            except (ValueError, TypeError):
                pass

            # Unique ratio for identifier-like columns
            unique_ratio = (
                non_null_series.nunique() / len(non_null_series)
                if len(non_null_series) > 0
                else 0
            )
            if unique_ratio >= 0.8:
                desc.append(
                    f"\nColumn '{column}' is likely an identifier based on unique ratio."
                )
            else:
                # Get unique values for categorical columns
                unique_values = non_null_series.unique()
                unique_count = len(unique_values)

                # Show all categorical values
                values_str = ", ".join([str(val) for val in unique_values])
                desc.append(
                    f"\nColumn '{column}' is categorical with {unique_count} unique values: {values_str}"
                )

        return "\n".join(desc)

    def recurse_structure(structure, description=""):
        """Recursively analyze nested data structures.

        Traverses through nested data structures (DataFrames, dictionaries,
        lists, tuples) and generates descriptions for any DataFrames found.

        :param structure: Data structure to analyze
        :type structure: pandas.DataFrame, dict, list, or tuple
        :param description: Accumulated description string
        :type description: str
        :return: Updated description string with found DataFrames
        :rtype: str
        """
        if isinstance(structure, pd.DataFrame):
            return f"\nDataFrame found: \n{describe_df(structure)}"
        elif isinstance(structure, dict):
            for key, value in structure.items():
                description += recurse_structure(value)
        elif isinstance(structure, (list, tuple)):
            for item in structure:
                description += recurse_structure(item)
        return description

    ret_str = ""
    for var_name, var_value in globals_dict.items():
        if isinstance(var_value, (pd.DataFrame, dict, list, tuple)):
            ret_str += f"\nVariable '{var_name}': {recurse_structure(var_value)}"

    return ret_str
