# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]",
#     "pydantic==2.12.3",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import llamabot as lmb

    return (lmb,)


@app.cell
def _():
    from llamabot.components.tools import write_and_execute_code

    return


@app.cell(hide_code=True)
def _():
    from enum import Enum
    from pydantic import BaseModel, Field, validator
    from typing import List, Optional, Dict, Any

    class ResponseType(str, Enum):
        """Types of response variables for GLM selection."""

        GAUSSIAN = "gaussian"
        POISSON = "poisson"
        BINOMIAL = "binomial"
        NEGATIVE_BINOMIAL = "negative_binomial"
        GAMMA = "gamma"
        BETA = "beta"

    class FactorType(str, Enum):
        """Types of experimental factors."""

        TREATMENT = "treatment"  # Primary effects of interest (drugs, doses, etc.)
        NUISANCE = "nuisance"  # Sources of variation to control for (plate, day, operator effects)
        BLOCKING = "blocking"  # Factors used for blocking/stratification in experimental design
        COVARIATE = "covariate"  # Continuous covariates
        REPLICATE = "replicate"  # Technical or biological replicates

    class InteractionType(str, Enum):
        """Types of scientific interactions."""

        GENE_GENE = (
            "gene_gene"  # Gene-gene interactions (epistasis, genetic interactions)
        )
        PROTEIN_RNA = (
            "protein_rna"  # Protein-RNA interactions (translation regulation, splicing)
        )
        PROTEIN_DNA = "protein_dna"  # Protein-DNA interactions (transcription regulation, chromatin)
        CELL_CELL = "cell_cell"  # Cell-cell interactions (signaling, adhesion)
        DRUG_TARGET = (
            "drug_target"  # Drug-target interactions (pharmacology, drug response)
        )
        ENVIRONMENT_GENE = "environment_gene"  # Environment-gene interactions (GxE)
        TIME_TREATMENT = (
            "time_treatment"  # Time-treatment interactions (temporal effects)
        )
        DOSE_RESPONSE = (
            "dose_response"  # Dose-response interactions (nonlinear effects)
        )
        OTHER = "other"  # Other scientific interactions

    class ExperimentalFactor(BaseModel):
        """Represents a single experimental factor."""

        name: str = Field(..., description="Name of the factor")
        factor_type: FactorType = Field(..., description="Type of factor")
        levels: List[str] = Field(..., description="Levels of the factor")
        description: Optional[str] = Field(
            None, description="Description of the factor"
        )

        # For continuous covariates
        is_continuous: bool = Field(
            False, description="Whether this is a continuous factor"
        )
        range_min: Optional[float] = Field(
            None, description="Minimum value for continuous factors"
        )
        range_max: Optional[float] = Field(
            None, description="Maximum value for continuous factors"
        )

        @validator("levels")
        def validate_levels(cls, v, values):
            """Validate the levels of the factor."""
            if not values.get("is_continuous") and len(v) < 2:
                raise ValueError("Categorical factors must have at least 2 levels")
            return v

    class ReplicateStructure(BaseModel):
        """Describes how replicates are nested in the experimental design."""

        replicate_type: str = Field(
            ..., description="Type of replicate (e.g., 'technical', 'biological')"
        )
        replicates_per_unit: int = Field(
            ..., description="Number of replicates per experimental unit"
        )
        nested_under: Optional[str] = Field(
            None,
            description="What the replicates are nested under (e.g., 'plate', 'mouse')",
        )
        description: Optional[str] = Field(
            None, description="Description of the replicate structure"
        )

    class PotentialInteraction(BaseModel):
        """Describes a potential interaction between experimental factors based on scientific knowledge."""  # noqa: E501

        factor1: str = Field(
            ..., description="Name of the first factor in the interaction"
        )
        factor2: str = Field(
            ..., description="Name of the second factor in the interaction"
        )
        interaction_type: InteractionType = Field(
            ..., description="Type of scientific interaction"
        )
        scientific_justification: str = Field(
            ...,
            description="Scientific justification for why this interaction might be important",  # noqa: E501
        )
        biological_mechanism: Optional[str] = Field(
            None,
            description="Biological mechanism underlying the potential interaction",
        )
        evidence_strength: str = Field(
            default="moderate",
            description="Strength of evidence for this interaction (weak, moderate, strong)",  # noqa: E501
        )
        should_include: bool = Field(
            default=True,
            description="Whether this interaction should be included in the statistical model",  # noqa: E501
        )
        notes: Optional[str] = Field(
            None, description="Additional notes about the interaction"
        )

    class ExperimentDescription(BaseModel):
        """
        Basic experiment description focused on PyMC model construction.

        This captures the core elements needed to construct PyMC coordinates
        and dimensions for R2D2 models.
        """

        # Core elements for PyMC model construction
        response: str = Field(..., description="Response variable name")
        response_type: ResponseType = Field(
            ..., description="Type of response for GLM selection"
        )

        # Experimental factors organized by type
        factors: List[ExperimentalFactor] = Field(
            default_factory=list, description="All experimental factors"
        )

        # Variables that become PyMC coordinates/dimensions
        units: Optional[str] = Field(
            None, description="Experimental units (e.g., 'mice', 'plots')"
        )
        timepoints: Optional[List[str]] = Field(
            None, description="Timepoints for longitudinal data"
        )

        # Replicate structure
        replicate_structure: Optional[ReplicateStructure] = Field(
            None, description="How replicates are structured in the experiment"
        )

        # Potential interactions based on scientific knowledge
        potential_interactions: List[PotentialInteraction] = Field(
            default_factory=list,
            description="Potential interactions between factors based on general scientific knowledge",  # noqa: E501
        )

        # Additional context
        aim: Optional[str] = Field(None, description="Primary aim of the experiment")
        description: Optional[str] = Field(
            None, description="Original experiment description"
        )

        def get_treatments(self) -> Dict[str, List[str]]:
            """Get treatment factors and their levels."""
            treatments = {}
            for factor in self.factors:
                if factor.factor_type == FactorType.TREATMENT:
                    treatments[factor.name] = factor.levels
            return treatments

        def get_nuisance_factors(self) -> Dict[str, List[str]]:
            """Get nuisance factors and their levels."""
            nuisance_factors = {}
            for factor in self.factors:
                if factor.factor_type == FactorType.NUISANCE:
                    nuisance_factors[factor.name] = factor.levels
            return nuisance_factors

        def get_blocking_factors(self) -> Dict[str, List[str]]:
            """Get blocking factors and their levels."""
            blocking_factors = {}
            for factor in self.factors:
                if factor.factor_type == FactorType.BLOCKING:
                    blocking_factors[factor.name] = factor.levels
            return blocking_factors

        def get_covariates(self) -> List[str]:
            """Get continuous covariate names."""
            return [
                factor.name
                for factor in self.factors
                if factor.factor_type == FactorType.COVARIATE
            ]

        def get_replicate_factors(self) -> Dict[str, List[str]]:
            """Get replicate factors and their levels."""
            replicate_factors = {}
            for factor in self.factors:
                if factor.factor_type == FactorType.REPLICATE:
                    replicate_factors[factor.name] = factor.levels
            return replicate_factors

        def get_treatment_factors(self) -> List[str]:
            """Get names of treatment factors."""
            return [
                factor.name
                for factor in self.factors
                if factor.factor_type == FactorType.TREATMENT
            ]

        def get_interaction_terms(self) -> List[PotentialInteraction]:
            """Get interaction terms that should be included in the model."""
            return [
                interaction
                for interaction in self.potential_interactions
                if interaction.should_include
            ]

        def get_interaction_coords(self) -> Dict[str, Any]:
            """Generate PyMC coordinates for interaction terms."""
            coords = {}
            for interaction in self.get_interaction_terms():
                # Create coordinate for the interaction term
                factor1_levels = next(
                    f.levels for f in self.factors if f.name == interaction.factor1
                )
                factor2_levels = next(
                    f.levels for f in self.factors if f.name == interaction.factor2
                )
                coords[f"{interaction.factor1}_{interaction.factor2}_interaction"] = [
                    f"{level1}_{level2}"
                    for level1 in factor1_levels
                    for level2 in factor2_levels
                ]
            return coords

        def get_pymc_coords(self) -> Dict[str, Any]:
            """
            Generate PyMC coordinates dictionary.

            Returns:
                Dict mapping coordinate names to coordinate values
            """
            coords = {
                "obs": None  # Will be set to range(len(data)) when data is available
            }

            # Add all categorical factors as coordinates
            for factor in self.factors:
                if not factor.is_continuous:
                    coords[factor.name] = factor.levels

            # Add unit coordinates
            if self.units:
                coords[self.units] = None  # Will be set to unique values from data

            # Add time coordinates
            if self.timepoints:
                coords["timepoint"] = self.timepoints

            # Add interaction coordinates
            interaction_coords = self.get_interaction_coords()
            coords.update(interaction_coords)

            return coords

        def get_pymc_dims(self) -> Dict[str, List[str]]:
            """
            Generate PyMC dimensions mapping.

            Returns:
                Dict mapping variable names to their dimension lists
            """
            dims = {}

            # Response variable dimensions
            dims[self.response] = ["obs"]

            # Treatment effect dimensions
            for factor in self.factors:
                if factor.factor_type == FactorType.TREATMENT:
                    dims[f"{factor.name}_effect"] = [factor.name]

            # Nuisance factor dimensions (plate effects, day effects, operator effects)
            for factor in self.factors:
                if factor.factor_type == FactorType.NUISANCE:
                    dims[f"{factor.name}_effect"] = [factor.name]

            # Blocking factor dimensions (experimental design stratification)
            for factor in self.factors:
                if factor.factor_type == FactorType.BLOCKING:
                    dims[f"{factor.name}_effect"] = [factor.name]

            # Unit intercept dimensions
            if self.units:
                dims[f"{self.units}_intercepts"] = [self.units]

            # Time effect dimensions
            if self.timepoints:
                dims["timepoint_effect"] = ["timepoint"]

            # Interaction effect dimensions
            for interaction in self.get_interaction_terms():
                dims[f"{interaction.factor1}_{interaction.factor2}_interaction"] = [
                    f"{interaction.factor1}_{interaction.factor2}_interaction"
                ]

            return dims

        def get_variance_components(self) -> Dict[str, Any]:
            """
            Get variance component structure for R2D2 models.

            Returns:
                Dict with variance component information
            """
            components = []
            component_names = []

            # Add treatment components
            for factor in self.factors:
                if factor.factor_type == FactorType.TREATMENT:
                    components.append(
                        {
                            "name": factor.name,
                            "type": "treatment",
                            "dim": factor.name,
                            "levels": factor.levels,
                        }
                    )
                    component_names.append(factor.name)

            # Add nuisance components (plate, day, operator effects)
            for factor in self.factors:
                if factor.factor_type == FactorType.NUISANCE:
                    components.append(
                        {
                            "name": factor.name,
                            "type": "nuisance",  # Sources of variation to control for
                            "dim": factor.name,
                            "levels": factor.levels,
                        }
                    )
                    component_names.append(factor.name)

            # Add blocking components (experimental design stratification)
            for factor in self.factors:
                if factor.factor_type == FactorType.BLOCKING:
                    components.append(
                        {
                            "name": factor.name,
                            "type": "blocking",  # Experimental design blocks/strata
                            "dim": factor.name,
                            "levels": factor.levels,
                        }
                    )
                    component_names.append(factor.name)

            # Add unit components
            if self.units:
                components.append(
                    {
                        "name": f"{self.units}_intercepts",
                        "type": "unit",
                        "dim": self.units,
                        "levels": None,  # Will be determined from data
                    }
                )
                component_names.append(f"{self.units}_intercepts")

            # Add time components
            if self.timepoints:
                components.append(
                    {
                        "name": "timepoint_effect",
                        "type": "time",
                        "dim": "timepoint",
                        "levels": self.timepoints,
                    }
                )
                component_names.append("timepoint_effect")

            # Add interaction components
            for interaction in self.get_interaction_terms():
                components.append(
                    {
                        "name": f"{interaction.factor1}_{interaction.factor2}_interaction",
                        "type": "interaction",
                        "dim": f"{interaction.factor1}_{interaction.factor2}_interaction",
                        "levels": None,  # Will be determined from factor combinations
                        "interaction_type": interaction.interaction_type.value,
                        "scientific_justification": interaction.scientific_justification,
                    }
                )
                component_names.append(
                    f"{interaction.factor1}_{interaction.factor2}_interaction"
                )

            return {
                "n_components": len(components),
                "components": components,
                "component_names": component_names,
            }

        def with_interactions_enabled(
            self, enabled: bool = True
        ) -> "ExperimentDescription":
            """
            Create a copy of the experiment with interactions enabled or disabled.

            Args:
                enabled: Whether to enable interactions

            Returns:
                A new ExperimentDescription with interactions toggled
            """
            experiment_copy = self.model_copy(deep=True)
            for interaction in experiment_copy.potential_interactions:
                interaction.should_include = enabled
            return experiment_copy

        def enable_interactions(self) -> "ExperimentDescription":
            """Enable all interactions in the experiment."""
            return self.with_interactions_enabled(True)

        def disable_interactions(self) -> "ExperimentDescription":
            """Disable all interactions in the experiment."""
            return self.with_interactions_enabled(False)

    return (ExperimentDescription,)


@app.cell
def _(ExperimentDescription, lmb):
    @lmb.tool
    def extract_statistical_info(description: str) -> str:
        """Return an ExperimentDescription from the given experiment description string.

        Use this tool to parse out experimental design information and identify what might be missing.

        :param description: A text description of the experiment.
        :return: A dictionary capturing the statistical design.
        """
        try:
            extractor_bot = lmb.StructuredBot(
                system_prompt="You are an expert in experimental design. Parse the given experiment description and extract statistical design information. If any factors have insufficient levels, provide reasonable defaults or mark them as continuous variables. Always ensure categorical factors have at least 2 levels.",
                pydantic_model=ExperimentDescription,
            )
            result = extractor_bot(description)
            return result.json()
        except Exception as e:
            # Return a structured error response that the agent can work with
            return f"Error parsing experimental design: {str(e)}. The experiment description may be incomplete or ambiguous. Please provide more details about the experimental factors, treatments, and design structure."

    return (extract_statistical_info,)


@app.cell(hide_code=True)
def _(lmb):
    @lmb.tool
    def read_r2d2m2_summary():
        """Return a summary of the R2D2M2 model family and example PyMC implementation.

        Use this tool if asked how to build a statistical model for a given experiment design.

        The R2D2M2 model family allows us to construct generalized linear models using
        R-squared (R2) Dirichlet Decomposition (D2) for Multilevel Models (M2),
        in which we use variance decomposition to competitively assign prior variance
        across factors in an experiment. This approach helps to control overfitting
        and improve interpretability in complex hierarchical models, in particuarly, we
        can make claims such as, "60% of the variance in output is coming from nuisance factors
        such as temperature, and as such we need to improve temperature control to
        better measure the effect of molecule treatment."

        :return: A markdown string summarizing R2D2M2 and providing PyMC code.
        """
        return '''
    # R2D2M2: R2D2 for Multilevel Models - Summary and PyMC Implementation

    ## Overview

    The R2D2M2 prior extends the R2D2 framework to **M**ultilevel **M**odels (M2), addressing the challenge of joint regularization across both population-level effects and group-specific effects in hierarchical models. This work recognizes that standard independent priors on multilevel model parameters create unrealistic implied priors on R² that become increasingly concentrated near 1 as model complexity grows.

    ## Understanding Group-Specific vs Population-Level Effects

    **Traditional "Random Effects" Terminology vs. Clearer Language:**

    | Traditional Term | Clearer Term | What It Actually Is |
    |-----------------|-------------|-------------------|
    | "Fixed Effects" | **Population-Level Effects** | Effects that apply to the entire population |
    | "Random Effects" | **Group-Specific Effects** | Group-specific deviations from population average |
    | "Random Intercepts" | **Group-Specific Intercepts** | Each group gets its own baseline adjustment |
    | "Random Slopes" | **Group-Specific Slopes** | Each group gets different effect sizes for predictors |

    **Example:** In a study of student test scores across schools:
    - **Population-level effect of study hours**: Average effect across all schools
    - **School-specific intercept**: How much each school's average differs from overall average
    - **School-specific slope for study hours**: How much the effect of study hours varies by school

    ## Key Problem: Implied R² in Multilevel Models

    ### The Issue with Standard Priors

    In multilevel models with:
    - Population-level effects: `bᵢ ~ Normal(0, λ̃ᵢ²)`
    - Group-specific effects: `uᵢgⱼ ~ Normal(0, λ̃ᵢg²)`

    The implied prior on R² becomes highly concentrated near 1 as:
    1. Number of predictors `p` increases
    2. Number of grouping factors increases
    3. Number of levels per group increases

    This leads to overfitting-prone models that expect near-perfect fit a priori.

    ### Variance Decomposition Framework

    The total variance of the linear predictor (Equation 2 in R2D2M2 paper) is:
    ```
    Var(μ) = Σg∈G₀ λ̃₀g² + Σᵢ₌₁ᵖ σ²ₓᵢ (λ̃ᵢ² + Σg∈Gᵢ λ̃ᵢg²)
    ```

    This shows how variance components from different model levels contribute to the total explained variance.

    ## R2D2M2 Mathematical Framework

    ### Core Parameterization

    **1. Standardized Variance Decomposition (Equation 3):**
    ```
    λ̃ᵢ² = (σ²/σ²ₓᵢ) λᵢ²
    λ̃ᵢg² = (σ²/σ²ₓᵢ) λᵢg²
    ```

    **2. Total Standardized Explained Variance (Equation 4):**
    ```
    τ² = Σg∈G₀ λ₀g² + Σᵢ₌₁ᵖ (λᵢ² + Σg∈Gᵢ λᵢg²)
    (population-level + group-specific variance components)
    ```

    **3. R² Relationship (Equation 5):**
    ```
    R² = Var(μ)/(Var(μ) + σ²) = τ²/(τ² + 1)
    ```

    **4. Global-Local Decomposition (Equation 6):**
    ```
    λᵢ² = φᵢ τ²    (population-level variance components)
    λᵢg² = φᵢg τ²  (group-specific variance components)
    ```
    where `φ` is a simplex: `Σᵢ φᵢ + Σᵢ Σg∈Gᵢ φᵢg = 1`

    ### Prior Specification

    **Hierarchy (Section 2.2):**
    1. `R² ~ Beta(μᵣ², φᵣ²)` (mean-precision parameterization, Equation 7)
    2. `τ² = R²/(1-R²) ~ BetaPrime(μᵣ², φᵣ²)` (Equation 8)
    3. `φ ~ Dirichlet(α)` where `α` can be symmetric with concentration `a_π` (Section 2.2.1)
    4. `bᵢ ~ Normal(0, (σ²/σ²ₓᵢ) φᵢ τ²)` (population-level effects, Equation 9)
    5. `uᵢgⱼ ~ Normal(0, (σ²/σ²ₓᵢ) φᵢg τ²)` (group-specific effects, Equation 10)

    ### Key Innovation: Scale-Invariant Parameterization

    The factor `σ²/σ²ₓᵢ` ensures:
    - Parameters are comparable across different scales of predictors
    - Prior specification is invariant to scaling of `y` and `X`
    - `λᵢ` represents standardized variance components

    ## PyMC Implementation

    ```python
    # /// script
    # requires-python = ">=3.12"
    # dependencies = [
    #   "pymc",
    #   "numpy",
    #   "pandas",
    #   "pytensor",
    #   "scipy",
    # ]
    # ///

    import pymc as pm
    import numpy as np
    import pandas as pd
    import pytensor.tensor as pt
    from scipy.special import gamma as gamma_fn

    def beta_prime_logp(value, alpha, beta):
        """Log-probability for Beta Prime distribution."""
        logp = (
            (alpha - 1) * pt.log(value)
            - (alpha + beta) * pt.log(1 + value)
            + pt.log(gamma_fn(alpha + beta))
            - pt.log(gamma_fn(alpha))
            - pt.log(gamma_fn(beta))
        )
        return logp

    class BetaPrime(pm.Continuous):
        """Beta Prime distribution for PyMC."""

        def __init__(self, alpha, beta, **kwargs):
            self.alpha = pt.as_tensor_variable(alpha)
            self.beta = pt.as_tensor_variable(beta)
            super().__init__(**kwargs)

        def logp(self, value):
            return beta_prime_logp(value, self.alpha, self.beta)

    def r2d2m2_model(X, y, groups=None, r2_mean=0.5, r2_precision=2,
                     concentration=0.5, sigma_prior='half_student_t'):
        """
        R2D2M2 prior for multilevel models.

        Parameters:
        -----------
        X : array-like, shape (n, p)
            Fixed effects design matrix (standardization recommended for interpretability)
        y : array-like, shape (n,)
            Response vector
        groups : dict, optional
            Dictionary mapping group names to group indicators
            e.g., {'school': school_ids, 'class': class_ids}
        r2_mean : float
            Prior mean for R²
        r2_precision : float
            Prior precision for R² (controls concentration)
        concentration : float
            Dirichlet concentration parameter a_π (smaller = more sparse)
        sigma_prior : str
            Prior for residual standard deviation
        """
        n, p = X.shape

        # Setup grouping structure
        if groups is None:
            groups = {}

        n_groups = len(groups)
        group_sizes = {}
        for group_name, group_ids in groups.items():
            group_sizes[group_name] = len(np.unique(group_ids))

        # Total number of variance components:
        # p population effects + sum over (group intercepts + group slopes per group)
        n_components = p  # Population-level effects
        component_names = [f'population_{i}' for i in range(p)]

        for group_name in groups:
            n_components += 1  # Group-specific intercept
            n_components += p  # Group-specific slopes (assuming all predictors vary)
            component_names.append(f'{group_name}_intercept')
            component_names.extend([f'{group_name}_slope_{i}' for i in range(p)])

        # Define coordinates for model dimensions
        coords = {
            'predictors': [f'x_{i}' for i in range(p)],
            'obs': range(n),
            'components': component_names,
        }

        # Add group-specific coordinates
        for group_name, group_ids in groups.items():
            coords[f'{group_name}_levels'] = range(group_sizes[group_name])

        with pm.Model(coords=coords) as model:
            # Intercept (not regularized)
            b0 = pm.Normal("b0", mu=np.mean(y), sigma=np.std(y))

            # R² prior with mean-precision parameterization
            # Convert to standard alpha-beta parameterization
            alpha_r2 = r2_mean * r2_precision
            beta_r2 = (1 - r2_mean) * r2_precision

            r_squared = pm.Beta("r_squared", alpha=alpha_r2, beta=beta_r2)

            # Total standardized explained variance τ²
            tau_squared = pm.Deterministic("tau_squared", r_squared / (1 - r_squared))

            # Dirichlet allocation of variance components
            phi = pm.Dirichlet("phi", a=np.full(n_components, concentration), dims="components")

            # Residual standard deviation
            if sigma_prior == 'half_student_t':
                sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(y))
            else:
                sigma = pm.HalfNormal("sigma", sigma=np.std(y))

            sigma_squared = pm.Deterministic("sigma_squared", sigma**2)

            # Compute predictor standard deviations
            # If predictors are standardized, sigma_x = 1 for all predictors
            # Otherwise, use empirical standard deviations: sigma_x = np.std(X, axis=0)
            sigma_x = np.ones(p)  # Assumes standardized predictors

            # Population-level effects
            component_idx = 0

            # Scale factor for population-level effects
            population_scale = pm.Deterministic(
                "population_scale",
                pt.sqrt((sigma_squared / sigma_x**2) * phi[:p] * tau_squared),
                dims="predictors"
            )

            beta = pm.Normal("beta", mu=0, sigma=population_scale, dims="predictors")
            component_idx += p

            # Linear predictor starts with population-level effects
            eta = b0 + pm.math.dot(X, beta)

            # Group-specific effects for each grouping factor
            group_effects = {}
            for group_name, group_ids in groups.items():
                n_levels = group_sizes[group_name]

                # Group-specific intercept
                group_intercept_scale = pm.Deterministic(
                    f"{group_name}_intercept_scale",
                    pt.sqrt(sigma_squared * phi[component_idx] * tau_squared)
                )

                group_intercepts = pm.Normal(
                    f"{group_name}_intercepts",
                    mu=0,
                    sigma=group_intercept_scale,
                    dims=f"{group_name}_levels"
                )
                component_idx += 1

                # Add group-specific intercepts to linear predictor
                eta += group_intercepts[group_ids]

                # Group-specific slopes (one per predictor)
                group_slopes = []
                for i in range(p):
                    group_slope_scale = pm.Deterministic(
                        f"{group_name}_slope_{i}_scale",
                        pt.sqrt((sigma_squared / sigma_x[i]**2) * phi[component_idx] * tau_squared)
                    )

                    group_slope = pm.Normal(
                        f"{group_name}_slope_{i}",
                        mu=0,
                        sigma=group_slope_scale,
                        dims=f"{group_name}_levels"
                    )

                    # Add group-specific slopes to linear predictor
                    eta += X[:, i] * group_slope[group_ids]
                    component_idx += 1
                    group_slopes.append(group_slope)

                group_effects[group_name] = {
                    'intercepts': group_intercepts,
                    'slopes': group_slopes
                }

            # Likelihood
            likelihood = pm.Normal("y", mu=eta, sigma=sigma, observed=y, dims="obs")

            # Derived quantities for interpretation
            total_variance = pm.Deterministic("total_variance", pm.math.var(eta) + sigma_squared)
            explained_variance = pm.Deterministic("explained_variance", pm.math.var(eta))

            # Variance decomposition (approximate)
            population_var_contrib = pm.Deterministic("population_var_contrib",
                                                     pt.sum(phi[:p] * tau_squared * sigma_squared))

            if groups:
                group_var_contrib = pm.Deterministic("group_var_contrib",
                                                    pt.sum(phi[p:] * tau_squared * sigma_squared))

        return model

    def r2d2m2_simple_group_intercepts(X, y, groups, r2_mean=0.5, r2_precision=2, concentration=0.5):
        """
        Simplified R2D2M2 with only group-specific intercepts (no group-specific slopes).
        More interpretable and computationally efficient.
        """
        n, p = X.shape
        n_levels = len(np.unique(groups))

        # Components: p population effects + 1 group-specific intercept
        n_components = p + 1

        # Define coordinates for model dimensions
        coords = {
            'predictors': [f'x_{i}' for i in range(p)],
            'obs': range(n),
            'groups': range(n_levels),
            'components': [f'population_{i}' for i in range(p)] + ['group_intercepts'],
        }

        with pm.Model(coords=coords) as model:
            # Intercept
            b0 = pm.Normal("b0", mu=np.mean(y), sigma=np.std(y))

            # R² prior
            alpha_r2 = r2_mean * r2_precision
            beta_r2 = (1 - r2_mean) * r2_precision
            r_squared = pm.Beta("r_squared", alpha=alpha_r2, beta=beta_r2)

            # Total explained variance
            tau_squared = pm.Deterministic("tau_squared", r_squared / (1 - r_squared))

            # Variance allocation
            phi = pm.Dirichlet("phi", a=np.full(n_components, concentration), dims="components")

            # Residual variance
            sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(y))
            sigma_squared = pm.Deterministic("sigma_squared", sigma**2)

            # Population-level effects (first p components of phi)
            population_scale = pm.Deterministic("population_scale",
                                               pt.sqrt(sigma_squared * phi[:p] * tau_squared),
                                               dims="predictors")
            beta = pm.Normal("beta", mu=0, sigma=population_scale, dims="predictors")

            # Group-specific intercepts (last component of phi)
            group_intercept_scale = pm.Deterministic("group_intercept_scale",
                                                    pt.sqrt(sigma_squared * phi[p] * tau_squared))
            group_intercepts = pm.Normal("group_intercepts", mu=0, sigma=group_intercept_scale, dims="groups")

            # Linear predictor
            eta = b0 + pm.math.dot(X, beta) + group_intercepts[groups]

            # Likelihood
            likelihood = pm.Normal("y", mu=eta, sigma=sigma, observed=y, dims="obs")

            # Variance decomposition
            population_variance = pm.Deterministic("population_variance",
                                                  phi[:p] * tau_squared * sigma_squared,
                                                  dims="predictors")
            group_variance = pm.Deterministic("group_variance",
                                             phi[p] * tau_squared * sigma_squared)

        return model

    # Example usage
    if __name__ == "__main__":
        # Generate synthetic multilevel data
        np.random.seed(42)
        n_schools = 20
        n_students_per_school = 15
        n = n_schools * n_students_per_school
        p = 3

        # School-specific intercepts
        school_intercepts = np.random.normal(0, 0.5, n_schools)

        # Student-level data
        schools = np.repeat(np.arange(n_schools), n_students_per_school)
        X = np.random.randn(n, p)

        # True population-level coefficients
        beta_true = np.array([1.0, -0.5, 0.3])

        # Generate responses
        eta_true = X @ beta_true + school_intercepts[schools]
        y = eta_true + np.random.normal(0, 0.8, n)

        print("Fitting R2D2M2 model with group-specific intercepts...")
        model = r2d2m2_simple_group_intercepts(
            X, y, schools,
            r2_mean=0.6,  # Expect moderate fit
            r2_precision=3,  # Moderately confident
            concentration=0.3  # Favor sparsity
        )

        with model:
            # Prior predictive check
            prior_pred = pm.sample_prior_predictive(samples=1000, random_seed=123)
            print(f"Prior R² mean: {prior_pred.prior['r_squared'].mean():.3f}")
            print(f"Prior R² std: {prior_pred.prior['r_squared'].std():.3f}")

            # Check variance allocation
            phi_prior = prior_pred.prior['phi']
            print(f"Prior population effects allocation: {phi_prior[:, :p].mean(axis=0)}")
            print(f"Prior group effects allocation: {phi_prior[:, p].mean():.3f}")

            # Posterior sampling would go here
            # trace = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=123)

        print("Model created successfully!")
        print(f"Model has {len(model.basic_RVs)} random variables")
    ```

    ## Key Features and Advantages

    ### 1. **Joint Regularization**
    - All variance components compete for the total explained variance `τ²`
    - Prevents unrealistic concentration of R² prior near 1
    - Natural shrinkage that scales appropriately with model complexity

    ### 2. **Interpretable Hyperparameters**
    - `r2_mean`: Expected proportion of variance explained
    - `r2_precision`: Confidence in the R² expectation
    - `concentration`: Controls sparsity (smaller values → more sparse)

    ### 3. **Scale Invariance**
    - Standardization factor `σ²/σ²ₓᵢ` makes priors invariant to scaling
    - Comparable shrinkage across predictors with different scales
    - Robust to preprocessing choices

    ### 4. **Flexible Structure**
    - Supports multiple grouping factors
    - Allows both varying intercepts and slopes
    - Can handle unbalanced designs

    ### 5. **Computational Benefits**
    - Normal base distributions enable efficient sampling
    - Compatible with gradient-based MCMC (HMC/NUTS)
    - Closed-form expressions for many quantities

    ## Implementation Notes

    ### Hyperparameter Selection

    **R² Prior:**
    - `r2_mean = 0.5`: Moderate fit expectation (reasonable default)
    - `r2_precision = 2-5`: Balance between informativeness and flexibility
    - Higher precision → more concentrated around mean

    **Dirichlet Concentration:**
    - `concentration < 1`: Promotes sparsity (bathtub distribution)
    - `concentration = 1`: Uniform over simplex
    - `concentration > 1`: Favors equal allocation

    ### Model Variants

    1. **Group-Specific Intercepts Only**: Most common, computationally efficient
    2. **Group-Specific Slopes**: Allows effect heterogeneity across groups
    3. **Multiple Grouping Factors**: Crossed or nested structures
    4. **Partial Pooling**: Automatic via hierarchical structure

    ### Laboratory Data Example: Multiple Grouping Factors

    **Scenario**: Laboratory experiment with multiple crossed grouping factors

    **Data structure:**
    - **Predictors (p=4)**: Gene expression, Age, Body weight, Treatment dose
    - **Grouping factors**:
      - Mouse ID (individual mice)
      - MicroRNA ID (different microRNAs)
      - Stress condition (Control, Heat, Cold)

    **Component calculation (intercepts only):**
    ```python
    n_components = p + n_grouping_factors  # 4 + 3 = 7 components

    component_names = [
        'population_gene_expr',      # β₁
        'population_age',            # β₂
        'population_weight',         # β₃
        'population_dose',           # β₄
        'mouse_intercepts',          # Different baseline per mouse
        'microRNA_intercepts',       # Different baseline per microRNA
        'stress_intercepts'          # Different baseline per stress condition
    ]
    ```

    **Implementation:**
    ```python
    def r2d2m2_laboratory_data(X, y, mouse_ids, microRNA_ids, stress_conditions,
                              r2_mean=0.5, r2_precision=2, concentration=0.5):
        n, p = X.shape

        # Group information
        n_mice = len(np.unique(mouse_ids))
        n_microRNAs = len(np.unique(microRNA_ids))
        n_stress_conditions = len(np.unique(stress_conditions))

        # Components: p population + 3 group-intercept types
        n_components = p + 3

        coords = {
            'predictors': [f'x_{i}' for i in range(p)],
            'obs': range(n),
            'components': [f'population_{i}' for i in range(p)] +
                         ['mouse_intercepts', 'microRNA_intercepts', 'stress_intercepts'],
            'mice': range(n_mice),
            'microRNAs': range(n_microRNAs),
            'stress_conditions': range(n_stress_conditions)
        }

        with pm.Model(coords=coords) as model:
            # R² prior
            alpha_r2 = r2_mean * r2_precision
            beta_r2 = (1 - r2_mean) * r2_precision
            r_squared = pm.Beta("r_squared", alpha=alpha_r2, beta=beta_r2)

            # Total explained variance
            tau_squared = pm.Deterministic("tau_squared", r_squared / (1 - r_squared))

            # Variance allocation across all component types
            phi = pm.Dirichlet("phi", a=np.full(n_components, concentration), dims="components")

            # Residual variance
            sigma = pm.HalfStudentT("sigma", nu=3, sigma=np.std(y))
            sigma_squared = pm.Deterministic("sigma_squared", sigma**2)

            # Population-level effects (first p components)
            population_scale = pm.Deterministic(
                "population_scale",
                pt.sqrt(sigma_squared * phi[:p] * tau_squared),
                dims="predictors"
            )
            beta = pm.Normal("beta", mu=0, sigma=population_scale, dims="predictors")

            # Mouse-specific intercepts (component p)
            mouse_scale = pm.Deterministic(
                "mouse_scale",
                pt.sqrt(sigma_squared * phi[p] * tau_squared)
            )
            mouse_intercepts = pm.Normal("mouse_intercepts", mu=0, sigma=mouse_scale, dims="mice")

            # MicroRNA-specific intercepts (component p+1)
            microRNA_scale = pm.Deterministic(
                "microRNA_scale",
                pt.sqrt(sigma_squared * phi[p+1] * tau_squared)
            )
            microRNA_intercepts = pm.Normal("microRNA_intercepts", mu=0, sigma=microRNA_scale, dims="microRNAs")

            # Stress-condition-specific intercepts (component p+2)
            stress_scale = pm.Deterministic(
                "stress_scale",
                pt.sqrt(sigma_squared * phi[p+2] * tau_squared)
            )
            stress_intercepts = pm.Normal("stress_intercepts", mu=0, sigma=stress_scale, dims="stress_conditions")

            # Linear predictor with all group effects
            eta = (pm.math.dot(X, beta) +
                   mouse_intercepts[mouse_ids] +
                   microRNA_intercepts[microRNA_ids] +
                   stress_intercepts[stress_conditions])

            # Likelihood
            likelihood = pm.Normal("y", mu=eta, sigma=sigma, observed=y, dims="obs")

        return model
    ```

    **Interpretation of φ components:**
    - `φ[0:4]`: How much variance each population-level predictor gets
    - `φ[4]`: How much variance mouse-to-mouse differences get
    - `φ[5]`: How much variance microRNA-to-microRNA differences get
    - `φ[6]`: How much variance stress-condition differences get

    **Example result**: If `φ = [0.2, 0.1, 0.3, 0.1, 0.15, 0.1, 0.05]`, then:
    - Gene expression accounts for 20% of total explained variance
    - Mouse differences account for 15% of explained variance
    - MicroRNA differences account for 10% of explained variance
    - Stress condition differences account for 5% of explained variance

    **Key benefits for laboratory data:**
    1. **Automatic factor importance**: Discover which grouping factors matter most
    2. **Controlled competition**: Factors compete for explained variance
    3. **Hierarchical pooling**: Mice with few observations borrow strength
    4. **Interpretable**: Direct control over total R² while decomposing sources

    ### Computational Considerations

    - **Standardization**: Predictors can be standardized for interpretability, but it's not required
    - **Initialization**: May require careful initialization for complex models
    - **Sampling**: Use target_accept ≥ 0.9 for better exploration
    - **Identification**: Group-specific intercepts and overall intercept can be confounded

    This framework provides a principled approach to regularization in multilevel models while maintaining the interpretability advantages of R² specification and avoiding the pitfalls of independent coefficient priors.
        '''

    return (read_r2d2m2_summary,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def stats_agent_persona():
        """Your role is a statistical consultant. Your mission is to provide statistical advice.

        Be a coach, ask questions if you do not have enough information to respond,
        and guide the user toward sound statistical practices.
        """

    return


@app.cell
def _(extract_statistical_info, lmb, read_r2d2m2_summary):
    bot = lmb.AgentBot(
        # =stats_agent_persona(),
        memory=lmb.ChatMemory(),
        tools=[
            extract_statistical_info,
            read_r2d2m2_summary,
        ],
        model_name="ollama_chat/qwen3:30b",
    )
    return (bot,)


@app.cell
def _(bot):
    import marimo as mo

    _ = bot("Can you check if my experimental design makes sense?")
    mo.md(_.content)
    return (mo,)


@app.cell
def _(bot, mo):
    _ = bot(
        "A three–year in–situ field experiment was conducted in Zhongjiang County, Deyang City, Sichuan Province (31.03 °N, 104.68 °E) from 2017 to 2019 maize growing seasons. Before sowing, soil samples were collected from the 0–30cm soil layer using the diagonal soil sampling method. Soil samples were air–dried and sieved to remove undecomposed plant material. The samples obtained were used for the analysis of soil organic matter, total N, Alkaline N, Available P, Available K, pH and the results were shown in Table 1. It was a fallow land in the winter of 2017–2018, and wheat was firstly planted before sowing maize in 2019, resulting in lower soil N content. The experimental materials were the pre–screened low N–tolerant maize variety ZhengHong311 (ZH311) and low N–sensitive maize variety XianYu508 (XY508) (Li et al., 2020, Wu et al., 2022). Seeds were provided by Sichuan Agricultural University Zhenghong Seed Co. and Tieling Pioneer Seed Research Co. respectively."
    )
    mo.md(_.content)
    return


@app.cell
def _(bot, mo):
    _ = bot("How should I do the statistical analysis?")
    mo.md(_.content)
    return


@app.cell
def _(bot, mo):
    _ = bot("How would you write the R2D2 model for this experiment?")
    mo.md(_.content)
    return


@app.cell
def _():
    return


@app.cell
def _(bot, mo):
    mo.mermaid(bot.memory.to_mermaid())
    return


@app.cell
def _(bot):
    print(bot.memory.to_mermaid())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    # def model(messages, config):
    #     # Each message has a `content` attribute, as well as a `role`
    #     # attribute ("user", "system", "assistant");
    #     response = bot(messages[-1].content)
    #     return response.content
    return


@app.cell(hide_code=True)
def _():
    example_prompts = [
        # Testing treatments
        "I want to test 3 different drugs on my cells. How should I set up this experiment?",
        "I'm comparing 2 treatments on mice. What's the best way to design this study?",
        # Sample size concerns
        "I have 24 mice and 3 treatments. Is this enough?",
        "How many replicates do I need for each treatment group?",
        # Practical organization
        "How should I organize my treatments to avoid bias?",
        "I'm worried about batch effects. How should I set up my experiment?",
        # Real problems
        "What if I can't get equal numbers in each group?",
        "What if some of my samples fail or get contaminated?",
        # Design validation
        "Is my current experimental setup good enough to answer my question?",
        "Can you check if my experimental design makes sense?",
        # Randomization (common confusion)
        "How do I randomize my samples properly?",
        "What's the difference between technical and biological replicates?",
        # Multiple factors
        "I want to test both drug dose and treatment duration. How should I design this?",
        "Should I test all my treatments at once or in separate experiments?",
        # Budget/resource constraints
        "I only have 20 samples available. What's the best way to allocate them?",
        "How can I make my experiment more efficient with limited resources?",
        # Controls and blocking
        "What controls do I need to include in my experiment?",
        "I have samples from different batches. How do I handle this in my design?",
        # Comparing treatments
        "I need to test if my new treatment works better than standard treatment. How should I design this?",
        "I'm comparing 4 different culture conditions. What's the best experimental setup?",
        # Feasibility
        "Can I test 5 treatments with only 30 samples total?",
        "I need to add another treatment group. Can I do this without redoing everything?",
        # Handling problems
        "Some of my mice died before the experiment ended. Does this ruin my design?",
        "I lost some samples during processing. What should I do?",
    ]
    return


@app.cell
def _():
    # chat = mo.ui.chat(model, prompts=example_prompts)
    # chat
    return


@app.cell
def _():
    # import marimo as mo
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
