# Script metadata for marimo/llamabot local dev
# Requires llamabot to be installed locally (editable install recommended)
# Requires: pip: llamabot

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "llamabot[all]>=0.17.1",
#     "polars==1.35.2",
# ]
#
# [tool.uv.sources]
# llamabot = { path = "../", editable = true }
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
    import polars as pl

    return (pl,)


@app.cell
def _(lmb):
    @lmb.prompt("system")
    def experiment_design_critique_sysprompt():
        """You are an expert statistician that will help poke holes
        in proposed experiment designs to help make them robust.

        You will be provided with a proposed experiment design.
        Your task is to identify potential flaws, biases, or weaknesses in the design,
        and to ask questions back of the propose to clarify any ambiguous aspects.
        If you are given back a list of things to talk through, go through them one by one.
        """

    return (experiment_design_critique_sysprompt,)


@app.cell
def _(experiment_design_critique_sysprompt, lmb):
    @lmb.nodeify(loopback_name="decide")
    @lmb.tool
    def critique_experiment_design(design: str) -> str:
        """Critique an experiment design and identify potential flaws, biases, or weaknesses.

        :param design: Description of the proposed experiment design
        :return: Critique of the experiment design with identified issues and questions, as well as suggestions for improvement.
        """
        bot = lmb.SimpleBot(system_prompt=experiment_design_critique_sysprompt())
        return bot(design)

    return (critique_experiment_design,)


@app.cell
def _(mo):
    files = mo.ui.file(filetypes=[".csv"])
    return (files,)


@app.cell
def _(files):
    files
    return


@app.cell(hide_code=True)
def _():
    import io

    return (io,)


@app.cell(hide_code=True)
def _():
    from caseconverter import snakecase

    return (snakecase,)


@app.cell(hide_code=True)
def _():
    from pathlib import Path

    return (Path,)


@app.cell(hide_code=True)
def _(Path, files, io, mo, pl, snakecase):
    if files.value:
        for file in files.value:
            print(f"## Results for file: {file.name}")

            print("### Data Preview")
            df = pl.read_csv(io.BytesIO(file.contents))
            mo.ui.dataframe(df)
            variable_name = snakecase(Path(file.name).stem)
            globals()[variable_name] = df

            print(variable_name)
    return


@app.cell
def _(files, ic50_data_with_confounders):
    display_df = None
    if files.value:
        display_df = ic50_data_with_confounders

    display_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""---""")
    return


@app.cell
def _():
    from llamabot.components.tools import write_and_execute_code

    return (write_and_execute_code,)


@app.cell(hide_code=True)
def _(critique_experiment_design, lmb, write_and_execute_code):
    bot = lmb.AgentBot(
        tools=[critique_experiment_design, write_and_execute_code(globals())]
    )
    return (bot,)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    example_prompts = [
        """For workflow optimization experiments, the peptide solution was aliquoted accordingly (12.5 to 200 μg peptides), frozen at −80 °C, and dried by vacuum centrifugation. Peptides were reconstituted in 50 mM HEPES (pH 8.5) and TMTzero reagent or a mix of TMT10-plex reagents (ThermoFisher) was added from stocks dissolved in 100% anhydrous ACN. The peptide–TMT mixture was incubated for 1 h at 25 °C and 400 rpm, and the labeling reaction was stopped by addition of either 5% hydroxylamine to a final concentration of 0.4% or 8 μl of 1 M Tris, pH 8, and incubation for 15 min at 25 °C and 400 rpm. Moreover, 17 samples were labeled in three experiments as singlicates, applying different TMT (40 to 400 μg) and peptide (40 or 200 μg) quantities and concentrations to explore the impact of these parameters on labeling performance and to examine the adaptability of optimized protocol parameters to lower peptide quantities. To assess interlaboratory robustness, four labeling experiments, in which the TMT quantity was titrated (50 to 400 μg) against a constant peptide amount (100 μg), were carried out as seven replicates of which two or three were performed in three independent laboratories.""",
        """Yeast lysate was spiked into human lysate to 10% of total protein concentration (1× group), 5% (2× group), and 3.3% (3× group) for a total of 11 samples so that the ratio between the first group and second was a 2-fold change, and between the first and third group was a 3-fold change. Samples for TMT were labeled with unique isobaric tags per sample, mixed, and fractionated by basic pH reversed phase HPLC (RP-HPLC). Eleven samples from LFQ and 11 fractions of the TMT set were analyzed on the same Orbitrap Fusion Lumos mass spectrometer with online RP-nHPLC separation with 3 h gradients.""",
        """The TMT-LC/LC-MS/MS method includes (in order) protein extraction from cells and tissues, protein digestion, peptide desalting, PTM enrichment, 16-plex TMT labeling, basic pH reverse-phase (RP) LC fractionation, and acidic pH RPLC-MS/MS. The protocol described here introduces a detailed, optimized method for TMT16, including protein extraction and digestion, TMT16 labeling, sample pooling and quality control, two-dimensional reverse-phase liquid chromatography fractionation, tandem mass spectrometry analysis, and post-MS data processing.""",
        """Add 800 μg TMT16 reagent to the di-GG peptides that are enriched from 4 mg peptides and incubate at room temperature for 30 min. As global ubiquitinome varies dramatically under different stress conditions such as heat shock, proteasome inhibition, etc., the normalization of sample loading bias cannot be performed the same as for the whole proteome analysis. Here we only adjust the mixing amount for the replicates in the 16-plex samples to reduce experimental variations that occur in sample preparation steps for the ubiquitinome samples.""",
        """For efficient labeling of low concentration protein samples, we designed and optimized a double labeling technique using various initial protein concentrations (2 μg/μL, 1.5 μg/μL, 1.0 μg/μL, and 0.5 μg/μL). Briefly, TMT labeling was conducted twice before quenching; 200 μg of TMT in 10 μL ACN was incubated with the protein solutions in 100 mM TEAB buffer (pH 8.5) for 1 hour followed by a second addition of TMT reagent and incubation for an additional hour. TMT-to-protein ratio remained constant and the final reactant concentrations were varied from 10.8 to 33.8 mM (TMT) and 0.5 to 1.4 μg/μL (protein).""",
        """Healthy and diseased cells were lysed in CHAPS buffer, prepared as detailed in our TMT labeling method, and submitted to the University of Florida Interdisciplinary Center for Biotechnology Research (UF-ICBR) Proteomics Core for Liquid Chromatography with tandem mass spectrometry. Following data acquisition and delivery from the core, the dataset was opened in vendor-supplied software and the following cutoff filters were applied: ≥2 unique peptides, reporter ions for each protein sample present in all channels, and include only significantly altered proteins (p ≤ 0.05). Table 1 summarizes the data: 39,653 total peptides, of which 7,211 have equal or greater than two unique peptides, and 3,829 include reporter ions for all channels.""",
        """We implemented a revised protocol with 500 mM HEPES (pH 8.5) in place of 50 mM HEPES (pH 8.5) to resuspend peptide samples derived from cryopulverized rat kidney and lung tissue before relabeling. We used this revised protocol to label 209 samples (19 TMT-11 sets). The LE of each set was above 99.3%, indicating that the new protocol successfully safeguards the samples against poor labeling reactions. Because calculating the LE for the entire TMT experimental set does not effectively identify individual channels that might be poorly labeled, we explored several methods to assess LE at the individual channel level for each TMT set.""",
        """Mix 50 μg of protein sample produced (concentration ≥1 μg/μL) with 200 μg TMT reagent in 10 μL ACN. Incubate the sample at room temperature for 1 h, then quench the reaction by addition of 5% hydroxylamine to a final concentration of >0.3% (final pH > 9.1) for 15 min at room temperature. If the protein concentration is lower than 1 μg/μL, double labeling can be applied. After the first 1 h reaction, add the same amount of TMT reagent (200 μg in 10 μL) to the solution and incubate for 1 additional hour, then quench the reaction using hydroxylamine to a final concentration of >0.3%.""",
        """Combined trypsin-digested lysates of HeLa cells and E. coli cells were extensively separated via isoelectric focusing into 24 fractions and analyzed via LC-MS/MS in three replicates. This was repeated with the same quantity of HeLa, but admixed with a 3-fold increased amount of E. coli lysate. In the resulting six samples all human proteins therefore should have had one-to-one ratios and all E. coli proteins should have had a ratio of three to one between replicate groups. Raw data were processed with MaxQuant and its built-in Andromeda search engine for feature extraction, peptide identification, and protein inference. Transferring identifications to other LC-MS runs by matching them to unidentified features based on their masses and recalibrated retention times increased the number of quantifiable isotope patterns more than 2-fold ("match-between-runs").""",
        """We introduce an experimental and data analysis scheme based on a block design with common references within each block for enabling large-scale label-free quantification. In this scheme, a large number of samples (e.g., >100 samples) are analyzed in smaller and more manageable blocks, minimizing instrument drift and variability within individual blocks. Each designated block also contains common reference samples (e.g., controls) for normalization across all blocks. We applied this strategy to profile the biological response of human THP-1-derived macrophages to a panel of 11 engineered nanomaterials (ENMs) at two different doses. By assigning a total of 116 ENM-treated samples to six blocks (controls as common references included in each block), we were able to generate highly reproducible data.""",
        """We subjected two identical samples to the Fast-quan workflow, which allowed us to systematically evaluate different parameters that impact the sensitivity and accuracy of the workflow. Using the statistics of significant test, we unraveled the existence of substantial falsely quantified differential proteins and estimated correlation of false quantification rate and parameters that are applied in label-free quantification. We optimized the setting of parameters that may substantially minimize the rate of falsely quantified differential proteins, and further applied them on a real biological process.""",
        """Label-free bottom-up shotgun MS-based proteomics is an extremely powerful and simple tool to provide high quality quantitative analyses of the yeast proteome with only microgram amounts of total protein. The elaboration of this robust workflow for data acquisition, processing, and analysis provides unprecedented insight into the dynamics of the yeast proteome. Key factors include the choice of an appropriate method for the preparation of protein extracts, careful evaluation of the instrument design and available analytical capabilities, the choice of the quantification method (intensity-based vs. spectral count), and the proper manipulation of the selected quantification algorithm.""",
        """HeLa cells were cultured in DMEM medium supplemented with 10% fetal bovine serum at 37°C in a 5% CO2 atmosphere. For SILAC labeling, cells were passaged five times in either light medium (normal L-arginine and L-lysine) or heavy medium (13C6-arginine and 2H4-lysine, Cambridge Isotope Laboratories) to achieve complete incorporation. Control cells were maintained in light medium throughout the experiment, while treated cells were switched to heavy medium 48 hours prior to treatment. After 24 hours of treatment with 10 μM compound X, cells were harvested by trypsinization, counted, and equal numbers of cells (5 × 10^6 cells per condition) were mixed. Mixed cells were pelleted, washed with PBS, and lysed in 8 M urea, 50 mM Tris-HCl pH 8.0 buffer containing protease inhibitors. Protein concentration was determined using BCA assay, and 100 μg of protein per sample was reduced with 10 mM DTT at 56°C for 30 minutes, alkylated with 20 mM iodoacetamide at room temperature for 20 minutes in the dark, and digested with trypsin (1:50 enzyme-to-protein ratio) at 37°C overnight. Peptides were desalted using C18 StageTips and analyzed by LC-MS/MS on an Orbitrap Fusion Lumos mass spectrometer with a 120-minute gradient. Protein ratios were calculated from light-to-heavy peptide intensity ratios using MaxQuant, with proteins showing greater than 1.5-fold change and p-value < 0.05 considered significantly altered.""",
        """MCF-7 breast cancer cells were grown in RPMI-1640 medium with 10% FBS and maintained at 37°C with 5% CO2. Cells were seeded at 2 × 10^5 cells per well in 6-well plates and allowed to attach overnight. Three treatment groups were established: control (vehicle only), treatment A (1 μM tamoxifen), and treatment B (5 μM tamoxifen). After 48 hours of treatment, cells were washed twice with ice-cold PBS and lysed in RIPA buffer (50 mM Tris-HCl pH 7.4, 150 mM NaCl, 1% NP-40, 0.5% sodium deoxycholate, 0.1% SDS) containing protease and phosphatase inhibitors. Protein concentration was quantified using BCA assay, and 50 μg of protein per sample was digested with trypsin after reduction and alkylation. All control samples (n=4) were processed on day 1, followed by all treatment A samples (n=4) on day 2, and all treatment B samples (n=4) on day 3. Peptides were analyzed using data-independent acquisition (DIA) on a Q Exactive HF mass spectrometer with a 60-minute LC gradient. DIA windows were set to 20 m/z with 1 m/z overlap, covering the mass range 400-1000 m/z. Data were processed using Spectronaut version 14 with default settings and a human spectral library. Proteins with p-value < 0.05 and fold change > 1.5 compared to control were considered significant.""",
        """HEK293T cells were cultured in DMEM/F12 medium with 10% FBS and transfected with either empty vector or a plasmid encoding the protein of interest using Lipofectamine 3000 according to manufacturer's instructions. After 24 hours, cells were serum-starved for 4 hours, then stimulated with 100 ng/mL EGF for 0, 5, 15, or 30 minutes. Cells were washed with ice-cold PBS and lysed in 8 M urea, 50 mM Tris-HCl pH 8.0 buffer. Protein concentration was determined, and 2 mg of protein per sample was digested with trypsin. Phosphopeptides were enriched using TiO2 beads (GL Sciences) at a ratio of 1:4 (beads:peptides) in loading buffer (80% acetonitrile, 5% TFA, 1 M glycolic acid). Control samples (0 min time point, n=3) were processed on day 1, and all treated samples (5, 15, 30 min time points, n=3 each) were processed on day 2. After enrichment, samples were labeled with TMT10-plex reagents (Thermo Fisher Scientific), with controls receiving TMT126-128 and treated samples receiving TMT129-131, 132-134, and 135-137 for the three time points respectively. Labeled peptides were pooled, desalted, and fractionated by high-pH reverse-phase chromatography into 12 fractions using an Agilent 1260 Infinity system. Each fraction was analyzed by LC-MS/MS on an Orbitrap Fusion Lumos with a 90-minute gradient. Phosphosite quantification was performed using TMT reporter ion intensities, and sites with >2-fold change and p-value < 0.01 were considered significantly regulated.""",
        """Mouse embryonic stem cells (mESCs) were maintained on gelatin-coated plates in mESC medium containing LIF. For differentiation experiments, cells were seeded at 1 × 10^5 cells per well in 6-well plates and induced to differentiate by removing LIF and adding 1 μM retinoic acid. Cells were harvested at 0, 6, 12, 24, and 48 hours after induction, with 2 biological replicates per time point. At each time point, cells were washed with PBS, trypsinized, counted, and 1 × 10^6 cells were pelleted and snap-frozen in liquid nitrogen. All samples were stored at -80°C until processing. Protein extraction was performed by resuspending cell pellets in 8 M urea, 50 mM ammonium bicarbonate buffer and sonicating for 3 cycles of 30 seconds each. After BCA quantification, 50 μg of protein per sample was reduced, alkylated, and digested with trypsin. Peptides were desalted using C18 columns and analyzed by label-free LC-MS/MS. Samples were run in chronological order: 0h replicate 1, 0h replicate 2, 6h replicate 1, 6h replicate 2, 12h replicate 1, 12h replicate 2, 24h replicate 1, 24h replicate 2, 48h replicate 1, 48h replicate 2. LC-MS/MS analysis was performed on an Orbitrap Exploris 480 with a 90-minute gradient (5-35% acetonitrile in 0.1% formic acid). MaxQuant version 1.6.14 was used for protein identification and quantification with match-between-runs enabled, and proteins identified in at least 70% of samples were retained for analysis.""",
        """Human liver tissue samples were obtained from the tissue bank at University Hospital, with 8 samples from patients with hepatocellular carcinoma (HCC) and 8 samples from adjacent normal tissue from the same patients. Tissue samples were flash-frozen in liquid nitrogen within 30 minutes of resection and stored at -80°C. For protein extraction, approximately 50 mg of frozen tissue was pulverized using a mortar and pestle under liquid nitrogen, then homogenized in RIPA buffer using a Dounce homogenizer. Protein concentration was determined using BCA assay, and 100 μg of protein per sample was processed. All normal tissue samples were processed first (samples 1-8), followed by all HCC samples (samples 9-16) to ensure consistent handling. Proteins were reduced with 10 mM DTT, alkylated with 55 mM iodoacetamide, and digested with trypsin (Promega, sequencing grade) at 37°C for 16 hours. Each biological sample was analyzed in triplicate (3 technical replicates), resulting in 24 LC-MS/MS runs for normal tissue and 24 runs for HCC tissue. Data acquisition was performed using data-dependent acquisition (DDA) mode on an Orbitrap Fusion mass spectrometer, selecting the top 20 most intense ions for fragmentation. Protein identification was performed using Sequest against the human UniProt database (downloaded March 2020), and quantification was based on spectral counting. Proteins with at least 2 unique peptides and spectral count fold change > 2.0 between groups were considered differentially expressed.""",
        """A549 lung adenocarcinoma cells were cultured in F-12K medium with 10% FBS. For SILAC triple-labeling, cells were passaged six times in either light medium (normal arginine and lysine), medium medium (13C6-lysine), or heavy medium (13C6-15N4-arginine and 13C6-15N2-lysine, all from Cambridge Isotope Laboratories). Condition A cells (control) were maintained in light medium, condition B cells (low dose treatment) were switched to medium medium 72 hours before treatment, and condition C cells (high dose treatment) were switched to heavy medium 72 hours before treatment. After 24 hours of treatment with 0.1 μM or 1.0 μM cisplatin, cells were harvested, counted, and equal numbers of cells (4 × 10^6 per condition) were combined. Mixed cells were lysed in 8 M urea buffer, and 150 μg of total protein was processed. Proteins were reduced with 5 mM TCEP at 37°C for 30 minutes, alkylated with 10 mM iodoacetamide at room temperature for 20 minutes, and digested with trypsin (1:25 ratio) overnight at 37°C. Peptides were desalted and analyzed by LC-MS/MS on an Orbitrap Eclipse mass spectrometer with a 120-minute gradient. Protein ratios were calculated relative to the light-labeled condition using MaxQuant, and proteins showing consistent directional changes in both medium/light and heavy/light ratios with p-value < 0.05 were considered validated.""",
        """Plasma samples were collected from 50 patients diagnosed with type 2 diabetes and 50 age- and sex-matched healthy controls at the Clinical Research Center. Blood was drawn into EDTA tubes after an overnight fast, centrifuged at 2000 × g for 15 minutes at 4°C, and plasma aliquots were stored at -80°C. All patient samples were processed in batch 1 (samples 1-50), and all control samples were processed in batch 2 (samples 51-100) to maintain consistency within groups. For protein depletion, 50 μL of plasma was incubated with a Human 14 Multiple Affinity Removal System column (Agilent) according to manufacturer's protocol to remove the 14 most abundant proteins. Depleted plasma was concentrated using 3 kDa molecular weight cutoff filters, and protein concentration was determined using BCA assay. A total of 50 μg of depleted protein per sample was reduced, alkylated, and digested with trypsin. Each sample was analyzed once by label-free LC-MS/MS on an Orbitrap Exploris 480 with a 120-minute gradient. To account for batch effects, protein intensities were normalized separately within each batch using quantile normalization before statistical comparison between groups. Data were processed using MaxQuant, and proteins with fold change > 2.0, p-value < 0.01, and present in at least 80% of samples within each group were considered potential biomarkers.""",
        """Jurkat T cells were cultured in RPMI-1640 medium with 10% FBS and treated with 10 ng/mL PMA and 500 ng/mL ionomycin to induce activation. Cells were harvested at 0, 2, 6, 12, and 24 hours after activation, with 2 biological replicates per time point. Due to limited cell numbers, we were able to process only one complete TMT11-plex set containing all time points, without technical replicates. At each time point, 5 × 10^6 cells were pelleted, washed with PBS, and lysed in 8 M urea buffer. After protein quantification, 50 μg of protein per sample was processed. Proteins were reduced with 10 mM DTT, alkylated with 20 mM iodoacetamide, and digested with trypsin. Peptides were labeled with TMT11-plex reagents such that time point 0 received TMT126-127, time point 2h received TMT128N-129N, time point 6h received TMT130N-131, time point 12h received TMT132N-133, and time point 24h received TMT134-135. Labeled peptides were pooled in equal amounts, desalted, and fractionated into 10 fractions using high-pH reverse-phase chromatography. Each fraction was analyzed by LC-MS/MS on an Orbitrap Fusion Lumos with a 90-minute gradient. Protein quantification was based on TMT reporter ion intensities measured in MS2 spectra. Missing values, which occurred in 15% of protein measurements, were imputed using the minimum value observed for each protein across all channels, and proteins with >1.5-fold change between any two time points were considered temporally regulated.""",
    ]
    return (example_prompts,)


@app.cell(hide_code=True)
def _(bot):
    def chat_turn(messages, config):
        user_message = messages[-1].content
        print(user_message)
        result = bot(user_message, globals())
        return result

    return (chat_turn,)


@app.cell
def _(chat_turn, example_prompts, mo):
    chat = mo.ui.chat(chat_turn, max_height=600, prompts=example_prompts)
    return (chat,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("# Statistics Agent"),
            mo.md(
                "Chat with a statistics agent for a first-pass critique of your experiment designs before bringing it to a human statistician. Paste in your experiment design written up as a methods paragraph in a paper or grant proposal, and the bot will help identify potential flaws, biases, or weaknesses in the design."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(chat, mo):
    ui = mo.vstack(
        [
            chat,
        ]
    )
    ui
    return


@app.cell
def _(bot):
    bot
    return


@app.cell
def _(bot):
    bot("show me the file I just uploaded, ic50", globals())
    return


@app.cell
def _(bot):
    response = bot("groupby compound ID", globals())
    response
    return


@app.cell
def _(bot):
    bot("after grouping by compound ID, rank order by mean IC50.")
    return


@app.cell
def _(bot):
    bot(
        "with this ic50 dataset, what are we missing from an experiment design perspective?"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
