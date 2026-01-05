California Estuary Mouth Dynamics (2018–2025)

Analysis Roadmap

This README defines a low → high complexity analysis plan for exploring, validating, and communicating patterns in a statewide estuary mouth dataset. It is intended as a living reference while analyses and figures are implemented.

⸻

Dataset Summary
	•	Spatial scope: 72 estuaries across California
	•	Temporal scope: Daily, 2018–2025
	•	Primary variable: Probability of estuary mouth being open (linearly interpolated to daily cadence)
	•	Metadata available:
	•	Latitude / Longitude
	•	System_Order
	•	Estuary_Hectares

Key characteristics
	•	Consistent temporal coverage across sites
	•	Probabilistic (not binary) state representation
	•	Statewide spatial coherence

⸻

Guiding Principles
	•	Start descriptive, not algorithmic
	•	Prefer interpretable summaries over complex models
	•	Accept heterogeneity and singleton behavior
	•	Treat clustering as exploratory, not definitive
	•	Every analysis should either:
	•	validate the dataset, or
	•	reveal coherent structure, or
	•	support a physical interpretation

⸻

LEVEL 0 — Sanity & Trust Checks

Goal: Establish confidence that the dataset is internally consistent and physically plausible.

Analyses
	•	Temporal coverage per estuary
	•	Length and frequency of interpolation gaps
	•	Distribution of daily p(open) values
	•	Example time series showing stable vs transitional behavior

Figures
	•	Heatmap: estuary × time (2018–2025) colored by p(open)
	•	Histograms of daily probabilities
	•	Example estuary time series (north / central / south)

Typically shown early or placed in Supplement.

⸻

LEVEL 1 — Descriptive Statewide Patterns (Core Results)

1. Seasonality

Questions
	•	Do estuaries show seasonal opening/closing cycles?
	•	How consistent is seasonality across sites?

Analyses
	•	Monthly mean p(open) per estuary
	•	Statewide mean seasonal cycle
	•	Seasonal spread (std / IQR)

Figures
	•	Line: statewide mean ± spread by month
	•	Small multiples of representative estuaries
	•	Heatmap: estuary × month-of-year

⸻

2. Interannual Variability

Questions
	•	Which years were more open or closed statewide?
	•	Which estuaries are most variable year-to-year?

Analyses
	•	Annual mean p(open) per estuary
	•	Interannual variance by site
	•	Identification of anomalous years

Figures
	•	Time series: statewide mean openness (2018–2025)
	•	Bar/dot plots of annual openness

⸻

3. Spatial Organization

Questions
	•	Are there spatial gradients in openness behavior?
	•	Do nearby estuaries behave similarly?

Analyses
	•	Long-term mean p(open) per estuary
	•	Seasonal amplitude (max–min monthly mean)
	•	Correlation with latitude

Figures
	•	Map: mean openness
	•	Map: seasonal amplitude
	•	Scatter: latitude vs amplitude

⸻

LEVEL 2 — Event Structure & Regimes

4. Open/Closed Events

Questions
	•	How long do estuaries remain open or closed?
	•	Which systems exhibit rapid switching?

Analyses
	•	Convert probability to binary (with smoothing / hysteresis)
	•	Extract open and closed events
	•	Event duration distributions

Figures
	•	Histograms / survival curves of open durations
	•	Boxplots by estuary size or system order

⸻

5. Behavioral Regimes (Lightweight)

Goal: Describe estuary behavior using a small set of interpretable features.

Features (per estuary)
	•	Mean openness
	•	Seasonal amplitude
	•	Interannual variance
	•	Mean open duration

Analyses
	•	PCA (or similar low-dimensional projection)
	•	Manual grouping into ~3–5 regimes

Figures
	•	PCA scatter with regime labels
	•	Map colored by regime
	•	Regime-mean seasonal cycles

⸻

LEVEL 3 — Temporal Coherence (Optional)

6. Similarity & Clustering (Exploratory)

Questions
	•	Which estuaries evolve similarly in time?
	•	Are there regional response groups?

Analyses
	•	Pairwise correlation of year–month or year–season series
	•	Distance-based hierarchical clustering
	•	Accept and retain singleton sites

Figures
	•	Reordered correlation heatmap
	•	Cluster maps (non-singletons)
	•	Representative time series per cluster

Interpret cautiously: similarity ≠ shared forcing.

⸻

7. Metadata Relationships

Questions
	•	Does estuary size matter?

Analyses
	•	Correlation / regression analyses
	•	Stratified comparisons

Figures
	•	Scatter plots with trend lines
	•	Boxplots by metadata category

⸻

Publication Framing

Suggested Title

Statewide patterns and temporal regimes of California estuary mouth dynamics (2018–2025)

Core Contributions
	•	First coherent, statewide daily estuary mouth state dataset
	•	Clear seasonal and interannual structure
	•	Spatial organization and behavioral regimes
	•	Event-based interpretation bridging ML outputs to physical processes

⸻