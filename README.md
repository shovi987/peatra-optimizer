# PEATRA MVP — Streamlit + OR-Tools

An interactive research/competition prototype for peatland restoration intervention selection.

## Important data note
All locations, costs, and indicator scores in the default dataset are **synthetic illustrative inputs**. They are not field measurements from Pelalawan and must not be used as operational restoration recommendations.

## Model
For each site `i` and intervention `j`, the binary decision variable `x_ij` is 1 if selected.

Weighted benefit:
`U_ij = w_fire * fire_score + w_hydro * hydro_score + w_ecology * ecology_score`

MILP:
- Maximize `sum(U_ij * x_ij)`
- Budget: `sum(cost_ij * x_ij) <= B`
- At most one action per site: `sum_j x_ij <= 1`
- Technical feasibility: `x_ij <= a_ij`
- Binary decision: `x_ij in {0,1}`

The solver uses OR-Tools CBC. The greedy benchmark is included only as a transparent heuristic comparison.

## Run locally
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start:
   `streamlit run app.py`

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and optionally this README.
2. Open Streamlit Community Cloud and connect the GitHub repository.
3. Select `app.py` as the main file and deploy.
4. Test the public URL in an incognito/private browser window.

The public URL cannot be created from this project package alone; deployment requires the user's own hosting account and authorization.

## Before real-world use
- Replace synthetic scores/costs with verified data and document sources.
- Define a defensible normalization procedure and weighting method (e.g., expert elicitation/AHP with sensitivity analysis).
- Have peatland/hydrology experts validate indicators and feasibility rules.
- Consider spatial interactions, hydrological model outputs, uncertainty, lifecycle costs, and monitoring data.
- Do not claim field validation unless it has actually been performed.
