import pandas as pd
from app import default_df, score_data, solve_milp, COST, FEAS

def test_solver_respects_budget_feasibility_and_one_action_per_site():
    weights={'Fire-risk reduction':0.4,'Hydrological benefit':0.35,'Ecological benefit':0.25}
    df=score_data(default_df(),weights)
    chosen,status=solve_milp(df,100)
    assert chosen is not None
    assert status=='OPTIMAL'
    assert chosen[COST].sum() <= 100 + 1e-6
    assert not chosen['Site'].duplicated().any()
    assert chosen[FEAS].all()
