import numpy as np

def get_fuzzy_approval(credit_score, income, existing_loans):
    """
    Computes a fuzzy approval chance (0-100) and a label.
    Falls back to simple weighted scoring if skfuzzy is not installed.
    """
    try:
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        # ── 1. Define universes ──────────────────────────────────
        credit   = ctrl.Antecedent(np.arange(300, 901, 1),  'credit_score')
        income_v = ctrl.Antecedent(np.arange(0, 200001, 1), 'income')
        loans    = ctrl.Antecedent(np.arange(0, 11, 1),     'existing_loans')
        approval = ctrl.Consequent(np.arange(0, 101, 1),    'approval_chance')

        # ── 2. Membership functions ──────────────────────────────
        credit['low']    = fuzz.trimf(credit.universe,   [300, 300, 580])
        credit['medium'] = fuzz.trimf(credit.universe,   [550, 650, 750])
        credit['high']   = fuzz.trimf(credit.universe,   [700, 900, 900])

        income_v['low']    = fuzz.trimf(income_v.universe, [0,     0,     30000])
        income_v['medium'] = fuzz.trimf(income_v.universe, [20000, 45000, 70000])
        income_v['high']   = fuzz.trimf(income_v.universe, [60000, 200000, 200000])

        loans['few']  = fuzz.trimf(loans.universe, [0, 0, 2])
        loans['many'] = fuzz.trimf(loans.universe, [1, 5, 10])

        approval['low']    = fuzz.trimf(approval.universe, [0,  0,  40])
        approval['medium'] = fuzz.trimf(approval.universe, [30, 50, 70])
        approval['high']   = fuzz.trimf(approval.universe, [60, 100, 100])

        # ── 3. Fuzzy rules ───────────────────────────────────────
        rules = [
            ctrl.Rule(credit['high']   & income_v['high']   & loans['few'],  approval['high']),
            ctrl.Rule(credit['high']   & income_v['medium'] & loans['few'],  approval['high']),
            ctrl.Rule(credit['medium'] & income_v['high']   & loans['few'],  approval['high']),
            ctrl.Rule(credit['medium'] & income_v['medium'] & loans['few'],  approval['medium']),
            ctrl.Rule(credit['medium'] & income_v['low']    & loans['few'],  approval['medium']),
            ctrl.Rule(credit['high']   & income_v['low']    & loans['few'],  approval['medium']),
            ctrl.Rule(credit['medium'] & income_v['medium'] & loans['many'], approval['low']),
            ctrl.Rule(credit['low']    & income_v['high']   & loans['few'],  approval['low']),
            ctrl.Rule(credit['low']    & income_v['medium'],                 approval['low']),
            ctrl.Rule(credit['low']    & income_v['low'],                    approval['low']),
            ctrl.Rule(loans['many'],                                          approval['low']),
        ]

        # ── 4. Simulate ──────────────────────────────────────────
        system = ctrl.ControlSystem(rules)
        sim    = ctrl.ControlSystemSimulation(system)
        sim.input['credit_score']   = credit_score
        sim.input['income']         = min(income, 200000)
        sim.input['existing_loans'] = min(existing_loans, 10)
        sim.compute()
        score = round(float(sim.output['approval_chance']), 1)

    except ImportError:
        # Fallback if skfuzzy not installed
        s  = 40 * (credit_score - 300) / 600
        s += 30 * min(income, 100000) / 100000
        s += 30 * max(0, (5 - existing_loans) / 5)
        score = round(min(max(s, 0), 100), 1)

    label = "High" if score >= 65 else "Moderate" if score >= 40 else "Low"
    return score, label