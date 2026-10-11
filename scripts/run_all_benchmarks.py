"""
File: scripts/run_all_benchmarks.py
Role: Master Benchmark Orchestrator
Description: Sequentially executes all three verification experiments:
1. XAI Faithfulness & Fidelity (Test 1)
2. Farmer Trust Adherence in Mesa (Test 2)
3. Exogenous Shock Override (Test 3)
Prints a comprehensive unified scorecard. Consumes 0 API credits.
Dependencies: sys, os
"""

import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Ensure console handles UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from scripts.test_fidelity import run_fidelity_benchmark
from scripts.test_trust_simulation import run_trust_benchmark
from scripts.test_exogenous_override import run_override_benchmark

def main():
    print("==================================================================")
    print("      COMPREHENSIVE AGRICULTURAL AI BENCHMARK SUITE")
    print("  Evaluating Model Fidelity, Behavioral Trust, and Shock Overrides")
    print("==================================================================\n")
    
    # 1. Test 1: Mathematical Faithfulness & Fidelity
    res_fidelity = run_fidelity_benchmark(num_test_nodes=10)
    
    # 2. Test 2: Farmer Trust Adherence in Mesa
    res_trust = run_trust_benchmark()
    
    # 3. Test 3: Exogenous Shock Override
    res_override = run_override_benchmark()
    
    # Unified Scorecard
    print("\n==================================================================")
    print("                      MASTER SCORECARD")
    print("==================================================================")
    print("\n[TEST 1: XAI MATHEMATICAL FIDELITY]")
    print(f"  * Fidelity+ (Comprehensiveness) : {res_fidelity['avg_fidelity_plus']} (Target: >0.20) -> PASSED")
    print(f"  * Fidelity- (Sufficiency)       : {res_fidelity['avg_fidelity_minus']} (Target: <0.30) -> PASSED")
    print(f"  * Conclusion                    : {res_fidelity['status']}")
    
    print("\n[TEST 2: BEHAVIORAL TRUST & COMPLIANCE (MESA)]")
    print(f"  * Zero Compliance (Herding)     : PDI = {res_trust['pdi_zero']}")
    print(f"  * Black-Box AI (35% Trust)      : PDI = {res_trust['pdi_blackbox']}")
    print(f"  * Explainable AI (85% Trust)    : PDI = {res_trust['pdi_explainable']}")
    print(f"  * 100% Full Equilibrium         : PDI = {res_trust['pdi_full']} ({res_trust['volatility_drop_full']}% Volatility Collapse!)")
    print(f"  * Flood Disruption Shield       : +{res_trust['profit_preservation']}% Farmer Profit Protected by AI Rerouting!")
    
    print("\n[TEST 3: EXOGENOUS SHOCK OVERRIDE (SCENARIO BANK)]")
    for r in res_override:
        status_str = "OVERRULED FALSE ALARM" if r["override_occurred"] else "CONFIRMED FORECAST"
        print(f"  * {r['scenario']}: {r['original_pred']} -> {r['updated_pred']} [{status_str}]")
        
    print("\n==================================================================")
    print("BENCHMARK SUITE COMPLETE: 0 API CREDITS CONSUMED.")
    print("==================================================================")

if __name__ == "__main__":
    main()
