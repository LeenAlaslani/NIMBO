from pathlib import Path

SRC = Path(__file__).resolve().parents[1]/'app/web/src'

def test_home_is_real_first_view():
    main=(SRC/'main.jsx').read_text()
    assert "useState('home')" in main
    assert '<NimboLanding' in main
    assert 'onEnter={()=>setView(\'mission\')}' in main

def test_workspace_has_no_automatic_selection():
    js=(SRC/'CompanyWorkspace.jsx').read_text()
    for marker in ["[problem,setProblem]=useState('')", "[operation,setOperation]=useState('')", "[goal,setGoal]=useState('')", "[baseline,setBaseline]=useState('')", "[candidate,setCandidate]=useState('')", "[inputMode,setInputMode]=useState('')"]:
        assert marker in js
    assert 'Select a workload first.' in js
    assert 'Select an execution path.' in js
    assert 'Choose an optimization objective.' in js

def test_landing_uses_supplied_logo_not_fake_workflow():
    text=(SRC/'NimboLanding.jsx').read_text()
    assert '/nimbo-brand-mark.png' in text
    assert 'Enter workspace' in text
    assert 'fake' not in text.lower()
