"""Add Evidence view to the existing NIMBO React frontend without replacing its files."""
from pathlib import Path
import shutil,sys

root=Path(__file__).resolve().parents[1]
project=Path(sys.argv[1]).expanduser().resolve() if len(sys.argv)>1 else Path.home()/"Desktop"/"NIMBO"
src=project/"app/web/src"
main=src/"main.jsx"
style=src/"style.css"
if not main.is_file() or not style.is_file():raise SystemExit("Could not find NIMBO app/web/src/main.jsx and style.css. No files changed.")
code=main.read_text(encoding='utf-8')
if 'EvidenceDashboard' in code:
 print('Evidence Dashboard is already installed. No changes made.');sys.exit(0)
anchors=["import NimboRobot from './NimboRobot';", "const [registry,setRegistry]", "<div className={'nav '+(view==='registry'?'active':'')}", "{view==='registry'?", "</section>:<>"]
for a in anchors:
 if a not in code:raise SystemExit('Unrecognized main.jsx layout at: '+a+'; no files changed. Send main.jsx for a compatible patch.')
# Import is added before a known existing import to avoid breaking module order.
code=code.replace("import NimboRobot from './NimboRobot';", "import EvidenceDashboard from './EvidenceDashboard'; import './evidence-dashboard.css';\nimport NimboRobot from './NimboRobot';",1)
nav="<div className={'nav '+(view==='evidence'?'active':'')} onClick={()=>setView('evidence')} role=\"button\">▤ &nbsp; Evidence Center</div>"
code=code.replace("<p className=\"sidebarFoot\">",nav+"<p className=\"sidebarFoot\">",1)
code=code.replace("{view==='registry'?", "{view==='evidence'?<EvidenceDashboard/>:view==='registry'?",1)
# Backup before updating anything
for f in (main,style):shutil.copy2(f,f.with_name(f.name+'.before-evidence-v2.bak'))
shutil.copy2(root/'app/web/src/EvidenceDashboard.jsx',src/'EvidenceDashboard.jsx')
shutil.copy2(root/'app/web/src/evidence-dashboard.css',src/'evidence-dashboard.css')
main.write_text(code,encoding='utf-8')
print('Installed Evidence Center in',main)
print('Backups created for main.jsx and style.css. README unchanged.')
