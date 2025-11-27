# FINAL SPEC FILE THAT FIXES saved_models ISSUE

import sys
import os
from PyInstaller.utils.hooks import collect_submodules

project_path = os.path.abspath(".")

hiddenimports = (
    collect_submodules("tensorflow") +
    collect_submodules("keras") +
    collect_submodules("numpy")
)

datas = [
    (os.path.join(project_path, "templates"), "templates"),
    (os.path.join(project_path, "static"), "static"),
    (os.path.join(project_path, "saved_models"), "saved_models"),
]

a = Analysis(
    ['app.py'],
    pathex=[project_path],
    datas=datas,
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AlzheimerDetect",
    console=True,
    icon=os.path.join(project_path, "logo.ico")
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="AlzheimerDetect"
)
