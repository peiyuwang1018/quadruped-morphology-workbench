#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
    print "Project environment is missing. Follow README to restore it."
    read -r "reply?Press Return to close…"
    exit 1
fi
.venv/bin/python -m quadruped_codesign.cli panel --config configs/fusion_reference.toml
result=$?
if (( result != 0 )); then
    print "The workbench did not start normally. Share the error above with your assistant."
    read -r "reply?Press Return to close…"
fi
exit $result
